#!/usr/bin/env python3
# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Arpan Suravi Prasad <prasadar@iis.ee.ethz.ch>

import numpy as np
from golden import *
from pwpa import *

_FMT = {
    "FP32": dict(exp_bits=8,  frac_bits=23, bias=127, storage=np.float32, uint=np.uint32),
    "FP16": dict(exp_bits=5,  frac_bits=10, bias=15,  storage=np.float16, uint=np.uint16),
    "BFP16": dict(exp_bits=8,  frac_bits=7,  bias=127, storage=np.float32, uint=np.uint16),  # store as f32, bf16 frac=7, no native numpy support
}

def _view_bits(x, storage, uint):
    a = np.asarray(x, dtype=storage)
    return a.view(uint)

def decompose_normal(x, prec=None):
    """Return (sign, e_unbiased, mant) with mant in [1,2). Assert normal & finite."""
    p = _FMT[prec]
    storage, uint = p["storage"], p["uint"]
    eb, fb, bias = p["exp_bits"], p["frac_bits"], p["bias"]

    u = _view_bits(x, storage, uint)
    sign = (u >> (eb + fb)) & 0x1
    exp  = (u >> fb) & ((1 << eb) - 1)
    frac = u & ((1 << fb) - 1)

    if np.any(exp == 0) or np.any(exp == (1 << eb) - 1):
        raise ValueError("Input must be normal & finite")
    
    e_unb = exp.astype(np.int64) - bias
    mant = 1.0 + frac.astype(np.float64) / (1 << fb)

    mant = np.asarray(mant, dtype=storage)

    return sign, e_unb, mant

def compose_normal(val, k):
    """Multiply 'val' by 2**k exactly using ldexp in float64, then return float64."""
    val = np.asarray(val, dtype=np.float64)
    k   = np.asarray(k, dtype=np.int64)
    return np.ldexp(val, k)

def invert_preprocess(x, prec=None):

    sign, exp, mant = decompose_normal(x, prec=prec)
    return sign, exp, mant

def invert_postprocess(y, sign, exp):
    y = compose_normal(y, -exp)
    return np.where(sign==1, -y, y)

def eps_check(x : np.ndarray, eps, prec):
    """When input becomes subnormal, return bypass."""
    numpy_prec = _FMT[prec]["storage"]
    x_prec = x.astype(numpy_prec)
    eps_prec = numpy_prec(eps)
    return np.abs(x_prec) < eps_prec

def eps_inv(bypass, y, eps_const):
    y = np.where(
            bypass, 
            np.where(y > 0, eps_const, -eps_const),
            y
        )
    return y


def invert(x, coeffs, bps, degree, eps=1e-6, eps_const=0.0, prec=None):
    x = np.asarray(x)
    np_prec = _FMT[prec]["storage"]
    bypass = eps_check(x, eps, prec)
    sign, exp, mant = invert_preprocess(x, prec=prec)
    part_id = compute_part_id(mant, bps)
    y_inv_normal = evaluate_pwpa(mant, coeffs, part_id=part_id, degree=degree, prec=np_prec)
    y = invert_postprocess(y_inv_normal, sign, exp)
    y = eps_inv(bypass, y, eps_const)
    return y

