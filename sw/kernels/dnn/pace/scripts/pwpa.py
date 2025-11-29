#!/usr/bin/env python3
# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Arpan Suravi Prasad <prasadar@iis.ee.ethz.ch>

import torch
import numpy as np

def generate_bps(xmin, xmax, parts, mode="linear"):
    if mode == "linear":
        raw_bps_lin = np.linspace(xmin, xmax, parts + 1, dtype=np.float64)
    return raw_bps_lin

def fit_pwpa(bps, degree, func, num_samples=1000):
    num_parts = len(bps) - 1
    coeffs = np.zeros((num_parts, degree + 1))
    for part in range(num_parts):
        left = bps[part]
        right = bps[part+1]
        xs = np.linspace(left, right, num_samples)
        ys = func(xs)
        p = np.polyfit(xs, ys, deg=degree)
        coeffs[part] = p[::-1]
    return coeffs

def compute_part_id(ifmap: np.ndarray, bps: list):
    ifmap_fp64 = np.asarray(ifmap, dtype=np.float64)
    bps_fp64   = np.asarray(bps, dtype=np.float64)
    search_idx = np.searchsorted(bps_fp64, ifmap_fp64, side="left")
    part_id = search_idx - 1
    part_id[ifmap_fp64==bps_fp64[0]] = 0
    return part_id

def evaluate_pwpa(ifmap: np.ndarray, coeffs: np.ndarray, part_id: np.ndarray, degree, prec):
    ifmap_prec  = np.asarray(ifmap, dtype=prec)
    ifmap_fp64  = np.asarray(ifmap_prec, dtype=np.float64)
    coeffs_prec = np.asarray(coeffs, dtype=prec)
    coeffs_fp64 = np.asarray(coeffs_prec, dtype=np.float64)
    ofmap       = np.zeros_like(ifmap, dtype=prec)

    for idx, feat in enumerate(ifmap_fp64):
        coeffs_part = coeffs_fp64[part_id[idx]]
        y = coeffs_part[degree]
        for deg in range(degree - 1, -1, -1):
            y = y * feat + coeffs_part[deg]
            y = np.asarray(y, dtype=prec).astype(np.float64)
        ofmap[idx] = np.asarray(y, dtype=prec)
    return ofmap

def build_bst_bps(bps):
    parts = len(bps) - 1
    if parts == 1:
        return [0, 1]
    indices = list(range(1, parts + 2))  # [1, 2, .. , N+1]
    max_stage = int(np.log2(parts)) 
    layout = [indices[0] - 1, indices[-1] - 1]  # [0, 1 ... , N]
    for stage in range(max_stage + 1):
        segment_size = 1 << (max_stage - stage + 1)
        half = segment_size // 2
        i = 0
        while i + segment_size <= len(indices):
            segment = indices[i:i + segment_size]
            center = segment[half]
            layout.append(center - 1)
            i += segment_size
    bps_bst = [bps[i] for i in layout]
    return bps_bst