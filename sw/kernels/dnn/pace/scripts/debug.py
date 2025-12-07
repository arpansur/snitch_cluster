#!/usr/bin/env python3
# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Arpan Suravi Prasad <prasadar@iis.ee.ethz.ch>

import torch
import numpy as np
import matplotlib.pyplot as plt

from invert import *

def float_to_hex(v, prec):
    if prec == np.float64:
        arr = np.asarray(v, dtype=np.float64)
        bits = arr.view(np.uint64).item()
        return f"0x{bits:016X}"    # 16 hex chars = 64 bits

    # ---- float32 (32 bits) ----
    if prec == np.float32:
        arr = np.asarray(v, dtype=np.float32)
        bits = arr.view(np.uint32).item()
        return f"0x{bits:08X}"     # 8 hex chars = 32 bits

    # ---- float16 (IEEE-754 half, 16 bits) ----
    if prec == np.float16:
        arr = np.asarray(v, dtype=np.float16)
        bits = arr.view(np.uint16).item()
        return f"0x{bits:04X}"     # 4 hex chars = 16 bits

    # ---- bfloat16 (custom) ----
    # BF16 = top 16 bits of IEEE754 float32
    if prec == "bfloat16":
        f32 = np.asarray(v, dtype=np.float32)
        bits32 = f32.view(np.uint32).item()
        bf16 = (bits32 >> 16) & 0xFFFF
        return f"0x{bf16:04X}"     # 4 hex chars = 16 bits

    raise ValueError(f"Unsupported precision: {prec}")

def debug_compute_part_id(ifmap: np.ndarray, bps: list, prec):
    ifmap_f64 = np.asarray(ifmap, dtype=np.float64)     # to float64
    ifmap_prec = ifmap_f64.astype(prec)                 # to prec
    bps_f64 = [np.float64(bp) for bp in bps[2:]]        # list of float64 scalars
    bps_prec = [prec(bp) for bp in bps_f64]             # list of prec scalars

    parts = len(bps) - 1
    max_stage = int(np.log2(parts))
    
    trace = []
    idx = 0
    path_bits = []
    feat = ifmap_prec
    for stage in range(max_stage):
        if idx >= len(bps_prec):
            raise ValueError(
                f"BST traversal out-of-bounds: stage={stage}, idx={idx}, len={len(bps_prec)}\n"
            )
        bp = bps_prec[idx]
        if feat > bp : 
            bit, decision = 1, "go_right"
        elif feat < bp : 
            bit, decision = 0, "go_left"
        else:
            bit = 0 
            decision = f"equal → {'go_left' if bit == 0 else 'go_right'}"
        path_bits.append(bit)
        trace.append(f"  Stage {stage}: x={feat:.6f} ({float_to_hex(feat, prec)}) vs bp[{idx}]={bp:.6f} ({float_to_hex(bp, prec)}) → {decision}\n")
        idx = 2 * idx + 1 + bit
    part_idx = int("".join(str(b) for b in path_bits), 2)
    trace.append(f"  Final part_idx = {part_idx}\n")

    return part_idx, trace

def debug_evaluate_pwpa(ifmap: np.ndarray, coeffs: np.ndarray, part_id: int, degree, prec):
    ifmap_prec  = np.asarray(ifmap, dtype=prec)
    ifmap_fp64  = np.asarray(ifmap_prec, dtype=np.float64)
    coeffs_prec = np.asarray(coeffs, dtype=prec)
    coeffs_fp64 = np.asarray(coeffs_prec, dtype=np.float64)
    ofmap       = np.zeros_like(ifmap, dtype=prec)

    fma_trace = []
    feat = ifmap_fp64
    count = 0 

    coeffs_part = coeffs_fp64[part_id]
    y = coeffs_part[degree]
    for idx in range(degree - 1, -1, -1):
        y_copy     = prec(y)
        feat_copy  = prec(feat)
        coeff_copy = prec(coeffs_part[idx])
        y = y * feat + coeffs_part[idx]
        y = np.asarray(y, dtype=prec).astype(np.float64)
        fma_trace.append(
            f"  FMA step {count}: y={float(y_copy):.6f}({float_to_hex(y_copy, prec)}) * "
            f"{float(feat_copy):.6f}({float_to_hex(feat_copy, prec)}) + "
            f"{float(coeff_copy):.6f}({float_to_hex(coeff_copy, prec)}) = {float(y):.6f}({float_to_hex(float(y), prec)}) \n"
        )
        count += 1
    ofmap = np.asarray(y, dtype=prec)
    return ofmap, fma_trace

def debug_plot_pwpa(ifmap : np.ndarray, golden_ofmap : np.ndarray, pwpa_ofmap : np.ndarray, fname : "debug.png"):
    plt.figure(figsize=(10, 6))
    plt.scatter(ifmap, golden_ofmap, label="GELU (golden)", color="blue", s=0.5)
    plt.scatter(ifmap, pwpa_ofmap, label="GELU (PWPA)", color="red", s=0.5, marker="d")
    plt.title("Piecewise Polynomial Approximation of GELU")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.grid()
    plt.savefig(fname)

def debug_eps_check(x : np.ndarray, eps, numpy_prec):
    """When input becomes subnormal, return bypass."""
    traces =[]
    x_prec = x.astype(numpy_prec)
    eps_prec = numpy_prec(eps)
    traces.append(f"  {np.abs(x_prec)} < {eps_prec} ?: {np.abs(x_prec) < eps_prec}\n")
    return traces, np.abs(x_prec) < eps_prec

def clean_value(v):
    """Convert NumPy scalar/array/list into plain Python floats recursively."""
    if isinstance(v, (list, tuple)):
        return [clean_value(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.astype(float).tolist()
    try:
        return float(v)
    except Exception:
        return v

def debug_pwpa(ifmap, coeffs : np.ndarray, bst_bps : list, degree : int, prec):
    pwpa_traces = []
    part_id, part_trace = debug_compute_part_id(ifmap, bst_bps, prec=prec)
    ofmap_approx, fma_trace = debug_evaluate_pwpa(ifmap, coeffs, part_id, degree, prec=prec)
    pwpa_traces.append(part_trace)
    pwpa_traces.append(fma_trace)
    return ofmap_approx, pwpa_traces

def debug_error(ofmap_golden, ofmap_approx):
    error_traces = []
    error_traces.append(f"  y_true: {ofmap_golden}\n")
    error_traces.append(f"  y_approx: {ofmap_approx}\n")
    error_traces.append(f"  error: {(float(ofmap_approx))-(float(ofmap_golden))}\n")
    return error_traces

def debug_invsqrt(ifmap, ofmap_golden, coeffs : np.ndarray, bst_bps : list, degree : int, prec, np_prec, eps, eps_const, fn_name=None):
    inv_traces = []
    eps_trace, bypass = debug_eps_check(ifmap, eps, np_prec)
    func = PRE_PROCESS[fn_name]
    sign, exp, mant = func(ifmap, prec=prec)
    ofmap_approx_mant, pwpa_trace = debug_pwpa(mant, coeffs, bst_bps, degree, np_prec)
    ofmap_approx = invert_sqrt_postprocess(ofmap_approx_mant, sign, exp)
    inv_traces.append(f"  {ifmap} decomposed to sign: {sign}, exp: {exp}, mantissa: {mant}\n")
    inv_traces.append(eps_trace)
    inv_traces.append(pwpa_trace[0])
    inv_traces.append(pwpa_trace[1])
    # inv_traces.append(pwpa_trace[1])
    inv_traces.append(f"  sign: {sign}, exp: {exp}, mantissa: {ofmap_approx_mant} composed to {ofmap_approx}\n") 
    ofmap_approx = eps_inv(bypass, ofmap_approx, eps_const)
    inv_traces.append(f"  After eps adjustment {ofmap_approx} {float_to_hex(ofmap_approx, prec=np_prec)}\n") 
    return ofmap_approx, inv_traces

def debug_pwpa_list(ifmap : np.ndarray, ofmap_golden : np.ndarray, coeffs : np.ndarray, bst_bps : list, degree : int, prec, np_prec, fn_name=None, eps=None, eps_const=None):
    pwpa_traces = []
    for i, feat in enumerate(ifmap):
        feat_process = feat
        pwpa_traces.append([f"\n*********************** Iteration: {i} ************************* \n"])
        if fn_name in ["inv", "sqrt", "rsqrt"]:
            ofmap_approx, pwpa_trace =  debug_invsqrt(feat_process, ofmap_golden[i], coeffs, bst_bps, degree, prec, np_prec, eps, eps_const, fn_name=fn_name)
        else:
            ofmap_approx, pwpa_trace =  debug_pwpa(feat_process, coeffs, bst_bps, degree, np_prec)

        error_traces = debug_error(ofmap_golden[i], ofmap_approx)
        pwpa_traces.append(pwpa_trace)   
        pwpa_traces.append(error_traces)   

    return pwpa_traces

def write_debug_file(filename, raw_bps, bst_bps, coeffs, pwpa_traces, prec):
    with open(filename, "w") as f:    
        f.write("\n=== RAW_BREAKPOINTS ===\n")
        for idx, bp in enumerate(raw_bps):
            hex_bps_prec = float_to_hex(bp, prec)
            f.write(f"bp{idx}: {bp} {hex_bps_prec}\n")     
        f.write("\n=== BST_BREAKPOINTS ===\n")
        for idx, bp in enumerate(bst_bps[2:]):
            hex_bps_prec = float_to_hex(bp, prec)
            f.write(f"bp{idx}: {bp} {hex_bps_prec}\n")    

        f.write("\n=== COEFFS ===\n")
        for idx, row in enumerate(coeffs):
            # print(idx, row)
            for coeff in row:
                hex_bps_prec = float_to_hex(coeff, prec)
                f.write(f"{coeff} {hex_bps_prec}, ")
            f.write("\n")

        f.write("=== PWPA_TRACES ===\n")
        for trace_list in pwpa_traces:
            for line in trace_list:
                f.write("".join(line))