#!/usr/bin/env python3
# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Arpan Suravi Prasad <prasadar@iis.ee.ethz.ch>

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
    """Map each input to the partition on its left breakpoint match.

    Breakpoints define intervals as `[bp[i], bp[i + 1]]` with equality on an
    interior breakpoint assigned to the interval on its left. For example,
    `x == bp[2]` maps to partition `1`. Values are clamped to the valid
    partition range so the first breakpoint still maps to partition `0`.
    """
    ifmap_fp64 = np.asarray(ifmap, dtype=np.float64)
    bps_fp64 = np.asarray(bps, dtype=np.float64)
    search_idx = np.searchsorted(bps_fp64, ifmap_fp64, side="right")
    part_id = search_idx - 1

    # Exact hits on interior breakpoints belong to the partition on the left.
    interior_matches = np.isin(ifmap_fp64, bps_fp64[1:-1])
    part_id = np.where(interior_matches, part_id - 1, part_id)

    return np.clip(part_id, 0, len(bps_fp64) - 2)

def compute_part_id_with_details(ifmap, bst_bps, prec):
    feat_prec = np.asarray(ifmap, dtype=prec)
    feat = feat_prec.item() if np.asarray(feat_prec).shape == () else np.asarray(feat_prec, dtype=prec).item()
    search_bps = np.asarray(bst_bps[2:], dtype=prec)
    parts = len(bst_bps) - 1
    max_stage = int(np.log2(parts))

    details = []
    idx = 0
    path_bits = []
    for stage in range(max_stage):
        bp = np.asarray(search_bps[idx], dtype=np.float64).item()
        if feat > bp:
            compare = "gt"
            bit = 1
        elif feat < bp:
            compare = "lt"
            bit = 0
        else:
            compare = "eq"
            bit = 0

        details.append(
            {
                "stage": stage,
                "bp_index": idx,
                "x": feat_prec.item() if np.asarray(feat_prec).shape == () else feat_prec,
                "bp": np.asarray(search_bps[idx], dtype=prec).item(),
                "comparison": compare,
                "bit": bit,
            }
        )
        path_bits.append(bit)
        idx = 2 * idx + 1 + bit

    part_id = int("".join(str(bit) for bit in path_bits), 2)
    details.append({"part_id": part_id})
    return part_id, details

def compute_part_id_bst(ifmap: np.ndarray, bst_bps: list, prec):
    ifmap_prec = np.asarray(ifmap, dtype=prec)
    flat_ifmap = ifmap_prec.reshape(-1)
    flat_part_ids = np.zeros(flat_ifmap.shape[0], dtype=np.int64)
    for idx, feat in enumerate(flat_ifmap):
        flat_part_ids[idx], _ = compute_part_id_with_details(feat, bst_bps, prec)
    return flat_part_ids.reshape(ifmap_prec.shape)

def evaluate_pwpa_scalar(ifmap, coeffs: np.ndarray, part_id: int, degree, np_prec, return_details=False):
    feat_prec = np.asarray(ifmap, dtype=np_prec)
    feat = np.asarray(feat_prec, dtype=np.float64).item()
    coeffs_prec = np.asarray(coeffs, dtype=np_prec)
    coeffs_fp64 = np.asarray(coeffs_prec, dtype=np.float64)
    coeffs_part = coeffs_fp64[int(part_id)]
    y = coeffs_part[degree]
    details = []

    for deg in range(degree - 1, -1, -1):
        y_before = np.asarray(y, dtype=np_prec)
        coeff_val = np.asarray(coeffs_part[deg], dtype=np_prec)
        y = y * feat + coeffs_part[deg]
        y = np.asarray(y, dtype=np_prec).astype(np.float64)
        if return_details:
            details.append(
                {
                    "step": degree - deg - 1,
                    "y_before": y_before.item(),
                    "x": feat_prec.item() if np.asarray(feat_prec).shape == () else feat_prec,
                    "coeff": coeff_val.item(),
                    "y_after": np.asarray(y, dtype=np_prec).item(),
                }
            )

    ofmap = np.asarray(y, dtype=np_prec)
    if return_details:
        return ofmap, details
    return ofmap

def evaluate_pwpa(ifmap: np.ndarray, coeffs: np.ndarray, part_id: np.ndarray, degree, np_prec):
    ifmap_prec  = np.asarray(ifmap, dtype=np_prec)
    ofmap       = np.zeros_like(ifmap, dtype=np_prec)

    for idx, feat in enumerate(ifmap_prec):
        ofmap[idx] = evaluate_pwpa_scalar(feat, coeffs, part_id[idx], degree, np_prec)
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
