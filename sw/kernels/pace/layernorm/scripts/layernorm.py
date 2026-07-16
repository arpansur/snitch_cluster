#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[5]
_PACE_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
for _path in (str(_REPO_ROOT), str(_PACE_SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from snitch.pace.scripts.golden import ACTIVATIONS
    from snitch.pace.scripts.invert import invert_sqrt
    from snitch.pace.scripts.debug import debug_invsqrt
except ModuleNotFoundError:
    from golden import ACTIVATIONS
    from invert import invert_sqrt
    from debug import debug_invsqrt


def generate_hwc_input(elements_per_channel, channels, seed):
    rng = np.random.default_rng(seed)
    means = rng.uniform(-1.0, 1.0, size=elements_per_channel).astype(np.float32)
    scales = rng.uniform(1.0, 1.75, size=elements_per_channel).astype(np.float32)
    offsets = np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)
    if channels != 4:
        raise ValueError(f"This kernel assumes exactly 4 channels, got {channels}")
    hwc = means[:, None] + scales[:, None] * offsets[None, :]
    return hwc.astype(np.float16)


def float_to_hex_scalar(value, dtype):
    arr = np.asarray(value, dtype=dtype)
    if dtype == np.float16:
        return f"0x{arr.view(np.uint16).item():04X}"
    if dtype == np.float32:
        return f"0x{arr.view(np.uint32).item():08X}"
    raise ValueError(f"Unsupported dtype for hex conversion: {dtype}")


def golden_model(ifmap_hwc, coeffs, raw_bps, bst_bps, degree, eps, prec):
    elements_per_channel = ifmap_hwc.shape[0]
    channels = ifmap_hwc.shape[1]
    if channels != 4:
        raise ValueError(f"This kernel assumes exactly 4 channels, got {channels}")
    unroll = 3

    inv_n_f16 = np.float16(1.0 / elements_per_channel)
    inv_n_vec = np.full((channels,), inv_n_f16, dtype=np.float16)
    eps_const = ACTIVATIONS["rsqrt"](eps)

    x_rows = ifmap_hwc.astype(np.float16)
    scaled_rows = np.zeros_like(x_rows, dtype=np.float16)
    prod_rows = np.zeros_like(x_rows, dtype=np.float16)
    accum_slot = np.zeros((elements_per_channel, channels), dtype=np.int32)
    sum_prev_trace = np.zeros_like(x_rows, dtype=np.float16)
    sum_trace = np.zeros_like(x_rows, dtype=np.float16)
    q_prev_trace = np.zeros_like(x_rows, dtype=np.float16)
    q_trace = np.zeros_like(x_rows, dtype=np.float16)

    sum_slots = [np.zeros((channels,), dtype=np.float16) for _ in range(unroll)]
    q_slots = [np.zeros((channels,), dtype=np.float16) for _ in range(unroll)]

    main_triplets = (elements_per_channel // unroll) * unroll
    for elem_idx in range(0, main_triplets, unroll):
        for slot in range(unroll):
            idx = elem_idx + slot
            x = x_rows[idx]
            x_scaled = np.float16(x * inv_n_vec)
            x_prod = np.float16(x_scaled * x)

            scaled_rows[idx] = x_scaled
            prod_rows[idx] = x_prod
            accum_slot[idx] = slot
            sum_prev_trace[idx] = sum_slots[slot]
            q_prev_trace[idx] = q_slots[slot]
            sum_slots[slot] = np.float16(sum_slots[slot] + x)
            q_slots[slot] = np.float16(q_slots[slot] + x_prod)
            sum_trace[idx] = sum_slots[slot]
            q_trace[idx] = q_slots[slot]

    tail = elements_per_channel - main_triplets
    for slot in range(tail):
        idx = main_triplets + slot
        x = x_rows[idx]
        x_scaled = np.float16(x * inv_n_vec)
        x_prod = np.float16(x_scaled * x)

        scaled_rows[idx] = x_scaled
        prod_rows[idx] = x_prod
        accum_slot[idx] = slot
        sum_prev_trace[idx] = sum_slots[slot]
        q_prev_trace[idx] = q_slots[slot]
        sum_slots[slot] = np.float16(sum_slots[slot] + x)
        q_slots[slot] = np.float16(q_slots[slot] + x_prod)
        sum_trace[idx] = sum_slots[slot]
        q_trace[idx] = q_slots[slot]

    sum_vec = np.float16(sum_slots[0] + sum_slots[1])
    sum_vec = np.float16(sum_vec + sum_slots[2])
    q_vec = np.float16(q_slots[0] + q_slots[1])
    q_vec = np.float16(q_vec + q_slots[2])
    mean_vec = np.float16(sum_vec * inv_n_vec)
    mean_sq_vec = np.float16(mean_vec * mean_vec)
    sigma2_vec = np.float16(q_vec - mean_sq_vec)
    rsqrt_vec = np.asarray(
        invert_sqrt(
            sigma2_vec.astype(np.float16),
            coeffs,
            raw_bps,
            degree,
            eps=eps,
            eps_const=eps_const,
            fn_name="rsqrt",
            prec=prec,
        ),
        dtype=np.float16,
    )

    centered_rows = np.float16(x_rows - mean_vec)
    ofmap = np.float16(centered_rows * rsqrt_vec)

    traces = []
    for ch_idx in range(channels):
        sigma2_h = np.float16(sigma2_vec[ch_idx])
        r_h = np.float16(rsqrt_vec[ch_idx])
        _, pwpa_trace = debug_invsqrt(
            sigma2_h,
            ACTIVATIONS["rsqrt"](np.asarray([sigma2_h], dtype=np.float16))[0],
            coeffs,
            bst_bps,
            degree,
            prec,
            np.float16,
            eps,
            eps_const,
            fn_name="rsqrt",
        )
        traces.append(
            {
                "channel": ch_idx,
                "x": x_rows[:, ch_idx].copy(),
                "x_scaled": scaled_rows[:, ch_idx].copy(),
                "x_prod": prod_rows[:, ch_idx].copy(),
                "accum_slot": accum_slot[:, ch_idx].copy(),
                "sum_prev_trace": sum_prev_trace[:, ch_idx].copy(),
                "sum_trace": sum_trace[:, ch_idx].copy(),
                "q_prev_trace": q_prev_trace[:, ch_idx].copy(),
                "q_trace": q_trace[:, ch_idx].copy(),
                "sum_slot0": np.float16(sum_slots[0][ch_idx]),
                "sum_slot1": np.float16(sum_slots[1][ch_idx]),
                "sum_slot2": np.float16(sum_slots[2][ch_idx]),
                "q_slot0": np.float16(q_slots[0][ch_idx]),
                "q_slot1": np.float16(q_slots[1][ch_idx]),
                "q_slot2": np.float16(q_slots[2][ch_idx]),
                "sum": np.float16(sum_vec[ch_idx]),
                "q": np.float16(q_vec[ch_idx]),
                "mean": np.float16(mean_vec[ch_idx]),
                "mean_sq": np.float16(mean_sq_vec[ch_idx]),
                "sigma2": sigma2_h,
                "rsqrt": r_h,
                "output": ofmap[:, ch_idx].copy(),
                "pwpa_trace": pwpa_trace,
            }
        )

    return ofmap, sigma2_vec.astype(np.float16), rsqrt_vec.astype(np.float16), traces


def write_layernorm_debug_file(filename, traces):
    with open(filename, "a") as f:
        f.write("=== LAYERNORM DEBUG ===\n")
        for trace in traces:
            f.write(f"\n[channel {trace['channel']}]\n")
            f.write("ex_accumulation_trace:\n")
            for elem_idx in range(len(trace["x"])):
                x = trace["x"][elem_idx]
                sum_prev = trace["sum_prev_trace"][elem_idx]
                sum_running = trace["sum_trace"][elem_idx]
                slot = int(trace["accum_slot"][elem_idx])
                f.write(
                    f"  i={elem_idx:4d}\n"
                    f"    accum_slot = {slot}\n"
                    f"    x_i        = {float(x): .8f} ({float_to_hex_scalar(x, np.float16)})\n"
                    f"    s_prev     = {float(sum_prev): .8f} ({float_to_hex_scalar(sum_prev, np.float16)})\n"
                    f"    s_new      = s_prev + x_i\n"
                    f"               = {float(sum_prev): .8f} + {float(x): .8f}\n"
                    f"               = {float(sum_running): .8f} ({float_to_hex_scalar(sum_running, np.float16)})\n"
                )

            f.write("ex2_accumulation_trace:\n")
            for elem_idx in range(len(trace["x"])):
                x = trace["x"][elem_idx]
                x_scaled = trace["x_scaled"][elem_idx]
                x_prod = trace["x_prod"][elem_idx]
                q_prev = trace["q_prev_trace"][elem_idx]
                q_running = trace["q_trace"][elem_idx]
                slot = int(trace["accum_slot"][elem_idx])
                f.write(
                    f"  i={elem_idx:4d}\n"
                    f"    accum_slot = {slot}\n"
                    f"    x_i        = {float(x): .8f} ({float_to_hex_scalar(x, np.float16)})\n"
                    f"    inv_n      = {float(np.float16(1.0 / len(trace['x']))): .8f} "
                    f"({float_to_hex_scalar(np.float16(1.0 / len(trace['x'])), np.float16)})\n"
                    f"    x_scaled   = x_i * inv_n\n"
                    f"               = {float(x): .8f} * {float(np.float16(1.0 / len(trace['x']))): .8f}\n"
                    f"               = {float(x_scaled): .8f} ({float_to_hex_scalar(x_scaled, np.float16)})\n"
                    f"    x_prod     = x_i * x_scaled\n"
                    f"               = {float(x): .8f} * {float(x_scaled): .8f}\n"
                    f"               = {float(x_prod): .8f} ({float_to_hex_scalar(x_prod, np.float16)})\n"
                    f"    q_prev     = {float(q_prev): .8f} ({float_to_hex_scalar(q_prev, np.float16)})\n"
                    f"    q_new      = q_prev + x_prod\n"
                    f"               = {float(q_prev): .8f} + {float(x_prod): .8f}\n"
                    f"               = {float(q_running): .8f} ({float_to_hex_scalar(q_running, np.float16)})\n"
                )

            f.write(
                f"sum_slot0_float: {float(trace['sum_slot0']): .8f}\n"
                f"sum_slot0_hex:   {float_to_hex_scalar(trace['sum_slot0'], np.float16)}\n"
                f"sum_slot1_float: {float(trace['sum_slot1']): .8f}\n"
                f"sum_slot1_hex:   {float_to_hex_scalar(trace['sum_slot1'], np.float16)}\n"
                f"sum_slot2_float: {float(trace['sum_slot2']): .8f}\n"
                f"sum_slot2_hex:   {float_to_hex_scalar(trace['sum_slot2'], np.float16)}\n"
                f"sum_float: {float(trace['sum']): .8f}\n"
                f"sum_hex:   {float_to_hex_scalar(trace['sum'], np.float16)}\n"
                f"q_slot0_float:   {float(trace['q_slot0']): .8f}\n"
                f"q_slot0_hex:     {float_to_hex_scalar(trace['q_slot0'], np.float16)}\n"
                f"q_slot1_float:   {float(trace['q_slot1']): .8f}\n"
                f"q_slot1_hex:     {float_to_hex_scalar(trace['q_slot1'], np.float16)}\n"
                f"q_slot2_float:   {float(trace['q_slot2']): .8f}\n"
                f"q_slot2_hex:     {float_to_hex_scalar(trace['q_slot2'], np.float16)}\n"
                f"q_float:   {float(trace['q']): .8f}\n"
                f"q_hex:     {float_to_hex_scalar(trace['q'], np.float16)}\n"
                f"mean_float:{float(trace['mean']): .8f}\n"
                f"mean_hex:  {float_to_hex_scalar(trace['mean'], np.float16)}\n"
                f"mean_sq_float: {float(trace['mean_sq']): .8f}\n"
                f"mean_sq_hex:   {float_to_hex_scalar(trace['mean_sq'], np.float16)}\n"
                f"sigma2_float: {float(trace['sigma2']): .8f}\n"
                f"sigma2_hex:   {float_to_hex_scalar(trace['sigma2'], np.float16)}\n"
                f"rsqrt_float:  {float(trace['rsqrt']): .8f}\n"
                f"rsqrt_hex:    {float_to_hex_scalar(trace['rsqrt'], np.float16)}\n"
            )
            f.write("output_trace:\n")
            for elem_idx in range(len(trace["output"])):
                out = trace["output"][elem_idx]
                f.write(
                    f"  i={elem_idx:4d} "
                    f"y={float(out): .8f} ({float_to_hex_scalar(out, np.float16)})\n"
                )
            f.write("pwpa_rsqrt_trace:\n")
            for item in trace["pwpa_trace"]:
                if isinstance(item, list):
                    for line in item:
                        f.write(str(line))
                else:
                    f.write(str(item))
