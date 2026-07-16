#!/usr/bin/env python3
# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Arpan Suravi Prasad <prasadar@iis.ee.ethz.ch>

import argparse
from pathlib import Path
import sys
import json

try:
    import json5
    _HAS_JSON5 = True
except ModuleNotFoundError:
    json5 = None
    _HAS_JSON5 = False
import numpy as np
try:
    import torch
except ModuleNotFoundError:
    torch = None

_REPO_ROOT = Path(__file__).resolve().parents[5]
_PACE_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
for _path in (str(_REPO_ROOT), str(_PACE_SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from snitch.pace.scripts.debug import (
        clean_value,
        debug_plot_pwpa,
        debug_pwpa_list,
        write_pwpa_debug_file,
    )
    from snitch.pace.scripts.golden import ACTIVATIONS, golden_model
    from snitch.pace.scripts.invert import invert_sqrt
    from snitch.pace.scripts.pwpa import build_bst_bps, compute_part_id_bst, evaluate_pwpa, fit_pwpa, generate_bps
except ModuleNotFoundError:
    from debug import clean_value, debug_plot_pwpa, debug_pwpa_list, write_pwpa_debug_file
    from golden import ACTIVATIONS, golden_model
    from invert import invert_sqrt
    from pwpa import build_bst_bps, compute_part_id_bst, evaluate_pwpa, fit_pwpa, generate_bps

try:
    from snitch.util.sim import data_utils
    from snitch.util.sim.data_utils import (
        _integer_precision_t,
        emit_license,
        format_array_declaration,
        format_array_definition,
    )
except ModuleNotFoundError:
    from util.sim import data_utils
    from util.sim.data_utils import (
        _integer_precision_t,
        emit_license,
        format_array_declaration,
        format_array_definition,
    )

BOUNDS = {
    "silu": (None, None),
    "exp": (None, None),
    "gelu": (None, None),
    "inv": (1, 2),
    "sqrt": (1, 4),
    "rsqrt": (1, 4),
}

INVERSE_FUNCTIONS = {"inv", "sqrt", "rsqrt"}
PACE_MODE_CODES = {
    "pwpa": 0b000,
    "inv": 0b001,
    "sqrt": 0b010,
    "rsqrt": 0b011,
}
PACE_SCALAR_FUNCT7 = {
    "FP32": 0x30,
    "FP16": 0x31,
    "AH": 0x32,
}
PACE_VECTOR_FUNCT3 = {
    "FP32": 0,
    "FP16": 1,
    "AH": 2,
}


def dtype_bits(dtype):
    return int(np.dtype(dtype).itemsize * 8)


def pace_lane_count(dtype, fpu_data_width):
    bits = dtype_bits(dtype)
    if fpu_data_width % bits != 0:
        raise ValueError(
            f"FPU data width {fpu_data_width} is not divisible by element width {bits}"
        )
    lanes = fpu_data_width // bits
    if lanes < 1:
        raise ValueError(
            f"Invalid lane count {lanes} for dtype={dtype} and fpu_data_width={fpu_data_width}"
        )
    return lanes


def load_config(path):
    text = path.read_text()
    if _HAS_JSON5:
        return json5.loads(text)
    filtered = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("//"))
    return json.loads(filtered)


def widen_fp_for_compare(raw, fmt):
    """
    This casting enables the reuse of a single FP32 comparator for both FP16 and FP32 operands, while incurring only minimal hardware overhead.

    The FP16 format consists of a 1-bit sign, a 5-bit exponent, and a 10-bit mantissa, whereas FP32 uses a 1-bit sign, an 8-bit exponent, and a 23-bit mantissa.
    An FP16 value is therefore encoded as:
    | s | e4 e3 e2 e1 e0 | m9 m8 m7 m6 m5 m4 m3 m2 m1 m0 |.

    To make FP16 values comparable using an FP32 comparator, the FP16 exponent is extended by padding its most significant bits with ones, and the mantissa is extended by padding with zeros.
    The resulting casted FP32-compatible representation is:
    | s | 1 1 1 e4 e3 e2 e1 e0 | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 m9 m8 m7 m6 m5 m4 m3 m2 m1 m0 |.

    This representation preserves the ordering of FP16 values under unsigned comparison, allowing correct comparisons using existing FP32 hardware without additional control logic.
    """
    raw = np.asarray(raw, dtype=fmt).view(np.uint16).astype(np.uint32)
    sign = (raw & 0x8000) >> 15
    exponent = (raw & 0x7C00) >> 10
    mantissa = raw & 0x03FF
    widened = (sign << 31) + 0x70000000 + (exponent << 23) + mantissa
    return widened.astype(np.uint32)


def widen_fp_for_fma(raw, fmt):
    raw = np.asarray(raw, dtype=fmt).view(np.uint16).astype(np.uint32)
    return raw


def arrange_params(np_type, bst_bps, coeffs, eps=10**-6, eps_const=0, super_fmt="FP32"):
    params = []
    rows, cols = coeffs.shape
    for deg in range(cols):
        coeff_idx = cols - 1 - deg
        for bp_idx in range(rows):
            coeff = coeffs[bp_idx, coeff_idx]
            if np_type == np.float16:
                if super_fmt == "FP32":
                    widened_coeff = widen_fp_for_fma(coeff, np_type)
                    params.append(int(clean_value(widened_coeff)))
                else:
                    encoded_coeff = np.asarray(coeff, dtype=np_type).view(np.uint16)
                    params.append(int(clean_value(encoded_coeff)))
            else:
                encoded_coeff = np.asarray(coeff, dtype=np_type).view(np.uint32)
                params.append(clean_value(encoded_coeff))

    for bp in bst_bps:
        if np_type == np.float16:
            if super_fmt == "FP32":
                params.append(int(clean_value(widen_fp_for_compare(bp, np_type))))
            else:
                encoded_bp = np.asarray(bp, dtype=np_type).view(np.uint16)
                params.append(int(clean_value(encoded_bp)))
        else:
            encoded_bp = np.asarray(bp, dtype=np_type).view(np.uint32)
            params.append(clean_value(encoded_bp))

    if np_type == np.float16:
        if super_fmt == "FP32":
            params.append(int(clean_value(widen_fp_for_compare(eps, np_type))))
            if eps_const is not None:
                val_eps_const = np_type(eps_const).view(np.uint16).astype(np.uint32)
                params.append(int(clean_value(val_eps_const)))
        else:
            params.append(int(clean_value(np_type(eps).view(np.uint16))))
            if eps_const is not None:
                val_eps_const = np_type(eps_const).view(np.uint16).astype(np.uint16)
                params.append(int(clean_value(val_eps_const)))
    else:
        if super_fmt == "FP32":
            eps_val = np_type(eps).view(np.uint32)
            eps_const_val = np.asarray(eps_const, dtype=np_type).view(np.uint32)
            params.append(clean_value(eps_val))
            params.append(clean_value(eps_const_val))
        else:
            eps_val = np_type(eps).view(np.uint16)
            eps_const_val = np_type(eps_const).view(np.uint16)
            params.append(clean_value(eps_val))
            params.append(clean_value(eps_const_val))
    return params


def execute_pwpa(x_min, x_max, n_part, degree, n_tests, fn_name, prec, np_prec, eps, eps_const, seed=None):
    bounds = BOUNDS[fn_name]
    min_bound = x_min if bounds[0] is None else bounds[0]
    max_bound = x_max if bounds[1] is None else bounds[1]
    raw_bps = generate_bps(min_bound, max_bound, n_part, mode="linear")
    bst_bps = build_bst_bps(raw_bps)
    coeffs = fit_pwpa(raw_bps, degree=degree, func=ACTIVATIONS[fn_name])
    ifmap = np.linspace(x_min, x_max, n_tests)
    np.random.default_rng(seed).shuffle(ifmap)
    ifmap = np.asarray(ifmap, dtype=np_prec)
    ofmap_golden = golden_model(ifmap, fn_name)
    if fn_name in INVERSE_FUNCTIONS:
        ofmap_pwpa = invert_sqrt(
            ifmap,
            coeffs,
            raw_bps,
            degree,
            eps=eps,
            eps_const=eps_const,
            prec=prec,
            fn_name=fn_name,
        )
    else:
        part_id = compute_part_id_bst(ifmap, bst_bps, np_prec)
        ofmap_pwpa = evaluate_pwpa(ifmap, coeffs, part_id=part_id, degree=degree, np_prec=np_prec)
    return ifmap, ofmap_golden, ofmap_pwpa, raw_bps, bst_bps, coeffs


def pace_dtype_key(prec, super_fmt):
    if prec == "FP32":
        return "FP32"
    if str(super_fmt).upper() in {"AH", "FP16ALT", "BF16"}:
        return "AH"
    if prec == "FP16":
        return "FP16"
    raise ValueError(f"Unsupported PACE dtype selection: prec={prec}, super_fmt={super_fmt}")


def pace_mode_name(keys):
    fn_name = keys.get("fn_name")
    mode_name = fn_name if fn_name in INVERSE_FUNCTIONS else "pwpa"

    explicit_modes = [name for name in ("inv", "sqrt", "rsqrt") if keys.get(name, False)]
    if len(explicit_modes) > 1:
        raise ValueError(
            "PACE config can enable only one of inv/sqrt/rsqrt at a time, "
            f"got {explicit_modes}"
        )
    if explicit_modes and explicit_modes[0] != mode_name:
        raise ValueError(
            f"fn_name selects '{mode_name}' but legacy mode flag selects '{explicit_modes[0]}'. "
            "Please keep them consistent or remove the legacy flag."
        )

    return mode_name


def generate_pace_insn_defines(keys):
    data_str = []
    mode_name = pace_mode_name(keys)
    mode_bits = PACE_MODE_CODES[mode_name] | (int(bool(keys["extend"])) << 2)
    dtype_key = pace_dtype_key(keys["prec"], keys["super_fmt"])
    scalar_funct7 = PACE_SCALAR_FUNCT7[dtype_key]
    vector_funct3 = PACE_VECTOR_FUNCT3[dtype_key]
    scalar_word = (
        (scalar_funct7 << 25)
        | (0 << 20)
        | (0 << 15)
        | (mode_bits << 12)
        | (1 << 7)
        | 0x53
    )
    vector_word = (
        (0xE << 28)
        | (mode_bits << 25)
        | (0 << 20)
        | (0 << 15)
        | (vector_funct3 << 12)
        | (1 << 7)
        | 0x33
    )

    data_str.append(f"#define PACE_MODE_BITS {mode_bits}")
    data_str.append(f"#define PACE_SCALAR_FUNCT7 0x{scalar_funct7:02x}")
    data_str.append(f"#define PACE_VECTOR_FUNCT3 {vector_funct3}")
    data_str.append(f"#define PACE_SCALAR_WORD 0x{scalar_word:08x}")
    data_str.append(f"#define PACE_VECTOR_WORD 0x{vector_word:08x}")
    data_str.append(f"#define PACE_SCALAR_ASM \".word 0x{scalar_word:08x}\"")
    data_str.append(f"#define PACE_VECTOR_SSR_ASM \".word 0x{vector_word:08x}\"")
    return data_str


def emit_header(**kwargs):
    prec = kwargs["prec"]
    ctype = data_utils.ctype_from_precision_t(prec)
    numpy_type = data_utils.numpy_type_from_precision_t(prec)
    fpu_data_width = kwargs.get("fpu_data_width", 64)
    lane_count = pace_lane_count(numpy_type, fpu_data_width)
    int_type = _integer_precision_t(prec)
    hex_ctype = data_utils.hex_ctype_from_precision_t(int_type)
    fn_name = kwargs["fn_name"]
    x_min = kwargs["x_min"]
    x_max = kwargs["x_max"]
    n_deg = kwargs["n_deg"]
    n_part = kwargs["n_part"]
    n_test = kwargs["n_test"]
    fname = kwargs["debug_fname"]
    fplot = kwargs["debug_plot"]
    super_fmt = kwargs["super_fmt"]
    eps = kwargs["eps"]
    seed = kwargs.get("seed")

    param_int_type = _integer_precision_t(super_fmt)
    param_hex_ctype = data_utils.hex_ctype_from_precision_t(param_int_type)

    eps_const = None
    if fn_name in INVERSE_FUNCTIONS:
        fn = ACTIVATIONS[fn_name]
        eps_const = fn(eps)

    ifmap, ofmap_golden, ofmap_pwpa, raw_bps, bst_bps, coeffs = execute_pwpa(
        x_min,
        x_max,
        n_part,
        n_deg,
        n_test,
        fn_name,
        prec=prec,
        np_prec=numpy_type,
        eps=eps,
        eps_const=eps_const,
        seed=seed,
    )
    pwpa_traces = debug_pwpa_list(
        ifmap,
        ofmap_golden,
        coeffs,
        raw_bps,
        bst_bps,
        n_deg,
        prec=prec,
        np_prec=numpy_type,
        fn_name=fn_name,
        eps=eps,
        eps_const=eps_const,
    )
    debug_plot_pwpa(ifmap, ofmap_golden, ofmap_pwpa, fplot)
    write_pwpa_debug_file(fname, raw_bps, bst_bps, coeffs, pwpa_traces, prec=numpy_type)

    params = arrange_params(numpy_type, bst_bps[2:], coeffs, eps, eps_const, super_fmt=super_fmt)
    params = np.asarray(params, dtype=np.uint32)
    ofmap = ofmap_pwpa.astype(numpy_type)
    ofmap_init = np.zeros_like(ofmap)

    ifmap_uid = "ifmap"
    ofmap_uid = "ofmap"
    params_uid = "params"

    data_str = [emit_license()]

    data_str += generate_pace_insn_defines(kwargs)
    data_str += [f"#define PACE_DEGREE {n_deg}"]
    data_str += [f"#define ENABLE_{prec} 1"]
    data_str += [f"#define FPU_DATA_WIDTH {fpu_data_width}"]
    data_str += [f"#define PACE_LANES {lane_count}"]
    data_str += [f"#define INPUTS_LEN {n_test}"]
    data_str += [f"#define PARAMS_LEN {len(params)}"]
    data_str += [f"typedef {hex_ctype} data_t;"]
    data_str += [f"typedef {param_hex_ctype} param_t;"]
    # Array forward declarations
    data_str += [format_array_declaration(f"extern {hex_ctype}", ifmap_uid, ifmap.shape, alignment=4096)]
    data_str += [format_array_declaration(f"extern {hex_ctype}", ofmap_uid, ofmap.shape, alignment=4096)]

    # Parameter definitions
    data_str += [format_array_definition(param_hex_ctype, params_uid, params, alignment=64, hex_format=True)]
    # Input definitions
    data_str += [format_array_definition(ctype, ifmap_uid, ifmap, alignment=4096, hex_format=True)]
    # Output storage populated by the kernel at runtime
    data_str += [format_array_definition(ctype, ofmap_uid, ofmap_init, alignment=4096, hex_format=True)]
    # Golden results for BIST
    data_str += [format_array_definition(ctype, "golden", ofmap, alignment=4096, hex_format=True)]
    data_str = "\n\n".join(data_str)

    return data_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--cfg",
        type=Path,
        required=True,
        help="Select param config file kernel",
    )
    parser.add_argument(
        "--section",
        type=str,
        help="Section to store matrices in",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path of the output header file",
    )
    args = parser.parse_args()

    # Load param config file
    param = load_config(args.cfg)
    param["debug_fname"] = args.output.parent / param["debug_fname"]
    param["debug_plot"] = args.output.parent / "debug.pdf"
    param["section"] = args.section
    param["name"] = args.output.stem

    with args.output.open("w") as f:
        f.write(emit_header(**param))


if __name__ == "__main__":
    main()
