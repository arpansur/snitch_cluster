#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path
import sys

import numpy as np

try:
    import json5
    _HAS_JSON5 = True
except ModuleNotFoundError:
    json5 = None
    _HAS_JSON5 = False

_REPO_ROOT = Path(__file__).resolve().parents[5]
_PACE_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
_PWPA_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "pwpa" / "scripts"
for _path in (str(_REPO_ROOT), str(_PACE_SCRIPTS_DIR), str(_PWPA_SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from snitch.pace.scripts.debug import debug_plot_pwpa, write_pwpa_debug_file
    from snitch.pace.scripts.golden import ACTIVATIONS
    from snitch.pace.scripts.invert import invert_sqrt
    from snitch.pace.pwpa.scripts.datagen import arrange_params, generate_pace_insn_defines, pace_lane_count
    from snitch.pace.scripts.pwpa import build_bst_bps, fit_pwpa, generate_bps
    from snitch.pace.layernorm.scripts.layernorm import (
        generate_hwc_input,
        golden_model,
        write_layernorm_debug_file,
    )
except ModuleNotFoundError:
    from debug import debug_plot_pwpa, write_pwpa_debug_file
    from golden import ACTIVATIONS
    from invert import invert_sqrt
    from pwpa import build_bst_bps, fit_pwpa, generate_bps
    from layernorm import generate_hwc_input, golden_model, write_layernorm_debug_file
    import importlib.util

    _pwpa_spec = importlib.util.spec_from_file_location(
        "pace_pwpa_datagen", _PWPA_SCRIPTS_DIR / "datagen.py"
    )
    _pwpa_datagen = importlib.util.module_from_spec(_pwpa_spec)
    _pwpa_spec.loader.exec_module(_pwpa_datagen)
    arrange_params = _pwpa_datagen.arrange_params
    generate_pace_insn_defines = _pwpa_datagen.generate_pace_insn_defines
    pace_lane_count = _pwpa_datagen.pace_lane_count

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


def load_config(path):
    text = path.read_text()
    if _HAS_JSON5:
        return json5.loads(text)
    filtered = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("//")
    )
    return json.loads(filtered)


def emit_header(**kwargs):
    prec = kwargs["prec"]
    numpy_type = data_utils.numpy_type_from_precision_t(prec)
    ctype = data_utils.ctype_from_precision_t(prec)
    int_type = _integer_precision_t(prec)
    hex_ctype = data_utils.hex_ctype_from_precision_t(int_type)
    param_int_type = _integer_precision_t(kwargs["super_fmt"])
    param_hex_ctype = data_utils.hex_ctype_from_precision_t(param_int_type)
    lane_count = pace_lane_count(numpy_type, kwargs.get("fpu_data_width", 64))
    channels = kwargs["channels"]
    elements_per_channel = kwargs["elements_per_channel"]
    eps = kwargs["eps"]
    seed = kwargs.get("seed", None)

    raw_bps = generate_bps(kwargs["x_min"], kwargs["x_max"], kwargs["n_part"], mode="linear")
    bst_bps = build_bst_bps(raw_bps)
    coeffs = fit_pwpa(raw_bps, degree=kwargs["n_deg"], func=ACTIVATIONS["rsqrt"])
    params = arrange_params(
        numpy_type,
        bst_bps[2:],
        coeffs,
        eps,
        ACTIVATIONS["rsqrt"](eps),
        super_fmt=kwargs["super_fmt"],
    )
    params = np.asarray(params, dtype=np.uint32)

    ifmap_hwc = generate_hwc_input(elements_per_channel, channels, seed)
    ofmap_hwc, sigma2, rsqrt_sigma2, layernorm_traces = golden_model(
        ifmap_hwc, coeffs, raw_bps, bst_bps, kwargs["n_deg"], eps, prec
    )

    debug_plot_pwpa(
        sigma2.flatten().astype(np.float32),
        ACTIVATIONS["rsqrt"](sigma2.flatten()).astype(np.float32),
        rsqrt_sigma2.astype(np.float32),
        kwargs["debug_plot"],
    )
    write_pwpa_debug_file(
        kwargs["debug_fname"], raw_bps, bst_bps, coeffs, [], prec=numpy_type
    )
    write_layernorm_debug_file(kwargs["debug_fname"], layernorm_traces)

    input_flat = ifmap_hwc.reshape(-1)
    output_flat = np.zeros_like(input_flat)
    golden_flat = ofmap_hwc.reshape(-1)

    data_str = [emit_license(), "#include <stdint.h>", ""]
    data_str += generate_pace_insn_defines(kwargs)
    data_str += [f"#define PACE_DEGREE {kwargs['n_deg']}"]
    data_str += [f"#define ENABLE_{prec} 1"]
    data_str += [f"#define FPU_DATA_WIDTH {kwargs.get('fpu_data_width', 64)}"]
    data_str += [f"#define PACE_LANES {lane_count}"]
    data_str += [f"#define CHANNELS {channels}"]
    data_str += [f"#define ELEMENTS_PER_CHANNEL {elements_per_channel}"]
    data_str += [f"#define INPUTS_LEN {input_flat.size}"]
    data_str += [f"#define PARAMS_LEN {len(params)}"]
    data_str += [f"typedef {hex_ctype} data_t;"]
    data_str += [f"typedef {param_hex_ctype} param_t;"]
    data_str += [
        format_array_declaration(f"extern {hex_ctype}", "ifmap", input_flat.shape, alignment=4096)
    ]
    data_str += [
        format_array_declaration(f"extern {hex_ctype}", "ofmap", output_flat.shape, alignment=4096)
    ]
    data_str += [format_array_definition(param_hex_ctype, "params", params, alignment=64, hex_format=True)]
    data_str += [format_array_definition(ctype, "ifmap", input_flat, alignment=4096, hex_format=True)]
    data_str += [format_array_definition(ctype, "ofmap", output_flat, alignment=4096, hex_format=True)]
    data_str += [format_array_definition(ctype, "golden", golden_flat, alignment=4096, hex_format=True)]
    return "\n\n".join(data_str)


def main():
    parser = argparse.ArgumentParser(description="Generate data for PACE layernorm kernel")
    parser.add_argument("-c", "--cfg", type=Path, required=True, help="Select param config file kernel")
    parser.add_argument("--section", type=str, help="Section to store matrices in")
    parser.add_argument("output", type=Path, help="Path of the output header file")
    args = parser.parse_args()

    param = load_config(args.cfg)
    param["section"] = args.section
    param["name"] = args.output.stem
    param["debug_fname"] = args.output.parent / param["debug_fname"]
    param["debug_plot"] = args.output.parent / "debug.pdf"

    with args.output.open("w") as f:
        f.write(emit_header(**param))


if __name__ == "__main__":
    main()
