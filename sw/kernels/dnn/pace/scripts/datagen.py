#!/usr/bin/env python3
# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Arpan Suravi Prasad <prasadar@iis.ee.ethz.ch>

import argparse
import pathlib
import json5
import torch
import numpy as np
import sys


from snitch.util.sim import data_utils
from snitch.util.sim.data_utils import _integer_precision_t, format_struct_definition, \
    format_array_definition, format_array_declaration, format_ifdef_wrapper, \
    emit_license

torch.manual_seed(42)
from golden import *
from debug import *
from pwpa import *
from invert import *


BOUNDS = {
    "silu": (None, None),
    "exp": (None, None),
    "gelu": (None, None),
    "inv": (1, 2),
    "sqrt": (1, 4),
    "rsqrt": (1, 4),
}


def arrange_params(np_type, bst_bps, coeffs, eps=10**-6, eps_const=0):
    params = []
    coeff_shape = coeffs.shape
    for deg in range(coeff_shape[1]):
        for bp in range(coeff_shape[0]):
            params.append(clean_value(np_type(coeffs[bp, coeff_shape[1]-1-deg])))
    for bp in bst_bps:
        params.append(clean_value(np_type(bp)))
    params.append(clean_value(np_type(eps)))
    params.append(clean_value(np_type(eps_const)))
    return params

def execute_pwpa(x_min, x_max, n_part, degree, n_tests, fn_name, prec, np_prec, eps, eps_const):
    bounds = BOUNDS[fn_name]
    min_bound = x_min if bounds[0] == None else bounds[0]
    max_bound = x_max if bounds[1] == None else bounds[1]
    print(min_bound, max_bound)
    raw_bps = generate_bps(min_bound, max_bound, n_part, mode="linear")
    bst_bps = build_bst_bps(raw_bps)
    coeffs  = fit_pwpa(raw_bps, degree=degree, func=ACTIVATIONS[fn_name])
    ifmap   = np.linspace(x_min, x_max, n_tests)
    np.random.shuffle(ifmap)
    ofmap_golden = golden_model(ifmap, fn_name)
    if fn_name in ["inv", "sqrt", "rsqrt"]:
        ofmap_pwpa = invert_sqrt(ifmap, coeffs, raw_bps, degree, eps=eps, eps_const=eps_const, prec=prec, fn_name=fn_name)
    else:
        part_id = compute_part_id(ifmap, raw_bps)
        ofmap_pwpa   = evaluate_pwpa(ifmap, coeffs, part_id=part_id, degree=degree, np_prec=np_prec)
    return ifmap, ofmap_golden, ofmap_pwpa, raw_bps, bst_bps, coeffs

def generate_csr_defines(keys):
    flag_keys = ["enable", "inv", "sqrt", "rsqrt", "extend", "enable_fp32"]
    data_str = []
    data_str += [f'#define ENABLE_OFFSET 4']
    data_str += [f'#define INV_OFFSET 0']
    data_str += [f'#define SQRT_OFFSET 1']
    data_str += [f'#define RSQRT_OFFSET 2']
    data_str += [f'#define EXTEND_OFFSET 3']
    for key in flag_keys:
        macro = key.upper()
        data_str.append(f'#define {macro}_VALUE {1 if keys[key] else 0}')
    data_str += [f'#define CSR_VALUE ((ENABLE_VALUE<<ENABLE_OFFSET) + (INV_VALUE<<INV_OFFSET) + (SQRT_VALUE<<SQRT_OFFSET) + (RSQRT_VALUE<<RSQRT_OFFSET) + (EXTEND_VALUE<<EXTEND_OFFSET))']

    return data_str

def emit_header(**kwargs):
    prec = kwargs['prec']
    ctype = data_utils.ctype_from_precision_t(prec)
    numpy_type   = data_utils.numpy_type_from_precision_t(prec)
    int_type = _integer_precision_t(prec)
    hex_ctype = data_utils.hex_ctype_from_precision_t(int_type)
    fn_name=kwargs["fn_name"]
    x_min  = kwargs["x_min"]
    x_max  = kwargs["x_max"]
    n_deg  = kwargs["n_deg"]
    n_part = kwargs["n_part"] 
    n_test = kwargs["n_test"] 
    fname  = kwargs["debug_fname"] 
    fplot  = kwargs["debug_plot"] 
    eps = kwargs["eps"]
    eps_const = None
    if fn_name in ["inv", "sqrt", "rsqrt"]:
        fn = ACTIVATIONS[fn_name]
        eps_const = fn(eps)
    ifmap, ofmap_golden, ofmap_pwpa, raw_bps, bst_bps, coeffs = execute_pwpa(
        x_min, x_max, n_part, n_deg, n_test, fn_name, prec=prec, np_prec=numpy_type, eps=eps, eps_const=eps_const
    )
    pwpa_traces  = debug_pwpa_list(ifmap, ofmap_golden, coeffs, bst_bps, n_deg, prec=prec, np_prec=numpy_type, fn_name=fn_name, eps=eps, eps_const=eps_const)
    debug_plot_pwpa(ifmap, ofmap_golden, ofmap_pwpa, fplot)
    write_debug_file(fname, raw_bps, bst_bps, coeffs, pwpa_traces, prec=numpy_type)

    params = arrange_params(numpy_type, bst_bps[2:], coeffs, eps, eps_const)
    params = np.asarray(params, dtype=numpy_type)
    ofmap = ofmap_pwpa.astype(numpy_type)

    ifmap_uid = 'ifmap'
    ofmap_uid = 'ofmap'
    params_uid = 'params'

    data_str = [emit_license()]

    data_str += generate_csr_defines(kwargs)
    data_str += [f'#define ENABLE_{prec} 1']
    data_str += [f'#define INPUTS_LEN {n_test}']
    data_str += [f'#define PARAMS_LEN {len(params)}']
    # Array forward declarations
    data_str += [format_array_declaration(f'extern {hex_ctype}', ifmap_uid, ifmap.shape)]
    data_str += [format_array_declaration(hex_ctype, ofmap_uid, ofmap.shape)]
    # Parameter definitions
    data_str += [format_array_definition(ctype, params_uid, params, hex_format=True)]
    # Input definitions
    data_str += [format_array_definition(ctype, ifmap_uid, ifmap, hex_format=True)]
    # Golden results for BIST
    data_str += [format_array_definition(ctype, 'golden', ofmap, hex_format=True)]
    # data_str += [format_ifdef_wrapper('BIST', result_def)]
    data_str = '\n\n'.join(data_str)

    return data_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--cfg",
        type=pathlib.Path,
        required=True,
        help='Select param config file kernel'
    )
    parser.add_argument(
        '--section',
        type=str,
        help='Section to store matrices in')
    parser.add_argument(
        'output',
        type=pathlib.Path,
        help='Path of the output header file')
    args = parser.parse_args()

    sys.path.append(args.output.parent/f"scripts")

    # Load param config file
    with args.cfg.open() as f:
        param = json5.loads(f.read())
    param['debug_fname']=args.output.parent / f"{param['debug_fname']}"
    param['debug_plot']=args.output.parent / f"debug.png"
    param['section'] = args.section
    param["name"] = args.output.stem

    # # Emit header file
    with open(args.output, 'w') as f:
        f.write(emit_header(**param))


if __name__ == '__main__':
    main()   
