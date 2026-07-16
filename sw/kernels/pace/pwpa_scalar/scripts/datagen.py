#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
from pathlib import Path


def load_pwpa_datagen():
    pwpa_datagen_path = Path(__file__).resolve().parents[2] / "pwpa" / "scripts" / "datagen.py"
    spec = importlib.util.spec_from_file_location("pace_pwpa_datagen", pwpa_datagen_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

    pwpa_datagen = load_pwpa_datagen()
    param = pwpa_datagen.load_config(args.cfg)
    param["debug_fname"] = args.output.parent / param["debug_fname"]
    param["debug_plot"] = args.output.parent / "debug.pdf"
    param["section"] = args.section
    param["name"] = args.output.stem

    with args.output.open("w") as f:
        f.write(pwpa_datagen.emit_header(**param))


if __name__ == "__main__":
    main()
