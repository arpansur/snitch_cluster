#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

from snitch.util.sim.data_utils import ctype_from_precision_t
from snitch.util.sim.verif_utils import Verifier


def detect_precision(snitch_bin):
    data_h = Path(snitch_bin).resolve().with_name("data.h")
    if data_h.exists():
        text = data_h.read_text()
        if "#define ENABLE_FP16 1" in text:
            return "FP16"
        if "#define ENABLE_FP32 1" in text:
            return "FP32"
    return "FP32"


class PaceScalarVerifier(Verifier):

    OUTPUT_UIDS = ["ofmap", "golden"]

    def __init__(self):
        super().__init__()
        self.prec = detect_precision(self.args.snitch_bin)

    def get_actual_results(self):
        return self.get_output_from_symbol("ofmap", ctype_from_precision_t(self.prec))

    def get_expected_results(self):
        return self.get_output_from_symbol("golden", ctype_from_precision_t(self.prec))

    def check_results(self, *args):
        return super().check_results(*args, rtol=0)


if __name__ == "__main__":
    sys.exit(PaceScalarVerifier().main())
