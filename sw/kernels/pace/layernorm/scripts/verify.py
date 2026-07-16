#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import sys

from snitch.util.sim.verif_utils import Verifier
from snitch.util.sim.data_utils import ctype_from_precision_t


class PaceLayernormVerifier(Verifier):

    OUTPUT_UIDS = ["ofmap", "golden"]

    def __init__(self):
        super().__init__()
        self.prec = "FP16"

    def get_actual_results(self):
        return self.get_output_from_symbol("ofmap", ctype_from_precision_t(self.prec))

    def get_expected_results(self):
        return self.get_output_from_symbol("golden", ctype_from_precision_t(self.prec))

    def check_results(self, *args):
        return super().check_results(*args, rtol=0)


if __name__ == "__main__":
    sys.exit(PaceLayernormVerifier().main())
