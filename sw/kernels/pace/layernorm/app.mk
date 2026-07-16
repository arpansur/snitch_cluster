# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Arpan Suravi Prasad <prasadar@iis.ee.ethz.ch>

APP              := pace_layernorm
$(APP)_BUILD_DIR ?= $(SN_ROOT)/sw/kernels/pace/layernorm/build
SRC_DIR          := $(SN_ROOT)/sw/kernels/pace/layernorm/src
SRCS             := $(SRC_DIR)/main.c

include $(SN_ROOT)/sw/kernels/pace/common.mk
