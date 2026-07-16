# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

APP              := pwpa
$(APP)_BUILD_DIR ?= $(SN_ROOT)/sw/kernels/pace/$(APP)/build
SRC_DIR          := $(SN_ROOT)/sw/kernels/pace/$(APP)/src
SRCS             := $(SRC_DIR)/main.c

include $(SN_ROOT)/sw/kernels/pace/common.mk
