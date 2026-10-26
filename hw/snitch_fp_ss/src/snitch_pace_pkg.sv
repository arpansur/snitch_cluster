// Copyright 2026 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

package snitch_pace_pkg;

  /// Configuration shared by the Snitch PACE integration.
  typedef struct packed {
    logic        enable;
    logic [31:0] memory_size;
    logic [31:0] degree;
    logic [31:0] parts;
    logic [31:0] eps;
    logic [31:0] data_width;
    logic [31:0] param_width;
    logic [31:0] fmt_config;
    logic [31:0] pipe_dist;
  } pace_cfg_t;

endpackage
