// Copyright 2025 ETH Zurich and University of Bologna.
//
// Copyright and related rights are licensed under the Solderpad Hardware
// License, Version 0.51 (the "License"); you may not use this file except in
// compliance with the License. You may obtain a copy of the License at
// http://solderpad.org/licenses/SHL-0.51. Unless required by applicable law
// or agreed to in writing, software, hardware and materials distributed under
// this License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
//
// SPDX-License-Identifier: SHL-0.51

// Author: Arpan Suravi Prasad <prasadar@iis.ee.ethz.ch>
`include "common_cells/registers.svh"
`include "common_cells/assertions.svh"


module axi_pace_mem #(
  /// AXI4+ATOP request type. See `include/axi/typedef.svh`.
  parameter type         axi_req_t  = logic,
  /// AXI4+ATOP response type. See `include/axi/typedef.svh`.
  parameter type         axi_resp_t = logic,
  /// Address width, has to be less or equal than the width off the AXI address field.
  /// Determines the width of `mem_addr_o`. Has to be wide enough to emit the memory region
  /// which should be accessible.
  parameter int unsigned AddrWidth  = 0,
  /// AXI4+ATOP data width.
  parameter int unsigned DataWidth  = 0,
  /// AXI4+ATOP ID width.
  parameter int unsigned IdWidth    = 0,
  /// Number of banks at output, must evenly divide `DataWidth`.
  parameter int unsigned NumBanks   = 1,
  /// Depth of memory response buffer. This should be equal to the memory response latency.
  parameter int unsigned BufDepth   = 1,
  parameter int unsigned PaceDegree = 2,
  parameter int unsigned PaceParts  = 16,
  parameter int unsigned PaceEps  = 1,
  parameter int unsigned PaceDataWidth = 32,
  
  localparam int unsigned PaceBounds     = PaceParts - 1,
  localparam int unsigned PaceCoeffWidth = (PaceDegree+1)*PaceParts*PaceDataWidth,
  localparam int unsigned PaceBoundWidth = (PaceParts-1)*PaceDataWidth,
  // 2 epsilon one is to check if the input is less than epsilon and the other is to assign the eps constant that should go to the output
  localparam int unsigned PaceEpsWidth   = 2*PaceDataWidth*PaceEps,
  localparam int unsigned PaceParamWidth = PaceCoeffWidth + PaceBoundWidth + PaceEpsWidth,
  /// Dependent parameter, do not override. Memory address type.
  localparam type addr_t       = logic [AddrWidth-1:0],
  /// Dependent parameter, do not override. Memory data type.
  localparam type mem_data_t   = logic [DataWidth/NumBanks-1:0],
  /// Dependent parameter, do not override. Memory write strobe type.
  localparam type mem_strb_t   = logic [DataWidth/NumBanks/8-1:0],
  localparam type pace_param_t = logic [PaceParamWidth-1:0]
) (
  /// Clock input.
  input  logic                           clk_i,
  /// Asynchronous reset, active low.
  input  logic                           rst_ni,
  /// The unit is busy handling an AXI4+ATOP request.
  output logic                           busy_o,
  /// AXI4+ATOP slave port, request input.
  input  axi_req_t                       axi_req_i,
  /// AXI4+ATOP slave port, response output.
  output axi_resp_t                      axi_resp_o,
  output pace_param_t                    pace_param_o
);
  logic           [NumBanks-1:0]  mem_req;
  logic           [NumBanks-1:0]  mem_gnt;
  addr_t          [NumBanks-1:0]  mem_addr;
  mem_data_t      [NumBanks-1:0]  mem_wdata;
  logic           [NumBanks-1:0]  mem_we;
  logic           [NumBanks-1:0]  mem_rvalid;
  mem_data_t      [NumBanks-1:0]  mem_rdata;

  axi_to_mem #(
    .axi_req_t   ( axi_req_t  ),
    .axi_resp_t  ( axi_resp_t ),
    .AddrWidth   ( AddrWidth  ),
    .DataWidth   ( DataWidth  ),
    .IdWidth     ( IdWidth    ),
    .NumBanks    ( 32'd1      ),
    .BufDepth    ( BufDepth   ),
    .HideStrb    (  1'b0      ),
    .OutFifoDepth( 32'd1      )
  ) i_axi_to_mem (
    .clk_i,
    .rst_ni,
    .busy_o,
    .axi_req_i   ( axi_req_i  ),
    .axi_resp_o  ( axi_resp_o ),
    .mem_req_o   ( mem_req    ),
    .mem_gnt_i   ( mem_gnt    ),
    .mem_addr_o  ( mem_addr   ),
    .mem_wdata_o ( mem_wdata  ),
    .mem_strb_o  (            ),
    .mem_atop_o  (            ),
    .mem_we_o    ( mem_we     ),
    .mem_rvalid_i( mem_rvalid ),
    .mem_rdata_i ( mem_rdata  )
  );

  `FFARN(mem_rvalid, mem_we & mem_req & mem_gnt, 1'b0, clk_i, rst_ni)
  
  `ASSERT_INIT(PACE_DATA_WIDTH, PaceDataWidth == 32, "Only DataWidth=32 is supported")
  `ASSERT_INIT(PACE_EPS_VALUE, (PaceEps == 0)|(PaceEps == 1) , "Only PaceEps=0 or 1 is supported it is an enable")



  localparam int unsigned TotalWords      = (PaceParamWidth + DataWidth - 1) / DataWidth;
  localparam int unsigned MemAddrWidth    = $clog2(TotalWords);
  localparam int unsigned AddrOffset    = $clog2(DataWidth / 8);
  
  logic [2**MemAddrWidth-1:0][DataWidth-1:0] mem_content;

  assign mem_gnt = 1'b1;

  register_file_1r_1w_all #(
    .ADDR_WIDTH(MemAddrWidth),
    .DATA_WIDTH(DataWidth)
  ) i_pace_param_mem (
    .clk (clk_i),
    .ReadEnable ( 1'b0),
    .ReadAddr   ( '0),
    .ReadData   ( ),
    .WriteEnable( mem_req & mem_gnt & mem_we),
    .WriteAddr  ( mem_addr[0][AddrOffset+:MemAddrWidth]),
    .WriteData  ( mem_wdata),
    .WriteBE    ( '1),
    .MemContent ( mem_content)
  );

  localparam int unsigned DataWidthRatio = DataWidth / PaceDataWidth;

  for(genvar ii=0; ii<PaceParamWidth/PaceDataWidth; ii++) begin
    localparam int jj_rem = ii % DataWidthRatio;
    localparam int jj_quo = ii / DataWidthRatio;
    assign pace_param_o[ii*PaceDataWidth+:PaceDataWidth] = mem_content[jj_quo][jj_rem*PaceDataWidth+:PaceDataWidth]; 
  end 
endmodule