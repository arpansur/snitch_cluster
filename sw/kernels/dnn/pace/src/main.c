// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Arpan Suravi Prasad <prasadar@iis.ee.ethz.ch>

#include "math.h"
#include "snrt.h"
#include "data.h"

void pace_fp32_vector_ssr(double* inp, double* oup, int len){
  snrt_ssr_loop_1d(SNRT_SSR_DM0, len, sizeof(double));
  snrt_ssr_loop_1d(SNRT_SSR_DM1, len, sizeof(double));
  snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, inp);
  snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, oup);
  snrt_ssr_enable();
  double y;
   __asm__ volatile(
      "frep.o  %[n], 1, 0, 0\n\t"
      "vfmul.s  ft1, ft0, %[y]\n\t"
      :
      : [n] "r"(len-1), [y] "f"(y)
      : "ft1", "memory");
}

void pace_fp32_vector(double* inp, double* oup, int len){
  double op_a, op_b, op_c;
  for(int i=0; i<len; i++)
  {
    op_a = 0;
    op_b = *(inp + i);
    __asm__ ("vfmul.s %0, %1, %2"
        : "=f"(op_c)
        : "f"(op_a), "f"(op_b));
    *(oup + i) = op_c;
    }
}

void pace_fp32_scalar(float* inp, float* oup, int len){
  float op_a, op_b, op_c;
  for(int i=0; i<len; i++)
  {
    op_a = 0;
    op_b = *(inp + i);
    __asm__ ("fmul.s %0, %1, %2"
        : "=f"(op_c)
        : "f"(op_a), "f"(op_b));
    *(oup + i) = op_c;
    }
}

void pace_fp32_exec(uint32_t* inp, uint32_t* oup, int len)
{
  write_csr(0xba0, CSR_VALUE);
#ifdef PACE_VECTOR
  double* dx_p = (double *)inp;
  double* dy_p = (double *)oup;
#ifdef PACE_SSR
  pace_fp32_vector_ssr(dx_p, dy_p, len / 2);
#else 
  pace_fp32_vector_ssr(dx_p, dy_p, len / 2);
#endif 
#else
  float* fx_p = (float *)inp;
  float* fy_p = (float *)oup;
  pace_fp32_scalar(fx_p, fy_p, INPUTS_LEN);
#endif

}

int check_output(uint32_t* actual, uint32_t* golden, int len)
{
  int errors = len; 
  for (int i=0; i<len; i++){
    uint32_t actual_data = *(actual+i);
    uint32_t golden_data = *(golden+i);
    if(actual_data == golden_data)
      errors--;
    else 
      printf("idx:%d, errors=%d, actual_data=%x, golden_data=%x, actual_ptr=%x, golden_ptr=%x\n", i, errors, actual_data, golden_data, (actual+i), (golden+i));
  }
  return errors;
}

int main() {
    uint32_t *local_x, *local_y;
    uint32_t *remote_x;
    uint32_t *remote_params = params;
    uint32_t *pace_mem = (uint32_t *)snrt_cluster()->pacemem.mem;
    local_x = (uint32_t *)snrt_l1_next();
    local_y = local_x + INPUTS_LEN;
    remote_x = ifmap;
    remote_params = params;
    //////////////////
    // DMA 1D WRITE //
    //////////////////
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(pace_mem, remote_params, PARAMS_LEN * sizeof(uint32_t));
        snrt_dma_wait_all();
        snrt_dma_start_1d(local_x, remote_x, INPUTS_LEN * sizeof(uint32_t));
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();
    if (snrt_cluster_core_idx() == 0) {
      pace_fp32_exec(local_x, local_y, INPUTS_LEN);
    }
    snrt_cluster_hw_barrier();
    if (snrt_cluster_core_idx() == 0) {
      int errors = check_output(local_y, golden, INPUTS_LEN);
      printf("errors = %d\n", errors);
    }
    snrt_cluster_hw_barrier();

    return 0;
}