// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Arpan Suravi Prasad <prasadar@iis.ee.ethz.ch>

#include "math.h"
#include "snrt.h"
#include "data.h"
#define SSR 1
// #define VECTOR 1
#define STRINGIFY_IMPL(x) #x
#define STRINGIFY(x) STRINGIFY_IMPL(x)
#define CSR_PACE 0xBA0

int check_output(data_t* actual, data_t* golden, int len)
{
  int errors = len; 
  for (int i=0; i<len; i++){
    data_t actual_data = *(actual+i);
    data_t golden_data = *(golden+i);
    if(actual_data == golden_data)
      errors--;
    else 
      printf("idx:%d, errors=%d, actual_data=%x, golden_data=%x, actual_ptr=%x, golden_ptr=%x\n", i, errors, actual_data, golden_data, (actual+i), (golden+i));
  }
  return errors;
}

int main() {
    data_t *local_x, *local_y, *local_param;
    data_t *remote_x;
    param_t *remote_params = params;
    data_t *pace_mem = (data_t *)snrt_cluster()->pacemem.mem;
    const uint32_t compute_core_count = snrt_cluster_compute_core_num();
    const uint32_t core_idx = snrt_cluster_core_idx();
    const uint32_t inputs_len_per_core = INPUTS_LEN / compute_core_count;
    local_x = (data_t *)snrt_l1_next();
    local_y = local_x + INPUTS_LEN;
    local_param = local_y + INPUTS_LEN;
    remote_x = ifmap;
    remote_params = params;

    int32_t start_cycle, end_cycle, pwpa_start_cycle, pwpa_end_cycle;
    int32_t dma_pace_start_cycle, dma_pace_end_cycle;

    data_t *core_local_x, *core_local_y;
    core_local_x = local_x + core_idx * inputs_len_per_core;
    core_local_y = local_y + core_idx * inputs_len_per_core;
    //////////////////
    // DMA 1D WRITE //
    //////////////////
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_x, remote_x, INPUTS_LEN * sizeof(data_t));
        snrt_dma_wait_all();
        snrt_dma_start_1d(local_param, params, PARAMS_LEN * sizeof(param_t));
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();
    if (snrt_is_compute_core()) {
        asm volatile("csrw " STRINGIFY(CSR_PACE) ", %0" : : "rK"(PACE_DEGREE) : "memory");
    }
    snrt_cluster_hw_barrier();
    dma_pace_start_cycle = snrt_mcycle();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(pace_mem, local_param, PARAMS_LEN * sizeof(param_t));
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();
    dma_pace_end_cycle = snrt_mcycle();
    start_cycle = snrt_mcycle();
    if (snrt_is_compute_core()) {
#ifdef SSR 
      snrt_ssr_loop_1d(SNRT_SSR_DM0, inputs_len_per_core / PACE_LANES, sizeof(double));
      snrt_ssr_loop_1d(SNRT_SSR_DM1, inputs_len_per_core / PACE_LANES, sizeof(double));
      snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, core_local_x);
      snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, core_local_y);
      snrt_ssr_enable();
      pwpa_start_cycle = snrt_mcycle();
      __asm__ volatile(
          "frep.o  %[n], 1, 0, 0\n\t"
          PACE_VECTOR_SSR_ASM "\n\t"
          :
          : [n] "r"(inputs_len_per_core / PACE_LANES - 1)
          : "ft1", "memory");
      snrt_fpu_fence();
      pwpa_end_cycle = snrt_mcycle();
      snrt_ssr_disable();
#else
#ifdef VECTOR 
    double op_b, op_c;
    double *inp_ptr, *oup_ptr;
    inp_ptr = (double*) local_x;
    oup_ptr = (double*) local_y;
    for(int i=0; i<INPUTS_LEN/2; i++) {
      op_b = *(inp_ptr + i);
      register double pace_in asm("ft0") = op_b;
      register double pace_out asm("ft1");
      __asm__ volatile("" : : "f"(pace_in));
      __asm__ volatile(PACE_VECTOR_SSR_ASM : "=f"(pace_out) : : "memory");
      op_c = pace_out;
      *(oup_ptr + i) = op_c;
    }
#else 
  float op_b, op_c;
  float* inp_ptr = (float*) core_local_x;
  float* oup_ptr = (float*) core_local_y;
  for(int i=0; i<inputs_len_per_core; i++)
  {
    op_b = *(inp_ptr + i);
    register float pace_in asm("ft0") = op_b;
    register float pace_out asm("ft1");
    __asm__ volatile("" : : "f"(pace_in));
    __asm__ volatile(PACE_SCALAR_ASM : "=f"(pace_out) : : "memory");
    op_c = pace_out;
    *(oup_ptr + i) = op_c;
  }
#endif
#endif
    }
    snrt_cluster_hw_barrier();
    end_cycle = snrt_mcycle();
    if(core_idx == 0)
    {
      printf("start cycle: %d, end_cycle: %d, diff_cycle:%d\n", start_cycle, end_cycle, end_cycle - start_cycle);
      printf("pwpa start cycle: %d, pwpa end_cycle: %d, pwpa diff_cycle:%d\n", pwpa_start_cycle, pwpa_end_cycle, pwpa_end_cycle - pwpa_start_cycle);
      printf("dma start cycle: %d, dma end_cycle: %d, dma diff_cycle:%d\n", dma_pace_start_cycle, dma_pace_end_cycle, dma_pace_end_cycle - dma_pace_start_cycle);
    }
    snrt_cluster_hw_barrier();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ofmap, local_y, INPUTS_LEN * sizeof(data_t));
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();
    if (core_idx == 0) {
      int errors = check_output(ofmap, golden, INPUTS_LEN);
      printf("errors = %d\n", errors);
    }
    snrt_cluster_hw_barrier();

    return 0;
}
