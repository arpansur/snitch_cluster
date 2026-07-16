// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Arpan Suravi Prasad <prasadar@iis.ee.ethz.ch>

#include "math.h"
#include "snrt.h"
#include "data.h"

#define INPUTS_LEN (Q_SIZE*K_SIZE)
#define OUTPUTS_LEN (Q_SIZE*K_SIZE)
#define FP32_NEG_INF_BITS 0xFF800000u
#define FP32_ZERO_BITS 0x00000000u
#define FP16_NEG_INF_BITS 0xFC00u
#define FP16_ZERO_BITS 0x0000u
#define FP16X4_NEG_INF_BITS 0xFC00FC00FC00FC00ULL
#define FP16X4_ZERO_BITS 0x0000000000000000ULL
#define STRINGIFY_IMPL(x) #x
#define STRINGIFY(x) STRINGIFY_IMPL(x)
#define CSR_PACE 0xBA0

#if defined(ENABLE_FP32)
#define PACE_FUNCT3 0u
#define PACE_NEG_INF_BITS FP32_NEG_INF_BITS
#define PACE_ZERO_BITS FP32_ZERO_BITS
typedef uint32_t raw_data_t;
#elif defined(ENABLE_FP16)
#define PACE_FUNCT3 1u
#define PACE_NEG_INF_BITS FP16_NEG_INF_BITS
#define PACE_ZERO_BITS FP16_ZERO_BITS
typedef uint16_t raw_data_t;
#else
#error "Unsupported precision configuration"
#endif

#if FPU_DATA_WIDTH == 64
#define PACE_SSR_STRIDE sizeof(double)
#elif FPU_DATA_WIDTH == 32
#define PACE_SSR_STRIDE sizeof(float)
#else
#error "Unsupported FPU_DATA_WIDTH configuration"
#endif

#if (PACE_LANES != 1) && (PACE_LANES != 2) && (PACE_LANES != 4)
#error "Unsupported PACE_LANES configuration"
#endif

#if (SOFTMAX_UNROLL != 3) && (SOFTMAX_UNROLL != 4) && (SOFTMAX_UNROLL != 8)
#error "SOFTMAX_UNROLL must be 3, 4 or 8"
#endif

#if defined(ENABLE_FP16) && (PACE_LANES == 4) && (FPU_DATA_WIDTH == 64)
#define PACE_ROW_INTERLEAVED 1
#define PACE_XMAX_ITERS (K_SIZE / 4)
#define PACE_EXP_ITERS (K_SIZE / SOFTMAX_UNROLL)
#define PACE_VEC_ITERS K_SIZE
#else
#define PACE_ROW_INTERLEAVED 0
#define PACE_XMAX_ITERS (K_SIZE / (4 * PACE_LANES))
#define PACE_EXP_ITERS (K_SIZE / (SOFTMAX_UNROLL * PACE_LANES))
#define PACE_VEC_ITERS (K_SIZE / PACE_LANES)
#endif

#if (K_SIZE % (4 * PACE_LANES)) != 0
#error "K_SIZE must be divisible by 4 * PACE_LANES"
#endif

#if PACE_ROW_INTERLEAVED
#if (K_SIZE % SOFTMAX_UNROLL) != 0
#error "K_SIZE must be divisible by SOFTMAX_UNROLL"
#endif
#else
#if (K_SIZE % (SOFTMAX_UNROLL * PACE_LANES)) != 0
#error "K_SIZE must be divisible by SOFTMAX_UNROLL * PACE_LANES"
#endif
#endif

#define FPREG_FT0 0
#define FPREG_FT1 1
#define FPREG_FT2 2
#define FPREG_FT3 3
#define FPREG_FT4 4
#define FPREG_FT5 5
#define FPREG_FT6 6
#define FPREG_FT7 7
#define FPREG_FS0 8
#define FPREG_FS1 9
#define FPREG_FA0 10
#define FPREG_FA1 11

#define PACE_VWORD(funct3, mode_bits, rd, rs1) \
  ((uint32_t)(((0xEu) << 28) | ((uint32_t)(mode_bits) << 25) | \
              ((uint32_t)(funct3) << 12) | \
              ((uint32_t)(rs1) << 15) | ((uint32_t)(rd) << 7) | 0x33u))

int check_output(raw_data_t* actual, raw_data_t* golden, int len)
{
  int errors = len; 
  for (int i=0; i<len; i++){
    raw_data_t actual_data = *(actual+i);
    raw_data_t golden_data = *(golden+i);
    if(actual_data == golden_data)
      errors--;
    else 
      printf("idx:%d, errors=%d, actual_data=%x, golden_data=%x, actual_ptr=%x, golden_ptr=%x\n", i, errors, actual_data, golden_data, (actual+i), (golden+i));
  }
  return errors;
}

#if defined(ENABLE_FP16) && (PACE_LANES == 4)
#if PACE_ROW_INTERLEAVED
#include "softmax_fp16x4_column_major.h"
#else
#include "softmax_fp16x4_row_major.h"
#endif
#endif


int main() {
    raw_data_t *input_buf, *exp_buf, *softmax_buf;
    raw_data_t *denom_buf, *inv_denom_buf;
    raw_data_t *input_src;
    raw_data_t *denom_inv_src;
    raw_data_t *inv_mul_src;
    volatile raw_data_t *denom_write_ptr;
    raw_data_t *scratch_buf;
    raw_data_t *xmax_lane_buf;
    volatile raw_data_t *denom_lane_buf;
    param_t *pace_mem = (param_t *)snrt_cluster()->pacemem.mem;
    const uint64_t fp16x4_neg_inf = FP16X4_NEG_INF_BITS;
    const uint64_t fp16x4_zero = FP16X4_ZERO_BITS;
    const uint32_t runtime_compute_core_count = snrt_cluster_compute_core_num();
    const uint32_t core_idx = snrt_cluster_core_idx();
    const uint32_t active_compute_core_count = NUM_CORES;
    const uint32_t core_active =
        snrt_is_compute_core() && (core_idx < active_compute_core_count);

    if (runtime_compute_core_count < NUM_CORES) {
      if (core_idx == 0) {
        printf("pace_softmax core mismatch: runtime=%u configured=%u\n",
               runtime_compute_core_count, NUM_CORES);
      }
      return 1;
    }

    const uint32_t work_items = PACE_ROW_INTERLEAVED ? (Q_SIZE / PACE_LANES) : Q_SIZE;
    const uint32_t rows_per_core = work_items / active_compute_core_count;
    const uint32_t extra_rows = work_items % active_compute_core_count;
    const uint32_t row_start =
        core_idx * rows_per_core + (core_idx < extra_rows ? core_idx : extra_rows);
    const uint32_t row_count =
        core_active ? (rows_per_core + (core_idx < extra_rows ? 1u : 0u)) : 0u;
    const uint32_t input_span = PACE_ROW_INTERLEAVED ? (K_SIZE * PACE_LANES) : K_SIZE;
    const uint32_t input_offset = row_start * input_span;
    const uint32_t denom_offset = row_start * PACE_LANES;
    const uint32_t scratch_plane_size = active_compute_core_count * PACE_LANES;

    input_buf = (raw_data_t *)snrt_l1_next();
    exp_buf = input_buf + INPUTS_LEN;
    softmax_buf = exp_buf + OUTPUTS_LEN;
    denom_buf = softmax_buf + OUTPUTS_LEN;
    inv_denom_buf = denom_buf + DENO_LENGTH;

    input_src = &(ifmap[0][0]);
    denom_write_ptr = denom_buf + denom_offset;
    denom_inv_src = denom_buf + denom_offset;
    inv_mul_src = inv_denom_buf + denom_offset;
    scratch_buf = inv_denom_buf + DENO_LENGTH;
    xmax_lane_buf = scratch_buf + core_idx * PACE_LANES;
    denom_lane_buf = scratch_buf + scratch_plane_size + core_idx * PACE_LANES;
    //////////////////
    // DMA 1D WRITE //
    //////////////////
    if (core_active) {
      asm volatile("csrw " STRINGIFY(CSR_PACE) ", %0" : : "rK"(PACE_DEGREE) : "memory");
    }
    snrt_cluster_hw_barrier();

    if (snrt_is_dm_core()) {
      snrt_dma_start_1d(pace_mem, &(exp_params[0]), EXP_PARAMS_LEN * sizeof(param_t));
      snrt_dma_wait_all();
      snrt_dma_start_1d(input_buf, input_src, INPUTS_LEN * sizeof(raw_data_t));
      snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();
    if (core_active && row_count > 0) {
      snrt_ssr_loop_1d(SNRT_SSR_DM0, PACE_VEC_ITERS * row_count, PACE_SSR_STRIDE);
      snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, input_buf + input_offset);
      snrt_ssr_loop_1d(SNRT_SSR_DM1, PACE_VEC_ITERS * row_count, PACE_SSR_STRIDE);
      snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, input_buf + input_offset);
      snrt_ssr_loop_1d(SNRT_SSR_DM2, PACE_VEC_ITERS * row_count, PACE_SSR_STRIDE);
      snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, exp_buf + input_offset);
      snrt_ssr_enable();


      for (uint32_t row = 0; row < row_count; row++) {
#if defined(ENABLE_FP16) && (PACE_LANES == 4)
        pace_xmax_fp16x4(xmax_lane_buf, &fp16x4_neg_inf);
#elif PACE_LANES == 2
        __asm__ volatile(
          "fmv.s.x ft3, %[neg_inf]\n"
          "fmv.s.x ft4, %[neg_inf]\n"
          "vfcpka.s.s ft3, ft3, ft4\n" 
          "fmv.d  ft4, ft3 \n\t"
          "fmv.d  ft5, ft3 \n\t"
          "fmv.d  ft6, ft3 \n\t"
          "fmv.d  ft7, ft3 \n\t"
          "frep.o  %[n], 4, 0, 0\n\t"
          // unrolled 4 times 
          // x_max computation on 4 parallel registers
          "vfmax.s  ft4, ft4, ft0\n\t"
          "vfmax.s  ft5, ft5, ft0\n\t"
          "vfmax.s  ft6, ft6, ft0\n\t"
          "vfmax.s  ft7, ft7, ft0\n\t"
          // reduction 
          // reduction of x_max across 4 parallel registers
          "vfmax.s  ft5, ft4, ft5\n\t"
          "vfmax.s  ft6, ft6, ft7\n\t"
          "vfmax.s  ft6, ft6, ft5\n\t"
          "fsd ft6, 0(%[max]) \n\t"
          :
          : [neg_inf] "r"(PACE_NEG_INF_BITS), [n] "r"(PACE_XMAX_ITERS-1), [max] "r" (xmax_lane_buf)
          : "ft3", "ft4", "ft5", "ft6", "ft7", "memory"
        );
#else
        __asm__ volatile(
          "fmv.s.x ft4, %[neg_inf]\n"
          "fmv.s.x ft5, %[neg_inf]\n"
          "fmv.s.x ft6, %[neg_inf]\n"
          "fmv.s.x ft7, %[neg_inf]\n"
          "frep.o  %[n], 4, 0, 0\n\t"
          "vfmax.s  ft4, ft4, ft0\n\t"
          "vfmax.s  ft5, ft5, ft0\n\t"
          "vfmax.s  ft6, ft6, ft0\n\t"
          "vfmax.s  ft7, ft7, ft0\n\t"
          "fmax.s   ft5, ft4, ft5\n\t"
          "fmax.s   ft6, ft6, ft7\n\t"
          "fmax.s   ft3, ft6, ft5\n\t"
          "fsw ft3, 0(%[max]) \n\t"
          :
          : [neg_inf] "r"(PACE_NEG_INF_BITS), [n] "r"(PACE_XMAX_ITERS-1), [max] "r" (xmax_lane_buf)
          : "ft3", "ft4", "ft5", "ft6", "ft7", "memory"
        );
#endif
#if defined(ENABLE_FP16) && (PACE_LANES == 4)
        pace_exp_deno_fp16x4((raw_data_t *)denom_lane_buf, xmax_lane_buf, &fp16x4_zero);
#elif PACE_LANES == 2
        __asm__ volatile(
          "fmv.s.x fs8, %[zero]\n"
          "fmv.s.x fs9, %[zero]\n"
          "vfcpka.s.s fs8, fs9, fs8\n"
          "fmv.d  fs9, fs8 \n\t"
          "fmv.d  fs10, fs8 \n\t"
          "fmv.d  fs11, fs8 \n\t"
          "flw ft3, 0(%[xmax])\n\t"
          "flw ft4, 4(%[xmax])\n\t"
          "fmax.s ft4, ft3, ft4\n\t"
          "vfcpka.s.s ft3, ft4, ft4\n\t"
          "frep.o  %[n], 32, 0, 0\n\t"
          // unrolled 4 times, total 8 instruction
          // x - xmax computation
          "vfsub.s  ft4, ft1, ft3\n\t"
          "vfsub.s  ft5, ft1, ft3\n\t"
          "vfsub.s  ft6, ft1, ft3\n\t"
          "vfsub.s  ft7, ft1, ft3\n\t"
          "vfsub.s  fs0, ft1, ft3\n\t"
          "vfsub.s  fs1, ft1, ft3\n\t"
          "vfsub.s  fa0, ft1, ft3\n\t"
          "vfsub.s  fa1, ft1, ft3\n\t"
          ".word %c[pace_exp_ft4]\n\t"
          ".word %c[pace_exp_ft5]\n\t"
          ".word %c[pace_exp_ft6]\n\t"
          ".word %c[pace_exp_ft7]\n\t"
          ".word %c[pace_exp_fs0]\n\t"
          ".word %c[pace_exp_fs1]\n\t"
          ".word %c[pace_exp_fa0]\n\t"
          ".word %c[pace_exp_fa1]\n\t"
          "fmv.d  ft2, ft4\n\t"
          "fmv.d  ft2, ft5\n\t"
          "fmv.d  ft2, ft6\n\t"
          "fmv.d  ft2, ft7\n\t"
          "fmv.d  ft2, fs0\n\t"
          "fmv.d  ft2, fs1\n\t"
          "fmv.d  ft2, fa0\n\t"
          "fmv.d  ft2, fa1\n\t"
          "vfadd.s  fs4, ft5, ft4\n\t"
          "vfadd.s  fs5, ft7, ft6\n\t"
          "vfadd.s  fs6, fs1, fs0\n\t"
          "vfadd.s  fs7, fa1, fa0\n\t"
          "vfadd.s  fs8, fs8, fs4\n\t"
          "vfadd.s  fs9, fs9, fs5\n\t"
          "vfadd.s  fs10, fs10, fs6\n\t"
          "vfadd.s  fs11, fs11, fs7\n\t"
          "vfadd.s  fs8, fs9, fs8\n\t"
          "vfadd.s  fs10, fs10, fs11\n\t"
          "vfadd.s  fs8, fs10, fs8\n\t"
          "fsd fs8, 0(%[sum]) \n\t"
          :
          : [zero] "r"(PACE_ZERO_BITS), [n] "r"(PACE_EXP_ITERS - 1),
            [sum] "r"(denom_lane_buf), [xmax] "r"(xmax_lane_buf),
            [pace_exp_ft4] "i"(PACE_VWORD(PACE_FUNCT3, EXP_PACE_MODE_BITS, FPREG_FT4, FPREG_FT4)),
            [pace_exp_ft5] "i"(PACE_VWORD(PACE_FUNCT3, EXP_PACE_MODE_BITS, FPREG_FT5, FPREG_FT5)),
            [pace_exp_ft6] "i"(PACE_VWORD(PACE_FUNCT3, EXP_PACE_MODE_BITS, FPREG_FT6, FPREG_FT6)),
            [pace_exp_ft7] "i"(PACE_VWORD(PACE_FUNCT3, EXP_PACE_MODE_BITS, FPREG_FT7, FPREG_FT7)),
            [pace_exp_fs0] "i"(PACE_VWORD(PACE_FUNCT3, EXP_PACE_MODE_BITS, FPREG_FS0, FPREG_FS0)),
            [pace_exp_fs1] "i"(PACE_VWORD(PACE_FUNCT3, EXP_PACE_MODE_BITS, FPREG_FS1, FPREG_FS1)),
            [pace_exp_fa0] "i"(PACE_VWORD(PACE_FUNCT3, EXP_PACE_MODE_BITS, FPREG_FA0, FPREG_FA0)),
            [pace_exp_fa1] "i"(PACE_VWORD(PACE_FUNCT3, EXP_PACE_MODE_BITS, FPREG_FA1, FPREG_FA1))
          : "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
            "fs0", "fs1", "fs4", "fs5", "fs6", "fs7", "fs8", "fs9", "fs10", "fs11",
            "fa0", "fa1", "memory"
        );
#else
        __asm__ volatile(
          "fmv.s.x fs8, %[zero]\n"
          "fmv.s.x fs9, %[zero]\n"
          "fmv.s.x fs10, %[zero]\n"
          "fmv.s.x fs11, %[zero]\n"
          "flw ft3, 0(%[xmax])\n\t"
          "frep.o  %[n], 32, 0, 0\n\t"
          // unrolled 4 times, total 8 instruction
          // x - xmax computation
          "vfsub.s  ft4, ft1, ft3\n\t"
          "vfsub.s  ft5, ft1, ft3\n\t"
          "vfsub.s  ft6, ft1, ft3\n\t"
          "vfsub.s  ft7, ft1, ft3\n\t"
          "vfsub.s  fs0, ft1, ft3\n\t"
          "vfsub.s  fs1, ft1, ft3\n\t"
          "vfsub.s  fa0, ft1, ft3\n\t"
          "vfsub.s  fa1, ft1, ft3\n\t"
          ".word %c[pace_exp_ft4]\n\t"
          ".word %c[pace_exp_ft5]\n\t"
          ".word %c[pace_exp_ft6]\n\t"
          ".word %c[pace_exp_ft7]\n\t"
          ".word %c[pace_exp_fs0]\n\t"
          ".word %c[pace_exp_fs1]\n\t"
          ".word %c[pace_exp_fa0]\n\t"
          ".word %c[pace_exp_fa1]\n\t"
          "fmv.s  ft2, ft4\n\t"
          "fmv.s  ft2, ft5\n\t"
          "fmv.s  ft2, ft6\n\t"
          "fmv.s  ft2, ft7\n\t"
          "fmv.s  ft2, fs0\n\t"
          "fmv.s  ft2, fs1\n\t"
          "fmv.s  ft2, fa0\n\t"
          "fmv.s  ft2, fa1\n\t"
          "vfadd.s  fs4, ft5, ft4\n\t"
          "vfadd.s  fs5, ft7, ft6\n\t"
          "vfadd.s  fs6, fs1, fs0\n\t"
          "vfadd.s  fs7, fa1, fa0\n\t"
          "vfadd.s  fs8, fs8, fs4\n\t"
          "vfadd.s  fs9, fs9, fs5\n\t"
          "vfadd.s  fs10, fs10, fs6\n\t"
          "vfadd.s  fs11, fs11, fs7\n\t"
          "vfadd.s  fs8, fs9, fs8\n\t"
          "vfadd.s  fs10, fs10, fs11\n\t"
          "vfadd.s  fs8, fs10, fs8\n\t"
          "fsw fs8, 0(%[sum]) \n\t"
          :
          : [zero] "r"(PACE_ZERO_BITS), [n] "r"(PACE_EXP_ITERS - 1),
            [sum] "r"(denom_lane_buf), [xmax] "r"(xmax_lane_buf),
            [pace_exp_ft4] "i"(PACE_VWORD(PACE_FUNCT3, EXP_PACE_MODE_BITS, FPREG_FT4, FPREG_FT4)),
            [pace_exp_ft5] "i"(PACE_VWORD(PACE_FUNCT3, EXP_PACE_MODE_BITS, FPREG_FT5, FPREG_FT5)),
            [pace_exp_ft6] "i"(PACE_VWORD(PACE_FUNCT3, EXP_PACE_MODE_BITS, FPREG_FT6, FPREG_FT6)),
            [pace_exp_ft7] "i"(PACE_VWORD(PACE_FUNCT3, EXP_PACE_MODE_BITS, FPREG_FT7, FPREG_FT7)),
            [pace_exp_fs0] "i"(PACE_VWORD(PACE_FUNCT3, EXP_PACE_MODE_BITS, FPREG_FS0, FPREG_FS0)),
            [pace_exp_fs1] "i"(PACE_VWORD(PACE_FUNCT3, EXP_PACE_MODE_BITS, FPREG_FS1, FPREG_FS1)),
            [pace_exp_fa0] "i"(PACE_VWORD(PACE_FUNCT3, EXP_PACE_MODE_BITS, FPREG_FA0, FPREG_FA0)),
            [pace_exp_fa1] "i"(PACE_VWORD(PACE_FUNCT3, EXP_PACE_MODE_BITS, FPREG_FA1, FPREG_FA1))
          : "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
            "fs0", "fs1", "fs4", "fs5", "fs6", "fs7", "fs8", "fs9", "fs10", "fs11",
            "fa0", "fa1", "memory"
        );
#endif

#if defined(ENABLE_FP16) && (PACE_LANES == 4)
      pace_store_deno_fp16x4(&denom_write_ptr, denom_lane_buf);
#elif PACE_LANES == 2
      __asm__ volatile(
        "flw ft3, %1\n\t"
        "flw ft4, %2\n\t"
        "fadd.s ft4, ft3, ft4\n\t"
        "vfcpka.s.s ft3, ft4, ft4\n\t"
        "fsd ft3, %0\n\t"
        : "=m"(*(volatile double *)denom_write_ptr)
        : "m"(denom_lane_buf[0]), "m"(denom_lane_buf[1])
        : "ft3", "ft4", "memory"
        );

      denom_write_ptr += 2;


#else
        __asm__ volatile(
          "flw ft3, 0(%[sum_src])\n\t"
          "fsw ft3, 0(%[sum_dst])\n\t"
          "addi %[sum_dst], %[sum_dst], 4\t\n"
          : [sum_dst] "+r"(denom_write_ptr)
          : [sum_src] "r"(denom_lane_buf)
          : "ft3", "memory"
        );
#endif
      }
        snrt_fpu_fence();
        snrt_ssr_disable();
    }
    snrt_cluster_hw_barrier();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(pace_mem, inv_params, INV_PARAMS_LEN * sizeof(param_t));
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();
    if (core_active && row_count > 0) {
        snrt_ssr_loop_1d(SNRT_SSR_DM0, row_count, PACE_SSR_STRIDE);
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, denom_inv_src);
        snrt_ssr_loop_1d(SNRT_SSR_DM2, row_count, PACE_SSR_STRIDE);
        snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, inv_denom_buf + denom_offset);
        snrt_ssr_enable();
#if PACE_LANES > 1
        __asm__ volatile(
            "frep.o  %[n], 1, 0, 0\n\t"
            ".word %c[pace_inv_ft2_ft0]\n\t"
            :
            : [n] "r"(row_count - 1),
              [pace_inv_ft2_ft0] "i"(PACE_VWORD(PACE_FUNCT3, INV_PACE_MODE_BITS, FPREG_FT2, FPREG_FT0))
            : "ft0", "ft2", "memory"
          );
#else
        __asm__ volatile(
            "frep.o  %[n], 1, 0, 0\n\t"
            ".word %c[pace_inv_ft2_ft0]\n\t"
            :
            : [n] "r"(row_count - 1),
              [pace_inv_ft2_ft0] "i"(INV_PACE_SCALAR_WORD)
            : "ft0", "ft2", "memory"
          );
#endif
        snrt_fpu_fence();
        snrt_ssr_disable();
        snrt_ssr_loop_1d(SNRT_SSR_DM0, row_count * PACE_VEC_ITERS, PACE_SSR_STRIDE);
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, exp_buf + input_offset);
        snrt_ssr_loop_1d(SNRT_SSR_DM2, row_count * PACE_VEC_ITERS, PACE_SSR_STRIDE);
        snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, softmax_buf + input_offset);
        snrt_ssr_enable();
        for (uint32_t row = 0; row < row_count; row++)
        {
#if defined(ENABLE_FP16) && (PACE_LANES == 4)
          __asm__ volatile(
            "fld ft3, 0(%[inv])\n\t"
            "frep.o  %[n], 1, 0, 0\n\t"
            "vfmul.h  ft2, ft3, ft0\n\t"
            :
            : [inv] "r"(inv_mul_src), [n] "r"(PACE_VEC_ITERS - 1)
            : "ft0", "ft2", "memory"
          );
          inv_mul_src += PACE_LANES;
#elif PACE_LANES == 2
          __asm__ volatile(
            "fld ft3, 0(%[inv])\n\t"
            "frep.o  %[n], 1, 0, 0\n\t"
            "vfmul.s  ft2, ft3, ft0\n\t"
            :
            : [inv] "r"(inv_mul_src), [n] "r"(PACE_VEC_ITERS - 1)
            : "ft0", "ft2", "memory"
          );
          inv_mul_src += PACE_LANES;
#else
          __asm__ volatile(
            "flw ft3, 0(%[inv])\n\t"
            "frep.o  %[n], 1, 0, 0\n\t"
            "vfmul.s  ft2, ft3, ft0\n\t"
            :
            : [inv] "r"(inv_mul_src), [n] "r"(PACE_VEC_ITERS - 1)
            : "ft0", "ft2", "memory"
          );
          inv_mul_src += PACE_LANES;
#endif
        }
        snrt_fpu_fence();
        snrt_ssr_disable();
    }
    snrt_cluster_hw_barrier();

    if (snrt_is_dm_core()) {
      snrt_dma_start_1d(&(ofmap[0][0]), softmax_buf, OUTPUTS_LEN * sizeof(raw_data_t));
      snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    if (core_idx == 0) {
      int attn_oup_errors = check_output((raw_data_t*)(&(ofmap[0][0])), (raw_data_t*)(&(golden[0][0])), OUTPUTS_LEN);
      printf("attn_oup_errors = %d\n", attn_oup_errors);
    }
    snrt_cluster_hw_barrier();

    return 0;
}
