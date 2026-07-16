// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "data.h"
#include "snrt.h"

#define STRINGIFY_IMPL(x) #x
#define STRINGIFY(x) STRINGIFY_IMPL(x)
#define CSR_PACE 0xBA0

#define FPREG_FT0 0
#define FPREG_FT1 1
#define FPREG_FT2 2
#define FPREG_FT3 3
#define FPREG_FT4 4
#define FPREG_FT5 5
#define FPREG_FT6 6
#define FPREG_FT7 7

#define PACE_RSQRT_SRC_REG FPREG_FT3
#define PACE_RSQRT_DST_REG FPREG_FT4

#define PACE_VWORD(funct3, mode_bits, rd, rs1) \
  ((uint32_t)(((0xEu) << 28) | ((uint32_t)(mode_bits) << 25) | \
              ((uint32_t)(funct3) << 12) | \
              ((uint32_t)(rs1) << 15) | ((uint32_t)(rd) << 7) | 0x33u))

#define LAYERNORM_PASS1_UNROLL 3
#define LAYERNORM_PASS1_GROUPS (ELEMENTS_PER_CHANNEL / LAYERNORM_PASS1_UNROLL)
#define LAYERNORM_PASS1_TAIL (ELEMENTS_PER_CHANNEL % LAYERNORM_PASS1_UNROLL)

#if (ELEMENTS_PER_CHANNEL < LAYERNORM_PASS1_UNROLL)
#error "ELEMENTS_PER_CHANNEL must be at least 3 for the 3-way unrolled layernorm pass"
#endif

typedef __fp16 fp16_t;
typedef union {
    fp16_t lane[4];
    uint64_t u64;
} fp16x4_bits_t;
typedef union {
    fp16_t f16;
    uint16_t u16;
} fp16_bits_t;

static int check_output(data_t *actual, data_t *golden_data, int len) {
    int errors = 0;
    for (int i = 0; i < len; i++) {
        if (actual[i] != golden_data[i]) {
            errors++;
            printf("idx:%d actual=0x%x golden=0x%x\n", i, actual[i], golden_data[i]);
        }
    }
    return errors;
}

static inline uint64_t pack_replicated_fp16_bits(uint16_t value_bits) {
    fp16x4_bits_t packed;
    fp16_bits_t value = {.u16 = value_bits};
    packed.lane[0] = value.f16;
    packed.lane[1] = value.f16;
    packed.lane[2] = value.f16;
    packed.lane[3] = value.f16;
    return packed.u64;
}

int main() {
    data_t *local_x;
    data_t *local_y;
    fp16_t *local_mean;
    fp16_t *local_r;
    uint64_t *local_inv_n_vec;
    uint64_t *local_zero_vec;
    param_t *pace_mem = (param_t *)snrt_cluster()->pacemem.mem;
    const uint32_t core_idx = snrt_cluster_core_idx();
    const uint32_t core_active = snrt_is_compute_core() && (core_idx == 0);
    const fp16_t inv_n_h = (fp16_t)(1.0f / (float)ELEMENTS_PER_CHANNEL);
    const fp16_bits_t inv_n_bits = {.f16 = inv_n_h};

    local_x = (data_t *)snrt_l1_next();
    local_y = local_x + INPUTS_LEN;
    local_mean = (fp16_t *)(local_y + INPUTS_LEN);
    local_r = local_mean + CHANNELS;
    local_inv_n_vec = (uint64_t *)(local_r + CHANNELS);
    local_zero_vec = local_inv_n_vec + 1;

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(pace_mem, params, PARAMS_LEN * sizeof(param_t));
        snrt_dma_wait_all();
        snrt_dma_start_1d(local_x, ifmap, INPUTS_LEN * sizeof(data_t));
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    if (core_active) {
        *local_inv_n_vec = pack_replicated_fp16_bits(inv_n_bits.u16);
        *local_zero_vec = 0ULL;
        asm volatile("csrw " STRINGIFY(CSR_PACE) ", %0" : : "rK"(PACE_DEGREE) : "memory");
    }
    snrt_cluster_hw_barrier();

    if (core_active) {
        // Initialize all SSR streams once:
        //   ft0 <- input stream for pass 1
        //   ft1 <- input stream for pass 2
        //   ft2 -> output stream for pass 2
        snrt_ssr_loop_1d(SNRT_SSR_DM0, ELEMENTS_PER_CHANNEL, sizeof(uint64_t));
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, local_x);
        snrt_ssr_loop_1d(SNRT_SSR_DM1, ELEMENTS_PER_CHANNEL, sizeof(uint64_t));
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, local_x);
        snrt_ssr_loop_1d(SNRT_SSR_DM2, ELEMENTS_PER_CHANNEL, sizeof(uint64_t));
        snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, local_y);
        snrt_ssr_enable();

        __asm__ volatile(
            "fld fs0, 0(%[inv_n])\n\t"
            "fld fs1, 0(%[zero])\n\t"
            // Pass 1:
            // consume ft0 only, keep ft1 and ft2 untouched for pass 2.
            // The body is 3-way unrolled to keep dependent multiplies apart.
            "fmv.d fs8, fs1\n\t"
            "fmv.d fs9, fs1\n\t"
            "fmv.d fs10, fs1\n\t"
            "fmv.d fs11, fs1\n\t"
            "fmv.d fa1, fs1\n\t"
            "fmv.d fa2, fs1\n\t"
            "frep.o  %[n_stats], 15, 0, 0\n\t"
            // Latch three streamed HWC words from ft0.
            "fmv.d  ft3, ft0\n\t"
            // q path 0: x_scaled = x * inv_n
            "vfmul.h ft5, ft3, fs0\n\t"
            "fmv.d  ft4, ft0\n\t"
            // q path 1: x_scaled = x * inv_n
            "vfmul.h ft7, ft4, fs0\n\t"
            "fmv.d  ft6, ft0\n\t"
            // q path 2: x_scaled = x * inv_n
            "vfmul.h fa0, ft6, fs0\n\t"
            // s path 0: s += x
            "vfadd.h fs8, fs8, ft3\n\t"
            // q path 0: x_prod = x_scaled * x
            "vfmul.h ft5, ft5, ft3\n\t"
            // s path 1: s += x
            "vfadd.h fs9, fs9, ft4\n\t"
            // q path 1: x_prod = x_scaled * x
            "vfmul.h ft7, ft7, ft4\n\t"
            // q path 2: x_prod = x_scaled * x
            "vfmul.h fa0, fa0, ft6\n\t"
            // s path 2: s += x
            "vfadd.h fs10, fs10, ft6\n\t"
            // q path 0: q += x_prod
            "vfadd.h fs11, fs11, ft5\n\t"
            // q path 1: q += x_prod
            "vfadd.h fa1, fa1, ft7\n\t"
            // q path 2: q += x_prod
            "vfadd.h fa2, fa2, fa0\n\t"
#if LAYERNORM_PASS1_TAIL >= 1
            // Tail element 0 folds into slot 0.
            "fmv.d  ft3, ft0\n\t"
            "vfmul.h ft5, ft3, fs0\n\t"
            "vfadd.h fs8, fs8, ft3\n\t"
            "vfmul.h ft5, ft5, ft3\n\t"
            "vfadd.h fs11, fs11, ft5\n\t"
#endif
#if LAYERNORM_PASS1_TAIL == 2
            // Tail element 1 folds into slot 1.
            "fmv.d  ft4, ft0\n\t"
            "vfmul.h ft7, ft4, fs0\n\t"
            "vfadd.h fs9, fs9, ft4\n\t"
            "vfmul.h ft7, ft7, ft4\n\t"
            "vfadd.h fa1, fa1, ft7\n\t"
#endif
            // Reduce the 3-way accumulators.
            "vfadd.h fs8, fs8, fs9\n\t"
            "vfadd.h fs8, fs8, fs10\n\t"
            "vfadd.h fs11, fs11, fa1\n\t"
            "vfadd.h fs11, fs11, fa2\n\t"
            // mean = s * inv_n
            "vfmul.h fs8, fs8, fs0\n\t"
            // sigma2 = q - mean^2
            "vfmul.h fa0, fs8, fs8\n\t"
            "vfsub.h ft3, fs11, fa0\n\t"
            // r = rsqrt(sigma2), explicitly encoded with named source/destination regs.
            ".word %c[pace_rsqrt_word]\n\t"
            // Preserve rsqrt in a saved register for pass 2.
            "fmv.d fs2, ft4\n\t"
            // Pass 2:
            // ft1 fetches input, ft2 writes output.
            "frep.o  %[n_norm], 8, 0, 0\n\t"
            "vfsub.h ft3, ft1, fs8\n\t"
            "vfsub.h ft4, ft1, fs8\n\t"
            "vfsub.h ft5, ft1, fs8\n\t"
            "vfsub.h ft6, ft1, fs8\n\t"
            "vfmul.h ft2, ft3, fs2\n\t"
            "vfmul.h ft2, ft4, fs2\n\t"
            "vfmul.h ft2, ft5, fs2\n\t"
            "vfmul.h ft2, ft6, fs2\n\t"
            // Store broadcast mean and rsqrt for visibility/debug.
            "fsd fs8, 0(%[mean])\n\t"
            "fsd fs2, 0(%[r])\n\t"
            :
            : [n_stats] "r"(LAYERNORM_PASS1_GROUPS - 1),
              [n_norm] "r"(ELEMENTS_PER_CHANNEL / 4 - 1),
              [inv_n] "r"(local_inv_n_vec), [zero] "r"(local_zero_vec),
              [mean] "r"(local_mean), [r] "r"(local_r),
              [pace_rsqrt_word] "i"(PACE_VWORD(PACE_VECTOR_FUNCT3, PACE_MODE_BITS,
                                               PACE_RSQRT_DST_REG,
                                               PACE_RSQRT_SRC_REG))
            : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
              "fa0", "fa1", "fa2", "fs0", "fs1", "fs2", "fs8", "fs9",
              "fs10", "fs11", "memory");

        snrt_fpu_fence();
        snrt_ssr_disable();
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
