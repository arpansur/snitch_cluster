// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stdio.h>

#include "data.h"
#include "snrt.h"

#define STRINGIFY_IMPL(x) #x
#define STRINGIFY(x) STRINGIFY_IMPL(x)
#define CSR_PACE 0xBA0

#if defined(ENABLE_FP16)
typedef __fp16 pace_scalar_t;
typedef union {
    pace_scalar_t f;
    uint16_t u;
} pace_scalar_bits_t;
#elif defined(ENABLE_FP32)
typedef float pace_scalar_t;
typedef union {
    pace_scalar_t f;
    uint32_t u;
} pace_scalar_bits_t;
#else
#error "Unsupported precision configuration"
#endif

static inline pace_scalar_t bits_to_scalar(data_t bits) {
    pace_scalar_bits_t conv = {.u = bits};
    return conv.f;
}

static inline data_t scalar_to_bits(pace_scalar_t value) {
    pace_scalar_bits_t conv = {.f = value};
    return conv.u;
}

static int check_output(const data_t *actual, const data_t *golden_data, uint32_t len) {
    int errors = 0;
    for (uint32_t i = 0; i < len; ++i) {
        if (actual[i] != golden_data[i]) {
            ++errors;
            printf("idx:%u actual=0x%x golden=0x%x\n", i, actual[i], golden_data[i]);
        }
    }
    return errors;
}

int main() {
    data_t *local_x;
    data_t *local_y;
    param_t *local_param;
    param_t *pace_mem = (param_t *)snrt_cluster()->pacemem.mem;
    const uint32_t core_idx = snrt_cluster_core_idx();
    const uint32_t compute_core_count = snrt_cluster_compute_core_num();

    local_x = (data_t *)snrt_l1_next();
    local_y = local_x + INPUTS_LEN;
    local_param = (param_t *)(local_y + INPUTS_LEN);

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_x, ifmap, INPUTS_LEN * sizeof(data_t));
        snrt_dma_wait_all();
        snrt_dma_start_1d(local_param, params, PARAMS_LEN * sizeof(param_t));
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    if (snrt_is_compute_core()) {
        asm volatile("csrw " STRINGIFY(CSR_PACE) ", %0" : : "rK"(PACE_DEGREE) : "memory");
    }
    snrt_cluster_hw_barrier();

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(pace_mem, local_param, PARAMS_LEN * sizeof(param_t));
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    if (snrt_is_compute_core()) {
        const uint32_t start = (INPUTS_LEN * core_idx) / compute_core_count;
        const uint32_t end = (INPUTS_LEN * (core_idx + 1)) / compute_core_count;

        for (uint32_t i = start; i < end; ++i) {
            register pace_scalar_t pace_in asm("ft0") = bits_to_scalar(local_x[i]);
            register pace_scalar_t pace_out asm("ft1");

            __asm__ volatile("" : : "f"(pace_in));
            __asm__ volatile(PACE_SCALAR_ASM : "=f"(pace_out) : : "memory");
            local_y[i] = scalar_to_bits(pace_out);
        }

        snrt_fpu_fence();
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
