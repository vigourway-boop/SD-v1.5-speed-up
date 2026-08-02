#ifndef COSINE_SKIP_H
#define COSINE_SKIP_H

#include <ap_int.h>

#define MAX_VECTOR_LEN 4096

typedef ap_int<8> data_t;
typedef ap_int<64> acc_t;
typedef ap_uint<64> norm_t;

extern "C" int cosine_skip(
    const data_t *x,
    int length,
    int step_index,
    int threshold_q15,
    int warmup_steps,
    int max_consecutive_skips,
    int reset_state,
    acc_t *dot_out,
    norm_t *norm_x_out,
    norm_t *norm_y_out,
    int *threshold_passed_out,
    int *skip_streak_out
);

#endif
