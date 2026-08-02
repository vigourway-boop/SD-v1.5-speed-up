#ifndef COSINE_SKIP_H
#define COSINE_SKIP_H

#include <ap_int.h>

#define MAX_VECTOR_LEN 32768

typedef ap_int<16> data_t;
typedef ap_int<64> acc_t;
typedef ap_uint<64> norm_t;

extern "C" int cosine_skip(
    const data_t *x,
    const data_t *y,
    int length,
    int threshold_q15,
    acc_t *dot_out,
    norm_t *norm_x_out,
    norm_t *norm_y_out
);

#endif
