#include "cosine_skip.h"

extern "C" int cosine_skip(
    const data_t *x,
    const data_t *y,
    int length,
    int threshold_q15,
    acc_t *dot_out,
    norm_t *norm_x_out,
    norm_t *norm_y_out
) {
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem0 depth=MAX_VECTOR_LEN max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem1 depth=MAX_VECTOR_LEN max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=y bundle=control
#pragma HLS INTERFACE s_axilite port=length bundle=control
#pragma HLS INTERFACE s_axilite port=threshold_q15 bundle=control
#pragma HLS INTERFACE s_axilite port=dot_out bundle=control
#pragma HLS INTERFACE s_axilite port=norm_x_out bundle=control
#pragma HLS INTERFACE s_axilite port=norm_y_out bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    *dot_out = 0;
    *norm_x_out = 0;
    *norm_y_out = 0;
    if (length <= 0 || length > MAX_VECTOR_LEN) {
        return 0;
    }

    acc_t dot = 0;
    norm_t norm_x = 0;
    norm_t norm_y = 0;

    for (int i = 0; i < MAX_VECTOR_LEN; i++) {
#pragma HLS PIPELINE II=1
        if (i < length) {
            data_t xi = x[i];
            data_t yi = y[i];
            ap_int<32> x32 = xi;
            ap_int<32> y32 = yi;
            dot += x32 * y32;
            norm_x += x32 * x32;
            norm_y += y32 * y32;
        }
    }

    *dot_out = dot;
    *norm_x_out = norm_x;
    *norm_y_out = norm_y;
    if (dot <= 0 || norm_x == 0 || norm_y == 0) {
        return 0;
    }

    ap_uint<96> dot_abs = (ap_uint<96>)dot;
    ap_uint<128> left = (ap_uint<128>)dot_abs * dot_abs;

    ap_uint<128> norm_prod = (ap_uint<128>)norm_x * (ap_uint<128>)norm_y;
    ap_uint<64> thresh = 0;
    if (threshold_q15 > 0) {
        thresh = (ap_uint<64>)threshold_q15;
    }
    ap_uint<128> thresh_sq = (ap_uint<128>)thresh * (ap_uint<128>)thresh;

    // Compare dot^2 / (norm_x * norm_y) > threshold^2.
    // threshold_q15 uses Q1.15, so threshold^2 uses Q2.30.
    ap_uint<128> right = (norm_prod * thresh_sq) >> 30;

    return left > right ? 1 : 0;
}
