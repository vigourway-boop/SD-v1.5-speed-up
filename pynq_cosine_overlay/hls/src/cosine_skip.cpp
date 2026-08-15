#include "cosine_skip.h"

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
    int *skip_streak_out,
    int distance_threshold_q20,
    int *distance_passed_out
) {
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem depth=MAX_VECTOR_LEN max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=length bundle=control
#pragma HLS INTERFACE s_axilite port=step_index bundle=control
#pragma HLS INTERFACE s_axilite port=threshold_q15 bundle=control
#pragma HLS INTERFACE s_axilite port=warmup_steps bundle=control
#pragma HLS INTERFACE s_axilite port=max_consecutive_skips bundle=control
#pragma HLS INTERFACE s_axilite port=reset_state bundle=control
#pragma HLS INTERFACE s_axilite port=dot_out bundle=control
#pragma HLS INTERFACE s_axilite port=norm_x_out bundle=control
#pragma HLS INTERFACE s_axilite port=norm_y_out bundle=control
#pragma HLS INTERFACE s_axilite port=threshold_passed_out bundle=control
#pragma HLS INTERFACE s_axilite port=skip_streak_out bundle=control
#pragma HLS INTERFACE s_axilite port=distance_threshold_q20 bundle=control
#pragma HLS INTERFACE s_axilite port=distance_passed_out bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static data_t reference[MAX_VECTOR_LEN];
#pragma HLS BIND_STORAGE variable=reference type=ram_1p impl=bram
    static ap_uint<1> has_reference = 0;
    static int reference_length = 0;
    static int consecutive_skips = 0;

    *dot_out = 0;
    *norm_x_out = 0;
    *norm_y_out = 0;
    *threshold_passed_out = 0;
    *skip_streak_out = 0;
    *distance_passed_out = 0;

    if (reset_state != 0) {
        has_reference = 0;
        reference_length = 0;
        consecutive_skips = 0;
        return 0;
    }
    if (length <= 0 || length > MAX_VECTOR_LEN) {
        return 0;
    }

    bool comparable = has_reference && reference_length == length;
    acc_t dot = 0;
    norm_t norm_x = 0;
    norm_t norm_y = 0;

    if (comparable) {
        for (int i = 0; i < MAX_VECTOR_LEN; i++) {
#pragma HLS PIPELINE II=1
            if (i < length) {
                ap_int<16> xi = x[i];
                ap_int<16> yi = reference[i];
                dot += xi * yi;
                norm_x += xi * xi;
                norm_y += yi * yi;
            }
        }
    }

    *dot_out = dot;
    *norm_x_out = norm_x;
    *norm_y_out = norm_y;

    bool threshold_passed = false;
    if (comparable && dot > 0 && norm_x != 0 && norm_y != 0) {
        ap_uint<96> dot_abs = (ap_uint<96>)dot;
        ap_uint<128> left = (ap_uint<128>)dot_abs * dot_abs;
        ap_uint<128> norm_prod = (ap_uint<128>)norm_x * (ap_uint<128>)norm_y;
        ap_uint<64> threshold = threshold_q15 > 0
            ? (ap_uint<64>)threshold_q15
            : (ap_uint<64>)0;
        ap_uint<128> threshold_sq =
            (ap_uint<128>)threshold * (ap_uint<128>)threshold;
        ap_uint<128> right = (norm_prod * threshold_sq) >> 30;
        threshold_passed = left > right;
    }
    *threshold_passed_out = threshold_passed ? 1 : 0;

    bool distance_passed = false;
    if (comparable && (norm_x != 0 || norm_y != 0)) {
        ap_uint<65> norm_sum = (ap_uint<65>)norm_x + (ap_uint<65>)norm_y;
        ap_int<66> distance_signed =
            (ap_int<66>)norm_sum - ((ap_int<66>)dot << 1);
        ap_uint<66> distance_num = distance_signed > 0
            ? (ap_uint<66>)distance_signed
            : (ap_uint<66>)0;
        ap_uint<32> distance_threshold = distance_threshold_q20 > 0
            ? (ap_uint<32>)distance_threshold_q20
            : (ap_uint<32>)0;
        ap_uint<128> distance_left = (ap_uint<128>)distance_num << 20;
        ap_uint<128> distance_right =
            (ap_uint<128>)distance_threshold * (ap_uint<128>)norm_sum;
        distance_passed = distance_left <= distance_right;
    }
    *distance_passed_out = distance_passed ? 1 : 0;

    bool should_skip = comparable
        && threshold_passed
        && distance_passed
        && step_index >= warmup_steps
        && consecutive_skips < max_consecutive_skips;

    if (should_skip) {
        consecutive_skips++;
    } else {
        for (int i = 0; i < MAX_VECTOR_LEN; i++) {
#pragma HLS PIPELINE II=1
            if (i < length) {
                reference[i] = x[i];
            }
        }
        has_reference = 1;
        reference_length = length;
        consecutive_skips = 0;
    }

    *skip_streak_out = consecutive_skips;
    return should_skip ? 1 : 0;
}
