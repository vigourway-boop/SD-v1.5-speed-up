#include <cmath>
#include <cstdlib>
#include <iostream>
#include "cosine_skip.h"

int main() {
    static data_t x[MAX_VECTOR_LEN];
    static data_t y[MAX_VECTOR_LEN];

    const int length = 1024;
    long long expected_dot = 0;
    unsigned long long expected_norm_x = 0;
    for (int i = 0; i < length; i++) {
        x[i] = (data_t)((i % 257) - 128);
        y[i] = x[i];
        long long value = (long long)x[i];
        expected_dot += value * value;
        expected_norm_x += value * value;
    }

    int threshold_q15 = (int)(0.999 * 32768.0);
    acc_t dot_out = 0;
    norm_t norm_x_out = 0;
    norm_t norm_y_out = 0;
    int skip_same = cosine_skip(
        x, y, length, threshold_q15,
        &dot_out, &norm_x_out, &norm_y_out
    );
    if (skip_same != 1) {
        std::cerr << "Expected identical vectors to skip" << std::endl;
        return 1;
    }
    if ((long long)dot_out != expected_dot ||
        (unsigned long long)norm_x_out != expected_norm_x ||
        (unsigned long long)norm_y_out != expected_norm_x) {
        std::cerr << "Incorrect dot/norm outputs for identical vectors" << std::endl;
        return 1;
    }

    long long expected_diff_dot = 0;
    unsigned long long expected_norm_y = 0;
    for (int i = 0; i < length; i++) {
        y[i] = (data_t)((i * 17) % 255 - 127);
        long long xi = (long long)x[i];
        long long yi = (long long)y[i];
        expected_diff_dot += xi * yi;
        expected_norm_y += yi * yi;
    }

    int skip_diff = cosine_skip(
        x, y, length, threshold_q15,
        &dot_out, &norm_x_out, &norm_y_out
    );
    if (skip_diff != 0) {
        std::cerr << "Expected different vectors not to skip" << std::endl;
        return 1;
    }
    if ((long long)dot_out != expected_diff_dot ||
        (unsigned long long)norm_x_out != expected_norm_x ||
        (unsigned long long)norm_y_out != expected_norm_y) {
        std::cerr << "Incorrect dot/norm outputs for different vectors" << std::endl;
        return 1;
    }

    int invalid = cosine_skip(
        x, y, 0, threshold_q15,
        &dot_out, &norm_x_out, &norm_y_out
    );
    if (invalid != 0 || dot_out != 0 || norm_x_out != 0 || norm_y_out != 0) {
        std::cerr << "Invalid input should return zero outputs" << std::endl;
        return 1;
    }

    std::cout << "cosine_skip test passed" << std::endl;
    return 0;
}
