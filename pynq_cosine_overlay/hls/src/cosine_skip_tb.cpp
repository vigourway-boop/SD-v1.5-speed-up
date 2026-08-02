#include <iostream>
#include "cosine_skip.h"

static int run_controller(
    data_t *x,
    int length,
    int step,
    int warmup,
    int max_skips,
    int reset,
    acc_t *dot,
    norm_t *norm_x,
    norm_t *norm_y,
    int *threshold_passed,
    int *skip_streak
) {
    return cosine_skip(
        x,
        length,
        step,
        (int)(0.999 * 32768.0),
        warmup,
        max_skips,
        reset,
        dot,
        norm_x,
        norm_y,
        threshold_passed,
        skip_streak
    );
}

int main() {
    static data_t x[MAX_VECTOR_LEN];
    const int length = 1024;
    for (int i = 0; i < length; i++) {
        x[i] = (data_t)((i % 255) - 127);
    }

    acc_t dot = 0;
    norm_t norm_x = 0;
    norm_t norm_y = 0;
    int threshold_passed = 0;
    int skip_streak = 0;

    int decision = run_controller(
        x, 0, 0, 2, 2, 1,
        &dot, &norm_x, &norm_y, &threshold_passed, &skip_streak
    );
    if (decision || threshold_passed || skip_streak || dot || norm_x || norm_y) {
        std::cerr << "Reset did not clear controller state" << std::endl;
        return 1;
    }

    decision = run_controller(
        x, length, 0, 2, 2, 0,
        &dot, &norm_x, &norm_y, &threshold_passed, &skip_streak
    );
    if (decision || threshold_passed || skip_streak) {
        std::cerr << "First vector must establish the reference" << std::endl;
        return 1;
    }

    decision = run_controller(
        x, length, 1, 2, 2, 0,
        &dot, &norm_x, &norm_y, &threshold_passed, &skip_streak
    );
    if (decision || !threshold_passed || skip_streak) {
        std::cerr << "Warmup must execute despite a matching vector" << std::endl;
        return 1;
    }
    if ((long long)dot <= 0 || norm_x == 0 || norm_y == 0) {
        std::cerr << "Cosine statistics were not produced" << std::endl;
        return 1;
    }

    decision = run_controller(
        x, length, 2, 2, 2, 0,
        &dot, &norm_x, &norm_y, &threshold_passed, &skip_streak
    );
    if (!decision || !threshold_passed || skip_streak != 1) {
        std::cerr << "First allowed skip failed" << std::endl;
        return 1;
    }

    decision = run_controller(
        x, length, 3, 2, 2, 0,
        &dot, &norm_x, &norm_y, &threshold_passed, &skip_streak
    );
    if (!decision || skip_streak != 2) {
        std::cerr << "Second allowed skip failed" << std::endl;
        return 1;
    }

    decision = run_controller(
        x, length, 4, 2, 2, 0,
        &dot, &norm_x, &norm_y, &threshold_passed, &skip_streak
    );
    if (decision || !threshold_passed || skip_streak != 0) {
        std::cerr << "Maximum consecutive skip limit failed" << std::endl;
        return 1;
    }

    for (int i = 0; i < length; i++) {
        x[i] = (data_t)(((i * 17) % 253) - 126);
    }
    decision = run_controller(
        x, length, 5, 2, 2, 0,
        &dot, &norm_x, &norm_y, &threshold_passed, &skip_streak
    );
    if (decision || threshold_passed || skip_streak != 0) {
        std::cerr << "Different vector should update the reference" << std::endl;
        return 1;
    }

    std::cout << "cosine_skip int8 controller test passed" << std::endl;
    return 0;
}
