#include <random>       // prng
#include <cstdint>      // uint32_t
#include <iostream>     // std::cout
#include <immintrin.h>  // AVX intrinsics

// timers distributed with this book
#include "../include/hpc_helpers.hpp"

void init(float * data, uint64_t length) {

    std::mt19937 engine(42);
    std::uniform_real_distribution<float> density(-1L<<28, 1L<<28);

    for (uint64_t i = 0; i < length; i++)
        data[i] = density(engine);
}

inline float hmax_sse3(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);        // broadcast elements 3,1 to 2,0
    __m128 maxs = _mm_max_ps(v, shuf);
    shuf        = _mm_movehl_ps(shuf, maxs); // high half -> low half
    maxs        = _mm_max_ss(maxs, shuf);
    return        _mm_cvtss_f32(maxs);
}

inline float hmax_avx(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);   // low 128
    __m128 hi = _mm256_extractf128_ps(v, 1); // high 128
           lo = _mm_max_ps(lo, hi);          // max the low 128
    return hmax_sse3(lo);                    // and inline the sse3 version
}

float avx_max(float * data, uint64_t length) {

    // neutral element "e" in monoid (|R, max) is -oo
    const float e = -INFINITY;
    __m256 X = _mm256_set1_ps(e);

    for (uint64_t i = 0; i < length; i += 8) {
        __m256 DATA = _mm256_load_ps(data+i);
        X = _mm256_max_ps(X, DATA);
    }

    return hmax_avx(X);
}

float avx_max_unroll_2(float * data, uint64_t length) {

    // neutral element "e" in monoid (|R, max) is -oo
    const float e = -INFINITY;
    __m256 X = _mm256_set1_ps(e);
    __m256 Y = _mm256_set1_ps(e);

    for (uint64_t i = 0; i < length; i += 16) {
        __m256 DATA_X = _mm256_load_ps(data+i+0);
        __m256 DATA_Y = _mm256_load_ps(data+i+8);
        X = _mm256_max_ps(X, DATA_X);
        Y = _mm256_max_ps(Y, DATA_Y);
    }

    return std::max(hmax_avx(X), hmax_avx(Y));
}

float plain_max(float * data, uint64_t length) {

    // neutral element "e" in monoid (|R, max) is -oo
    float max = -INFINITY;

    for (uint64_t i = 0; i < length; i++)
        max = std::max(max, data[i]);

    return max;
}

float plain_max_unroll_2(float * data, uint64_t length) {

    // neutral element "e" in monoid (|R, max) is -oo
    float max_0 = -INFINITY;
    float max_1 = -INFINITY;

    for (uint64_t i = 0; i < length; i += 2) {
        max_0 = std::max(max_0, data[i+0]);
        max_1 = std::max(max_1, data[i+1]);
    }

    return std::max(max_0, max_1);
}

float plain_max_unroll_4(float * data, uint64_t length) {

    // neutral element "e" in monoid (|R, max) is -oo
    float max_0 = -INFINITY;
    float max_1 = -INFINITY;
    float max_2 = -INFINITY;
    float max_3 = -INFINITY;

    for (uint64_t i = 0; i < length; i += 4) {
        max_0 = std::max(max_0, data[i+0]);
        max_1 = std::max(max_1, data[i+1]);
        max_2 = std::max(max_2, data[i+2]);
        max_3 = std::max(max_3, data[i+3]);
    }

    return std::max(max_0,
           std::max(max_1,
           std::max(max_2, max_3)));
}

float plain_max_unroll_8(float * data, uint64_t length) {

    // neutral element "e" in monoid (|R, max) is -oo
    float max_0 = -INFINITY;
    float max_1 = -INFINITY;
    float max_2 = -INFINITY;
    float max_3 = -INFINITY;
    float max_4 = -INFINITY;
    float max_5 = -INFINITY;
    float max_6 = -INFINITY;
    float max_7 = -INFINITY;

    for (uint64_t i = 0; i < length; i += 8) {
        max_0 = std::max(max_0, data[i+0]);
        max_1 = std::max(max_1, data[i+1]);
        max_2 = std::max(max_2, data[i+2]);
        max_3 = std::max(max_3, data[i+3]);
        max_4 = std::max(max_4, data[i+0]);
        max_5 = std::max(max_5, data[i+1]);
        max_6 = std::max(max_6, data[i+2]);
        max_7 = std::max(max_7, data[i+3]);
    }

    return std::max(max_0,
           std::max(max_1,
           std::max(max_2,
           std::max(max_3,
           std::max(max_4,
           std::max(max_5,
           std::max(max_6, max_7)))))));
}

int main () {

    const uint64_t num_entries = 1UL << 28;
    const uint64_t num_bytes = num_entries*sizeof(float);

    TIMERSTART(alloc_memory)
    auto data = static_cast<float*>(_mm_malloc(num_bytes , 32));
    TIMERSTOP(alloc_memory)

    TIMERSTART(init)
    init(data, num_entries);
    TIMERSTOP(init)

    TIMERSTART(plain_max)
    std::cout << plain_max(data, num_entries) << std::endl;
    TIMERSTOP(plain_max)

    TIMERSTART(plain_max_unroll_2)
    std::cout << plain_max_unroll_2(data, num_entries) << std::endl;
    TIMERSTOP(plain_max_unroll_2)

    TIMERSTART(plain_max_unroll_4)
    std::cout << plain_max_unroll_4(data, num_entries) << std::endl;
    TIMERSTOP(plain_max_unroll_4)

    TIMERSTART(plain_max_unroll_8)
    std::cout << plain_max_unroll_8(data, num_entries) << std::endl;
    TIMERSTOP(plain_max_unroll_8)

    TIMERSTART(avx_max)
    std::cout << avx_max(data, num_entries) << std::endl;
    TIMERSTOP(avx_max)

    TIMERSTART(avx_max_unroll_2)
    std::cout << avx_max_unroll_2(data, num_entries) << std::endl;
    TIMERSTOP(avx_max_unroll_2)

    TIMERSTART(free_memory)
    _mm_free(data);
    TIMERSTOP(free_memory)
}
