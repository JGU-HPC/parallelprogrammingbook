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

void plain_pointwise_max(float * x,
                         float * y,
                         float * z, uint64_t length) {

    for (uint64_t i = 0; i < length; i++)
        z[i] = std::max(x[i], y[i]);
}

void avx_pointwise_max(float * x,
                       float * y,
                       float * z, uint64_t length) {


    for (uint64_t i = 0; i < length; i += 8) {
        __m256 X = _mm256_load_ps(x+i);
        __m256 Y = _mm256_load_ps(y+i);
        _mm256_store_ps(z+i, _mm256_max_ps(X, Y));
    }
}


int main () {

    const uint64_t num_entries = 1UL << 28;
    const uint64_t num_bytes = num_entries*sizeof(float);

    TIMERSTART(alloc_memory)
    auto x = static_cast<float*>(_mm_malloc(num_bytes , 32));
    auto y = static_cast<float*>(_mm_malloc(num_bytes , 32));
    auto z = static_cast<float*>(_mm_malloc(num_bytes , 32));
    TIMERSTOP(alloc_memory)

    TIMERSTART(init)
    init(x, num_entries);
    init(y, num_entries);
    TIMERSTOP(init)

    TIMERSTART(plain_pointwise_max)
    plain_pointwise_max(x, y, z, num_entries);
    TIMERSTOP(plain_pointwise_max)

    TIMERSTART(avx_pointwise_max)
    avx_pointwise_max(x, y, z, num_entries);
    TIMERSTOP(avx_pointwise_max)

    TIMERSTART(free_memory)
    _mm_free(x);
    _mm_free(y);
    _mm_free(z);
    TIMERSTOP(free_memory)
}
