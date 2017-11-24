#include <random>       // prng
#include <cstdint>      // uint32_t
#include <iostream>     // std::cout
#include <immintrin.h>  // AVX intrinsics

// timers distributed with this book
#include "../include/hpc_helpers.hpp"

void soa_init(float * x,
              float * y,
              float * z,
              uint64_t length) {

    std::mt19937 engine(42);
    std::uniform_real_distribution<float> density(-1, 1);

    for (uint64_t i = 0; i < length; i++) {
        x[i] = density(engine);
        y[i] = density(engine);
        z[i] = density(engine);
    }

}

void avx_soa_norm(float * x,
                    float * y,
                    float * z,
                    uint64_t length) {

    for (uint64_t i = 0; i < length; i += 8) {

        // aligned loads
        __m256 X = _mm256_load_ps(x+i);
        __m256 Y = _mm256_load_ps(y+i);
        __m256 Z = _mm256_load_ps(z+i);

        // R <- X*X+Y*Y+Z*Z
        __m256 R = _mm256_add_ps(_mm256_mul_ps(X, X),
                   _mm256_add_ps(_mm256_mul_ps(Y, Y),
                                 _mm256_mul_ps(Z, Z)));
        // R <- 1/sqrt(R)
               R = _mm256_rsqrt_ps(R);

        // aligned stores
        _mm256_store_ps(x+i, _mm256_mul_ps(X, R));
        _mm256_store_ps(y+i, _mm256_mul_ps(Y, R));
        _mm256_store_ps(z+i, _mm256_mul_ps(Z, R));
    }
}

void soa_check(float * x,
               float * y,
               float * z,
               uint64_t length) {

    for (uint64_t i = 0; i < length; i++) {
        float rho = x[i]*x[i]+y[i]*y[i]+z[i]*z[i];
        if ((rho-1)*(rho-1) > 1E-6)
            std::cout << "error too big at position "
                      << i << std::endl;
    }
}

int main () {

    const uint64_t num_vectors = 1UL << 28;
    const uint64_t num_bytes = num_vectors*sizeof(float);

    TIMERSTART(alloc_memory)
    auto x = static_cast<float*>(_mm_malloc(num_bytes , 32));
    auto y = static_cast<float*>(_mm_malloc(num_bytes , 32));
    auto z = static_cast<float*>(_mm_malloc(num_bytes , 32));
    TIMERSTOP(alloc_memory)

    TIMERSTART(init)
    soa_init(x, y, z, num_vectors);
    TIMERSTOP(init)

    TIMERSTART(avx_soa_normalize)
    avx_soa_norm(x, y, z, num_vectors);
    TIMERSTOP(avx_soa_normalize)

    TIMERSTART(check)
    soa_check(x, y, z, num_vectors);
    TIMERSTOP(check)

    TIMERSTART(free_memory)
    _mm_free(x);
    _mm_free(y);
    _mm_free(z);
    TIMERSTOP(free_memory)
}
