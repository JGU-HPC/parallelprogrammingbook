#include <iostream>     // std::cout
#include <cstdint>      // uint64_t
#include <cmath>        // INFINITY
#include <random>       // random
#include <immintrin.h>  // AVX intrinsics

struct avxop {

    __m256 neutral;

    avxop() : neutral(_mm256_set1_ps(-INFINITY)) {}

    inline __m256 operator()(
        const __m256& lhs,
        const __m256& rhs) const {

        return _mm256_max_ps(lhs, rhs);
    }
};

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

int main () {

    const uint64_t num_entries = 1UL << 28;
    const uint64_t num_bytes = num_entries*sizeof(float);
    auto data = static_cast<float*>(_mm_malloc(num_bytes , 32));
    init(data, num_entries);
 
    #pragma omp declare reduction(avx_max : __m256 :  \
    omp_out = avxop()(omp_out, omp_in))               \
    initializer (omp_priv=avxop().neutral)

    auto result = avxop().neutral;

    # pragma omp parallel for reduction(avx_max:result)
    for (uint64_t i = 0; i < num_entries; i += 8)
        result = avxop()(result, _mm256_load_ps(data+i));

    std::cout << hmax_avx(result) << std::endl;

    _mm_free(data);
}
