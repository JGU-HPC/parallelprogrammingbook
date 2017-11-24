#include <cstdint>      // uint32_t
#include <iostream>     // std::cout
#include <random>       // prng

// timers distributed with this book
#include "../include/hpc_helpers.hpp"

void aos_init(float * xyz, uint64_t length) {

    std::mt19937 engine(42);
    std::uniform_real_distribution<float> density(-1, 1);

    for (uint64_t i = 0; i < 3*length; i++)
        xyz[i] = density(engine);
}

void plain_aos_norm(float * xyz, uint64_t length) {

    for (uint64_t i = 0; i < 3*length; i += 3) {
        const float x = xyz[i+0];
        const float y = xyz[i+1];
        const float z = xyz[i+2];

        float irho = 1.0f/std::sqrt(x*x+y*y+z*z);

        xyz[i+0] *= irho;
        xyz[i+1] *= irho;
        xyz[i+2] *= irho;
    }
}

void aos_check(float * xyz, uint64_t length) {

    for (uint64_t i = 0; i < 3*length; i += 3) {

        const float x = xyz[i+0];
        const float y = xyz[i+1];
        const float z = xyz[i+2];

        float rho = x*x+y*y+z*z;

        if ((rho-1)*(rho-1) > 1E-6)
            std::cout << "error too big at position "
                      << i << std::endl;
    }
}

int main () {

    const uint64_t num_vectors = 1UL << 28;

    TIMERSTART(alloc_memory)
    auto xyz = new float[3*num_vectors];
    TIMERSTOP(alloc_memory)

    TIMERSTART(init)
    aos_init(xyz, num_vectors);
    TIMERSTOP(init)

    TIMERSTART(plain_aos_normalize)
    plain_aos_norm(xyz, num_vectors);
    TIMERSTOP(plain_aos_normalize)

    TIMERSTART(check)
    aos_check(xyz, num_vectors);
    TIMERSTOP(check)

    TIMERSTART(free_memory)
    delete [] xyz;
    TIMERSTOP(free_memory)
}
