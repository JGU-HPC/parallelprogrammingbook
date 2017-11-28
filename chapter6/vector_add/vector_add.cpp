#include <iostream>
#include <cstdint>
#include <vector>
#include <omp.h>

#include "../include/hpc_helpers.hpp"

int main() {

    TIMERSTART(alloc)
    const uint64_t num_entries = 1UL << 30;
    std::vector<no_init_t<uint64_t>> x(num_entries);
    std::vector<no_init_t<uint64_t>> y(num_entries);
    std::vector<no_init_t<uint64_t>> z(num_entries);
    TIMERSTOP(alloc)

    TIMERSTART(init)
    #pragma omp parallel for
    for (uint64_t i = 0; i < num_entries; i++) {
        x[i] = i;
        y[i] = num_entries-i;
    }
    TIMERSTOP(init)

    TIMERSTART(add)
    #pragma omp parallel for
    for (uint64_t i = 0; i < num_entries; i++)
        z[i] = x[i]+y[i];
    TIMERSTOP(add)

    TIMERSTART(check)
    #pragma omp parallel for
    for (uint64_t i = 0; i < num_entries; i++)
        if(z[i]-num_entries)
            std::cout << "error at position "
                      << i << std::endl;
    TIMERSTOP(check)
}
