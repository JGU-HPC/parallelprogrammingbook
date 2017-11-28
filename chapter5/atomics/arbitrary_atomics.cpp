#include <iostream>
#include <cstdint>
#include <vector>
#include <thread>
#include <atomic>
#include "../include/hpc_helpers.hpp"

template <
    typename atomc_t,
    typename value_t,
    typename funct_t,
    typename predc_t>
value_t binary_atomic(
    atomc_t& atomic,
    const value_t& operand,
    funct_t function,
    predc_t predicate) {

    value_t expect = atomic.load();
    value_t target;

    do {
        // compute preliminary new value
        target = function(expect, operand);

        // immediately return if not fulfilling
        // the given constraint for a valid result
        if (!predicate(target))
            return expect;

    // try to atomically swap new and old value
    } while (!atomic.compare_exchange_weak(expect, target));

    // either new value if successful or the old
    // value for unsuccessful swap attempts:
    // in both cases it corresponds to atomic.load()
    return expect;
}


int main( ) {

    std::vector<std::thread> threads;
    const uint64_t num_threads = 10;
    const uint64_t num_iters = 100'000'000;

    auto even_max =
        [&] (volatile std::atomic<uint64_t>* counter,
             const auto& id) -> void {

        auto func = [] (const auto& lhs,
                        const auto& rhs) {
            return lhs > rhs ? lhs : rhs;
        };

        auto pred = [] (const auto& val) {
            return val % 2 == 0;
        };

        for (uint64_t i = id; i < num_iters; i += num_threads)
            binary_atomic(*counter, i, func, pred);
    };

    TIMERSTART(even_max)
    std::atomic<uint64_t> even_counter(0);
    for (uint64_t id = 0; id < num_threads; id++)
        threads.emplace_back(even_max, &even_counter, id);
    for (auto& thread : threads)
        thread.join();
    TIMERSTOP(even_max)

    std::cout << even_counter << std::endl;
}
