#include <iostream>
#include <cstdint>
#include <vector>
#include <thread>
#include <atomic>
#include "../include/hpc_helpers.hpp"

template <
    typename atomc_t,
    typename value_t,
    typename funcp_t,
    typename funcn_t,
    typename predc_t>
value_t ternary_atomic(
    atomc_t& atomic,
    const value_t& operand,
    funcp_t pos_function,
    funcn_t neg_function,
    predc_t predicate) {

    value_t expect = atomic.load();
    value_t target;

    do {

        if (predicate(expect, operand))
            target = pos_function(expect, operand);
        else
            target = neg_function(expect, operand);

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

        auto pos_func = [] (const auto& lhs,
                            const auto& rhs) {
            return lhs;
        };

        auto neg_func = [] (const auto& lhs,
                            const auto& rhs) {
            return rhs;
        };

        auto pred = [] (const auto& lhs,
                        const auto& rhs) {
            return lhs > rhs && lhs % 2 == 0;
        };

        for (uint64_t i = id; i < num_iters; i += num_threads)
            ternary_atomic(*counter, i, pos_func, neg_func, pred);
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
