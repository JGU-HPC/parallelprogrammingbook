#include <iostream>
#include <cstdint>
#include <vector>
#include <thread>
#include <atomic>
#include "../include/hpc_helpers.hpp"

int main( ) {

    std::vector<std::thread> threads;
    const uint64_t num_threads = 10;
    const uint64_t num_iters = 100'000'000;

    // WARNING: this closure produces incorrect results
    auto false_max =
        [&] (volatile std::atomic<uint64_t>* counter,
             const auto& id) -> void {

        for (uint64_t i = id; i < num_iters; i += num_threads)
            if(i > *counter)
                *counter = i;
    };

    auto correct_max =
        [&] (volatile std::atomic<uint64_t>* counter,
             const auto& id) -> void {

        for (uint64_t i = id; i < num_iters; i += num_threads) {
            auto previous = counter->load();
            while (previous < i &&
                !counter->compare_exchange_weak(previous, i)) {}
        }
    };

    TIMERSTART(incorrect_max)
    std::atomic<uint64_t> false_counter(0);
    threads.clear();
    for (uint64_t id = 0; id < num_threads; id++)
        threads.emplace_back(false_max, &false_counter, id);
    for (auto& thread : threads)
        thread.join();
    TIMERSTOP(incorrect_max)

    TIMERSTART(correct_max)
    std::atomic<uint64_t> correct_counter(0);
    threads.clear();
    for (uint64_t id = 0; id < num_threads; id++)
        threads.emplace_back(correct_max, &correct_counter, id);
    for (auto& thread : threads)
        thread.join();
    TIMERSTOP(correct_max)

    std::cout << false_counter << " "
              << correct_counter << std::endl;
}
