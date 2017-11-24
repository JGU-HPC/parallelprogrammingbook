#include <iostream>
#include "threadpool_basic.hpp"

ThreadPool TP(8);

int main () {

    auto square = [](const uint64_t x) {
        return x*x;
    };

    const uint64_t num_tasks = 32;
    std::vector<std::future<uint64_t>> futures;

    for (uint64_t task = 0; task < num_tasks; task++) {
        auto future = TP.enqueue(square, task);
        futures.emplace_back(std::move(future));
    }

    for (auto& future : futures)
        std::cout << future.get() << std::endl;
}
