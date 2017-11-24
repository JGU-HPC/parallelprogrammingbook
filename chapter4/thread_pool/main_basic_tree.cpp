#include <iostream>
#include "threadpool_basic.hpp"

ThreadPool TP(8);

int main () {

    auto square = [](const uint64_t x) {
        return x*x;
    };

    const uint64_t num_nodes = 32;
    std::vector<std::future<uint64_t>> futures;

    typedef std::function<void(uint64_t)> traverse_t;
    traverse_t traverse = [&] (uint64_t node){
        if (node < num_nodes) {

            // submit the job
            auto future = TP.enqueue(square, node);
            futures.emplace_back(std::move(future));

            // traverse a complete binary tree
            traverse(2*node+1);
            traverse(2*node+2);
        }
    };

    // start at the root node
    traverse(0);

    // get the results
    for (auto& future : futures)
        std::cout << future.get() << std::endl;
}
