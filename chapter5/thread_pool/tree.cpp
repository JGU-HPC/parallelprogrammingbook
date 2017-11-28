#include <iostream>
#include <cstdint>
#include "threadpool.hpp"
#include "../include/hpc_helpers.hpp"

ThreadPool TP(8);

void waste_cycles(uint64_t num_cycles) {

    volatile uint64_t counter = 0;
    for (uint64_t i = 0; i < num_cycles; i++)
        counter++;
}

void traverse(uint64_t node, uint64_t num_nodes) {

    if (node < num_nodes) {

        waste_cycles(1<<15);

        TP.spawn(traverse, 2*node+1, num_nodes);
        traverse(2*node+2, num_nodes);
    }
}

int main() {

    TIMERSTART(traverse)
    TP.spawn(traverse, 0, 1<<20);
    TP.wait_and_stop();
    TIMERSTOP(traverse)

}
