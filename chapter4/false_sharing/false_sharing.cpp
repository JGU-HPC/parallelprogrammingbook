#include "../include/hpc_helpers.hpp"

#include <thread>

struct pack_t {
    uint64_t ying;
    uint64_t yang;

    pack_t() : ying(0), yang(0) {}
};

void sequential_increment(
    volatile pack_t& pack) {

    for (uint64_t index = 0; index < 1UL << 30; index++) {
        pack.ying++;
        pack.yang++;
    }
}


void false_sharing_increment(
    volatile pack_t& pack) {

    auto eval_ying = [&pack] () -> void {
        for (uint64_t index = 0; index < 1UL << 30; index++)
            pack.ying++;
    };

    auto eval_yang = [&pack] () -> void {
        for (uint64_t index = 0; index < 1UL << 30; index++)
            pack.yang++;
    };

    std::thread ying_thread(eval_ying);
    std::thread yang_thread(eval_yang);
    ying_thread.join();
    yang_thread.join();
}

int main(int argc, char* argv[]) {

    pack_t seq_pack;

    TIMERSTART(sequential_increment)
    sequential_increment(seq_pack);
    TIMERSTOP(sequential_increment)

    std::cout << seq_pack.ying << " " << seq_pack.yang << std::endl;

    pack_t par_pack;

    TIMERSTART(false_sharing_increment_increment)
    false_sharing_increment(par_pack);
    TIMERSTOP(false_sharing_increment_increment)

    std::cout << par_pack.ying << " " << par_pack.yang << std::endl;
}
