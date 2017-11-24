#include <iostream>
#include <cstdint>
#include <vector>
#include <thread>
#include <future>

template <
    typename value_t,
    typename index_t>
void fibo(
    value_t n,
    std::promise<value_t> && result) {

    value_t a_0 = 0;
    value_t a_1 = 1;

    for (index_t index = 0; index < n; index++) {
        const value_t tmp = a_0; a_0 = a_1; a_1 += tmp;
    }

    result.set_value(a_0);
}

int main(int argc, char * argv[]) {

    const uint64_t num_threads = 32;

    std::vector<std::thread> threads;
    std::vector<std::future<uint64_t>> results;

    for (uint64_t id = 0; id < num_threads; id++) {
        std::promise<uint64_t> promise;
        results.emplace_back(promise.get_future());

        threads.emplace_back(
            std::thread(
                fibo<uint64_t, uint64_t>, id, std::move(promise)
            )
        );
    }


    for (auto& result: results)
        std::cout << result.get() << std::endl;

    for (auto& thread: threads)
        thread.detach();

}
