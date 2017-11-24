#include "../include/hpc_helpers.hpp"

#include <iostream>
#include <cstdint>
#include <vector>
#include <thread>

template <
    typename value_t,
    typename index_t>
void init(
    std::vector<value_t>& A,
    std::vector<value_t>& x,
    index_t m,
    index_t n) {

    for (index_t row = 0; row < m; row++)
        for (index_t col = 0; col < n; col++)
            A[row*n+col] = row >= col ? 1 : 0;

    for (index_t col = 0; col < m; col++)
        x[col] = col;
}

template <
    typename value_t,
    typename index_t>
void sequential_mult(
    std::vector<value_t>& A,
    std::vector<value_t>& x,
    std::vector<value_t>& b,
    index_t m,
    index_t n) {

    for (index_t row = 0; row < m; row++) {
        value_t accum = value_t(0);
        for (index_t col = 0; col < n; col++)
            accum += A[row*n+col]*x[col];
        b[row] = accum;
    }
}

template <
    typename value_t,
    typename index_t>
void cyclic_parallel_mult(
    std::vector<value_t>& A, // linear memory for A
    std::vector<value_t>& x, // to be mapped vector
    std::vector<value_t>& b, // result vector
    index_t m,               // number of rows
    index_t n,               // number of cols
    index_t num_threads=8) { // number of threads p

    // this  function  is  called  by the  threads
    auto cyclic = [&] (const index_t& id) -> void {

        // indices are incremented with a stride of p
        for (index_t row = id; row < m; row += num_threads) {
            value_t accum = value_t(0);
	    for (index_t col = 0; col < n; col++)
                accum += A[row*n+col]*x[col];
            b[row] = accum;
        }
    };

    // business as usual
    std::vector<std::thread> threads;

    for (index_t id = 0; id < num_threads; id++)
        threads.emplace_back(cyclic, id);

    for (auto& thread : threads)
        thread.join();
}

template <
    typename value_t,
    typename index_t>
void block_parallel_mult(
    std::vector<value_t>& A,
    std::vector<value_t>& x,
    std::vector<value_t>& b,
    index_t m,
    index_t n,
    index_t num_threads=32) {

    // this function is called by the threads
    auto block = [&] (const index_t& id) -> void {
        //        ^-- capture whole scope by reference

        // compute chunk size, lower and upper task id
        const index_t chunk = SDIV(m, num_threads);
        const index_t lower = id*chunk;
        const index_t upper = std::min(lower+chunk, m);

        // only computes rows between lower and upper
        for (index_t row = lower; row < upper; row++) {
            value_t accum = value_t(0);
            for (index_t col = 0; col < n; col++)
                accum += A[row*n+col]*x[col];
            b[row] = accum;
        }
    };

    // business as usual
    std::vector<std::thread> threads;

    for (index_t id = 0; id < num_threads; id++)
        threads.emplace_back(block, id);

    for (auto& thread : threads)
        thread.join();
}


template <
    typename value_t,
    typename index_t>
void block_cyclic_parallel_mult(
    std::vector<value_t>& A,
    std::vector<value_t>& x,
    std::vector<value_t>& b,
    index_t m,
    index_t n,
    index_t num_threads=8,
    index_t chunk_size=64/sizeof(value_t)) {


    // this  function  is  called  by the  threads
    auto block_cyclic = [&] (const index_t& id) -> void {

        // precomupute the stride
	const index_t stride = num_threads*chunk_size;
	const index_t offset = id*chunk_size;

        // for each block of size chunk_size in cyclic order
        for (index_t lower = offset; lower < m; lower += stride) {

            // compute the upper border of the block
            const index_t upper = std::min(lower+chunk_size, m);

	    // for each row in the block
            for (index_t row = lower; row < upper; row++) {

		// accumulate the contributions
		value_t accum = value_t(0);
		for (index_t col = 0; col < n; col++)
                    accum += A[row*n+col]*x[col];
                b[row] = accum;
            }
	}
    };

    // business as usual
    std::vector<std::thread> threads;

    for (index_t id = 0; id < num_threads; id++)
        threads.emplace_back(block_cyclic, id);

    for (auto& thread : threads)
        thread.join();
}



int main(int argc, char* argv[]) {

    const uint64_t n = 1UL << 15;
    const uint64_t m = 1UL << 15;

    TIMERSTART(overall)
    TIMERSTART(alloc)
    std::vector<no_init_t<uint64_t>> A(m*n);
    std::vector<no_init_t<uint64_t>> x(n);
    std::vector<no_init_t<uint64_t>> b(m);
    TIMERSTOP(alloc)

    TIMERSTART(init)
    init(A, x, m, n);
    TIMERSTOP(init)

    TIMERSTART(mult)
    block_cyclic_parallel_mult(A, x, b, m, n);
    TIMERSTOP(mult)

    TIMERSTOP(overall)

    //for (const auto& entry: b)
    //    std::cout << entry << std::endl;

    for (uint64_t index = 0; index < m; index++)
        if (b[index] != index*(index+1)/2)
            std::cout << "error at position " << index << " "
                      << b[index] << std::endl;

}
