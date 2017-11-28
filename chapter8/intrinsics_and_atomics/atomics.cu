#include "../include/cbf_generator.hpp"
#include "../include/hpc_helpers.hpp"

typedef uint64_t index_t;
typedef uint8_t label_t;
typedef float value_t;

template <
    typename index_t,
    typename value_t,
    index_t warp_size=32> __global__
void global_reduction_kernel(
    value_t * Input,         // pointer to the data
    value_t * Output,        // pointer to the result
    index_t   length) {      // number of entries (n)

    // get thread and block identifiers
    const index_t thid = threadIdx.x;
    const index_t blid = blockIdx.x;
    const index_t base = blid*warp_size;

    // store entries in registers
    value_t x = 0;
    if (base+thid < length)
        x = Input[base+thid];

    // do the Kepler shuffle
    for (index_t offset = warp_size/2; offset > 0; offset /= 2)
        x += __shfl_down(x, offset, warp_size);

    // write down result
    if (thid == 0)
      atomicAdd(Output, x);
}

template <
    typename index_t,
    typename value_t,
    index_t warp_size=32> __global__
void static_reduction_kernel(
    value_t * Input,         // pointer to the data
    value_t * Output,        // pointer to the result
    index_t length) {        // number of entries (n)

    // get global thread identifier
    const index_t thid = blockDim.x*blockIdx.x+threadIdx.x;

    // here we store the result
    value_t accum = value_t(0);

    // block-cyclic summation over all spawned blocks
    for (index_t i = thid; i < length; i += blockDim.x*gridDim.x)
        accum += Input[i];

    // reduce all values within a warp
    for (index_t offset = warp_size/2; offset > 0; offset /= 2)
        accum += __shfl_down(accum, offset, warp_size);

    // first thread of every warp writes result
    if (thid % 32  == 0)
      atomicAdd(Output, accum);
}


int main () {

    constexpr index_t num_features = 32;
    constexpr index_t num_entries = 1UL << 10;

    // small letters for hosts, capital letters for device
    value_t * data = nullptr, * result = nullptr,
            * Data = nullptr, * Result = nullptr;
    label_t * labels = nullptr;

    // malloc memory
    cudaMallocHost(&data, sizeof(value_t)*num_entries*num_features);      CUERR
    cudaMalloc    (&Data, sizeof(value_t)*num_entries*num_features);      CUERR
    cudaMallocHost(&result, sizeof(value_t));                             CUERR
    cudaMalloc    (&Result, sizeof(value_t));                             CUERR
    cudaMallocHost(&labels, sizeof(label_t)*num_entries);                 CUERR

    // create CBF data set on host
    TIMERSTART(generate_data)
    generate_cbf(data, labels, num_entries, num_features);
    TIMERSTOP(generate_data)

    TIMERSTART(copy_data_to_device)
    cudaMemcpy(Data, data, sizeof(value_t)*num_entries*num_features, H2D);CUERR
    cudaMemset(Result, 0, sizeof(value_t));
    TIMERSTOP(copy_data_to_device)

    value_t accum = 0;
    for (index_t i = 0; i < num_entries*num_features; i++)
        accum += data[i];
    std::cout << accum << std::endl;

    TIMERSTART(global_reduction)
    global_reduction_kernel<<<SDIV(num_entries*num_features, 32), 32>>>
                                       (Data, Result, num_entries*num_features);    CUERR
    TIMERSTOP(global_reduction)

    TIMERSTART(static_reduction)
    static_reduction_kernel<<<32, 32>>>(Data, Result, num_entries*num_features);    CUERR
    TIMERSTOP(static_reduction)

    TIMERSTART(copy_data_to_host)
    cudaMemcpy(result, Result, sizeof(value_t), D2H);                               CUERR
    TIMERSTOP(copy_data_to_host)


    std::cout << *result << std::endl;

    // get rid of the memory
    cudaFreeHost(labels);
    cudaFreeHost(result);
    cudaFreeHost(data);
    cudaFree(Result);
    cudaFree(Data);

}
