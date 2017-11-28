#include "../include/hpc_helpers.hpp"

template <
    typename index_t,
    typename value_t,
    index_t num_iters=256> __global__
void square_root_kernel(
    value_t * Data,
    index_t   length) {

    const index_t thid = blockDim.x*blockIdx.x+threadIdx.x;

    for (index_t i = thid; i < length; i += blockDim.x*gridDim.x){

        value_t value = Data[i];
        value_t root  = value;

        # pragma unroll (32)
        for (index_t iters = 0; iters < num_iters && value; iters++)
            root = 0.5*(root+value/root);

       Data[i] = root;
    }
}

int main () {

    typedef float    value_t;
    typedef uint64_t index_t;

    const index_t length = 1UL << 30;

    value_t * data = nullptr, * Data = nullptr;

    cudaMallocHost(&data, sizeof(value_t)*length);                CUERR
    cudaMalloc    (&Data, sizeof(value_t)*length);                CUERR

    for (index_t index = 0; index < length; index++)
        data[index] = index;

    TIMERSTART(overall)
    TIMERSTART(host_to_device)
    cudaMemcpy(Data, data, sizeof(value_t)*length,
               cudaMemcpyHostToDevice);                           CUERR
    TIMERSTOP(host_to_device)

    TIMERSTART(square_root_kernel)
    square_root_kernel<<<1024, 1024>>>(Data, length);             CUERR
    TIMERSTOP(square_root_kernel)

    TIMERSTART(device_to_host)
    cudaMemcpy(data, Data, sizeof(value_t)*length,
               cudaMemcpyDeviceToHost);                           CUERR
    TIMERSTOP(device_to_host)
    TIMERSTOP(overall)

    for (index_t index = 0; index < 10; index++)
        std::cout << index << " " << data[index] << std::endl;

    cudaFreeHost(data);                                           CUERR
    cudaFree(Data);                                               CUERR
}
