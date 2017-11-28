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

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    const index_t batch_size = length/num_gpus;

    value_t * data = nullptr, * Data[num_gpus];

    cudaMallocHost(&data, sizeof(value_t)*length);                 CUERR

    for (index_t gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        cudaMalloc(&Data[gpu], sizeof(value_t)*batch_size);        CUERR
    }

    for (index_t index = 0; index < length; index++)
        data[index] = index;

    TIMERSTART(overall)
    for (index_t gpu = 0; gpu < num_gpus; gpu++) {
        const index_t offset = gpu*batch_size;
        cudaSetDevice(gpu);                                        CUERR
        cudaMemcpy(Data[gpu], data+offset, sizeof(value_t)*batch_size,
                   cudaMemcpyHostToDevice);                        CUERR
    }

    for (index_t gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);                                        CUERR
        square_root_kernel<<<1024, 1024>>>(Data[gpu], batch_size); CUERR
    }

    for (index_t gpu = 0; gpu < num_gpus; gpu++) {
        const index_t offset = gpu*batch_size;
        cudaSetDevice(gpu);                                        CUERR
        cudaMemcpy(data+offset, Data[gpu], sizeof(value_t)*batch_size,
                   cudaMemcpyDeviceToHost);                        CUERR
    }
    TIMERSTOP(overall)

    for (index_t index = 0; index < length; index += batch_size/10)
        std::cout << index << " " << data[index] << std::endl;

    cudaFreeHost(data);                                            CUERR
    for (index_t gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        cudaFree(Data[gpu]);                                       CUERR
    }
}
