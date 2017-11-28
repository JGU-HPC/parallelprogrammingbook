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
    const index_t num_streams = 32;
    const index_t batch_size = length/num_streams;

    cudaStream_t streams[num_streams];
    for (index_t streamID = 0; streamID < num_streams; streamID++)
        cudaStreamCreate(streams+streamID);                        CUERR

    value_t * data = nullptr, * Data = nullptr;

    cudaMallocHost(&data, sizeof(value_t)*length);                 CUERR
    cudaMalloc    (&Data, sizeof(value_t)*length);                 CUERR

    for (index_t index = 0; index < length; index++)
        data[index] = index;

    TIMERSTART(overall)
    for (index_t streamID = 0; streamID < num_streams; streamID++) {
        const index_t offset = streamID*batch_size;
        cudaMemcpyAsync(Data+offset, data+offset,
                        sizeof(value_t)*batch_size,
                        cudaMemcpyHostToDevice, streams[streamID]); CUERR
    }

    for (index_t streamID = 0; streamID < num_streams; streamID++) {
        const index_t offset = streamID*batch_size;
        square_root_kernel<<<1024, 1024, 0, streams[streamID]>>>
                          (Data+offset, batch_size);                CUERR
    }

    for (index_t streamID = 0; streamID < num_streams; streamID++) {
        const index_t offset = streamID*batch_size;
        cudaMemcpyAsync(data+offset, Data+offset,
                        sizeof(value_t)*batch_size,
                        cudaMemcpyDeviceToHost, streams[streamID]); CUERR
    }

    cudaDeviceSynchronize();
    TIMERSTOP(overall)



    for (index_t index = 0; index < 10; index++)
        std::cout << index << " " << data[index] << std::endl;

    for (index_t streamID = 0; streamID < num_streams; streamID++)
            cudaStreamDestroy(streams[streamID]);                  CUERR

    cudaFreeHost(data);                                            CUERR
    cudaFree(Data);                                                CUERR
}
