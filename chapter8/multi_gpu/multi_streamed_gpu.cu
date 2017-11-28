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

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    const index_t batch_size = length/(num_gpus*num_streams);

    value_t * data = nullptr, * Data[num_gpus];
    cudaStream_t streams[num_gpus][num_streams];

    cudaMallocHost(&data, sizeof(value_t)*length);                 CUERR

    for (index_t gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        cudaMalloc(&Data[gpu],
                   sizeof(value_t)*batch_size*num_streams);        CUERR

        for (index_t streamID = 0; streamID < num_streams; streamID++)
            cudaStreamCreate(&streams[gpu][streamID]);             CUERR
    }

    for (index_t index = 0; index < length; index++)
        data[index] = index;

    TIMERSTART(overall)
    for (index_t gpu = 0; gpu < num_gpus; gpu++) {
        const index_t offset = gpu*num_streams*batch_size;
        cudaSetDevice(gpu);                                        CUERR

        for (index_t streamID = 0; streamID < num_streams; streamID++) {
            const index_t loc_off = streamID*batch_size;
            const index_t glb_off = loc_off+offset;
            cudaMemcpyAsync(Data[gpu]+loc_off, data+glb_off,
                       sizeof(value_t)*batch_size,
                       cudaMemcpyHostToDevice,
                       streams[gpu][streamID]);                    CUERR
        }
    }

    for (index_t gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);                                        CUERR
        for (index_t streamID = 0; streamID < num_streams; streamID++) {
            const index_t offset = streamID*batch_size;
            square_root_kernel<<<1024, 1024, 0, streams[gpu][streamID]>>>
                              (Data[gpu]+offset, batch_size);      CUERR
        }
    }

    for (index_t gpu = 0; gpu < num_gpus; gpu++) {
        const index_t offset = gpu*num_streams*batch_size;
        cudaSetDevice(gpu);                                        CUERR

        for (index_t streamID = 0; streamID < num_streams; streamID++) {
            const index_t loc_off = streamID*batch_size;
            const index_t glb_off = loc_off+offset;
            cudaMemcpyAsync(data+glb_off, Data[gpu]+loc_off,
                       sizeof(value_t)*batch_size,
                       cudaMemcpyDeviceToHost,
                       streams[gpu][streamID]);                    CUERR
        }
    }

    for (index_t gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        cudaDeviceSynchronize();
    }
    TIMERSTOP(overall)


    for (index_t index = 0; index < length; index += batch_size/10)
        std::cout << index << " " << data[index] << std::endl;

    cudaFreeHost(data);                                            CUERR
    for (index_t gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        cudaFree(Data[gpu]);                                       CUERR

        for (index_t streamID = 0; streamID < num_streams; streamID++)
            cudaStreamDestroy(streams[gpu][streamID]);             CUERR
    }
}
