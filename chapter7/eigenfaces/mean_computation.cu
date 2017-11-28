#include "../include/hpc_helpers.hpp"
#include "../include/binary_IO.hpp"
#include "../include/bitmap_IO.hpp"

template <
    typename index_t,
    typename value_t> __global__
void compute_mean_kernel(
    value_t * Data,
    value_t * Mean,
    index_t num_entries,
    index_t num_features);

int main (int argc, char * argv[]) {

    // set the identifier of the used CUDA device
    cudaSetDevice(0);

    // 202599 grayscale images each of shape 55 x 45
    constexpr uint64_t imgs = 202599, rows = 55, cols = 45;

    // pointer for data matrix and mean vector
    float * data = nullptr, * mean = nullptr;
    cudaMallocHost(&data, sizeof(float)*imgs*rows*cols);                  CUERR
    cudaMallocHost(&mean, sizeof(float)*rows*cols);                       CUERR

    // allocate storage on GPU
    float * Data = nullptr, * Mean = nullptr;
    cudaMalloc(&Data, sizeof(float)*imgs*rows*cols);                      CUERR
    cudaMalloc(&Mean, sizeof(float)*rows*cols);                           CUERR

    // load data matrix from disk
    TIMERSTART(read_data_from_disk)
    std::string file_name = "./data/celebA_gray_lowres.202599_55_45_32.bin";
    load_binary(data, imgs*rows*cols, file_name);
    TIMERSTOP(read_data_from_disk)

    // copy data to device and reset Mean
    TIMERSTART(data_H2D)
    cudaMemcpy(Data, data, sizeof(float)*imgs*rows*cols,
               cudaMemcpyHostToDevice);                                   CUERR
    cudaMemset(Mean, 0, sizeof(float)*rows*cols);                         CUERR
    TIMERSTOP(data_H2D)

    // compute mean
    TIMERSTART(compute_mean_kernel)
    compute_mean_kernel<<<SDIV(rows*cols, 32), 32>>>
                       (Data, Mean, imgs, rows*cols);                     CUERR
    TIMERSTOP(compute_mean_kernel)


    // transfer mean back to host
    TIMERSTART(mean_D2H)
    cudaMemcpy(mean, Mean, sizeof(float)*rows*cols,
               cudaMemcpyDeviceToHost);                                   CUERR
    TIMERSTOP(mean_D2H)

    // write mean image to disk
    TIMERSTART(write_mean_image_to_disk)
    dump_bitmap(mean, rows, cols, "./imgs/celebA_mean.bmp");
    TIMERSTOP(write_mean_image_to_disk)

    // get rid of the memory
    cudaFreeHost(data);                                                   CUERR
    cudaFreeHost(mean);                                                   CUERR
    cudaFree(Data);                                                       CUERR
    cudaFree(Mean);                                                       CUERR

}

template <
    typename index_t,
    typename value_t> __global__
void compute_mean_kernel(
    value_t * Data,
    value_t * Mean,
    index_t num_entries,
    index_t num_features) {

    auto thid = blockDim.x*blockIdx.x + threadIdx.x;

    if (thid < num_features) {

        value_t accum = 0;

        # pragma unroll 32
        for (index_t entry = 0; entry < num_entries; entry++)
            accum += Data[entry*num_features+thid];

        Mean[thid] = accum/num_entries;
    }
}

