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

template <
    typename index_t,
    typename value_t> __global__
void correction_kernel(
    value_t * Data,
    value_t * Mean,
    index_t num_entries,
    index_t num_features) {

     auto thid = blockDim.x*blockIdx.x + threadIdx.x;

     if (thid < num_features) {

        value_t value = Mean[thid];

        for (index_t entry = 0; entry < num_entries; entry++)
            Data[entry*num_features+thid] -= value;

     }
}

template <
    typename index_t,
    typename value_t> __global__
void covariance_kernel(
    value_t * Data,
    value_t * Cov,
    index_t num_entries,
    index_t num_features) {

    auto j = blockDim.x*blockIdx.x + threadIdx.x;
    auto i = blockDim.y*blockIdx.y + threadIdx.y;

    if (i < num_features && j < num_features) {

        value_t accum = 0;

        for (index_t entry = 0; entry < num_entries; entry++)
            accum += Data[entry*num_features+i] *
                     Data[entry*num_features+j];


        Cov[i*num_features+j] = accum/num_entries;
    }
}

template <
    typename index_t,
    typename value_t> __global__
void symmetric_covariance_kernel(
    value_t * Data,
    value_t * Cov,
    index_t num_entries,
    index_t num_features) {

    auto j = blockDim.x*blockIdx.x + threadIdx.x;
    auto i = blockDim.y*blockIdx.y + threadIdx.y;

    if (i < num_features && j <= i) {

        value_t accum = 0;

        for (index_t entry = 0; entry < num_entries; entry++)
            accum += Data[entry*num_features+i] *
                     Data[entry*num_features+j];


        Cov[i*num_features+j] = Cov[j*num_features+i] = accum/num_entries;
    }
}



template <
    typename index_t,
    typename value_t,
    uint32_t chunk_size=32 > __global__
void shared_covariance_kernel(
    value_t * Data,
    value_t * Cov,
    index_t num_entries,
    index_t num_features) {


    // convenience variables
    const index_t base_x = blockIdx.x*chunk_size;
    const index_t base_y = blockIdx.y*chunk_size;

    const index_t thid_y = threadIdx.y;
    const index_t thid_x = threadIdx.x;

    const index_t x = base_x + thid_x;
    const index_t y = base_y + thid_y;

    // optional early exit: -500ms
    if (base_x > base_y) return;

    // allocate shared memory
    __shared__ value_t cache_x[chunk_size][chunk_size];
    __shared__ value_t cache_y[chunk_size][chunk_size];

    // compute the number of chunks to be computed
    const index_t num_chunks = SDIV(num_entries, chunk_size);

    // accumulated value of scalar product
    value_t accum = 0;

    // for each chunk
    for (index_t chunk = 0; chunk < num_chunks; chunk++) {

            // assign thread IDs to rows and columns
            const index_t row   = thid_y + chunk*chunk_size;
            const index_t col_x = thid_x + base_x;
            const index_t col_y = thid_x + base_y;

            // check if valid row or column indices
            const bool valid_row   = row   < num_entries;
            const bool valid_col_x = col_x < num_features;
            const bool valid_col_y = col_y < num_features;

            // fill shared memory with tiles where thid_y enumerates
            // image identifiers (entries) and thid_x denotes feature
            // coordinates (pixels). cache_x corresponds to x and
            // cache_y to y where Cov[x,y] is the pairwise covariance
            cache_x[thid_y][thid_x] = valid_row*valid_col_x ?
                                      Data[row*num_features+col_x] : 0;
            cache_y[thid_y][thid_x] = valid_row*valid_col_y ?
                                      Data[row*num_features+col_y] : 0;

            // this is needed to ensure that all threads finished writing
            // shared memory
            __syncthreads();

            // optional early exit: -100ms
            if (x <= y)
                // here we actually evaluate the scalar product
                for (index_t entry = 0; entry < chunk_size; entry++)
                    accum += cache_y[entry][thid_y]*cache_x[entry][thid_x];

            // this is needed to ensure that shared memory can be over-
            // written again in the next iteration
            __syncthreads();
    }

    // since Cov[x,y] = Cov[y,x] we only compute one entry
    if (y < num_features && x <= y)
        Cov[y*num_features+x] =
        Cov[x*num_features+y] = accum/num_entries;

}

int main (int argc, char * argv[]) {

    // set the identifier of the used CUDA device
    cudaSetDevice(0);
    cudaDeviceReset();

    // 202599 grayscale images each of shape 55 x 45
    constexpr uint64_t imgs = 202599, rows = 55, cols = 45;

    // pointer for data matrix and mean vector
    float * data = nullptr, * cov = nullptr;
    cudaMallocHost(&data, sizeof(float)*imgs*rows*cols);                  CUERR
    cudaMallocHost(&cov,  sizeof(float)*rows*cols*rows*cols);             CUERR

    // allocate storage on GPU
    float * Data = nullptr, * Mean = nullptr, * Cov = nullptr;
    cudaMalloc(&Data, sizeof(float)*imgs*rows*cols);                      CUERR
    cudaMalloc(&Mean, sizeof(float)*rows*cols);                           CUERR
    cudaMalloc(&Cov,  sizeof(float)*rows*cols*rows*cols);                 CUERR

    // load data matrix from disk
    TIMERSTART(read_data_from_disk)
    auto file_name = "./data/celebA_gray_lowres.202599_55_45_32.bin";
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
    compute_mean_kernel<<<SDIV(rows*cols, 1024), 1024>>>
                       (Data, Mean, imgs, rows*cols);                     CUERR
    TIMERSTOP(compute_mean_kernel)

    // correct mean
    TIMERSTART(correction_kernel)
    correction_kernel<<<SDIV(rows*cols, 1024), 1024>>>
                       (Data, Mean, imgs, rows*cols);                     CUERR
    TIMERSTOP(correction_kernel)

    // compute covariance matrix
    TIMERSTART(covariance_kernel)

    #ifdef COV_MODE_NAIVE
    dim3 blocks(SDIV(rows*cols, 8), SDIV(rows*cols, 8));
    dim3 threads(8, 8);
    covariance_kernel<<<blocks, threads>>>
                       (Data, Cov, imgs, rows*cols);                      CUERR
    #elif COV_MODE_SYMMETRIC
    dim3 blocks(SDIV(rows*cols, 8), SDIV(rows*cols, 8));
    dim3 threads(8, 8);
    symmetric_covariance_kernel<<<blocks, threads>>>
                       (Data, Cov, imgs, rows*cols);                      CUERR
    #else
    dim3 blocks(SDIV(rows*cols, 32), SDIV(rows*cols, 32));
    dim3 threads(32, 32, 1);
    shared_covariance_kernel<<<blocks, threads>>>
                       (Data, Cov, imgs, rows*cols);                      CUERR
    #endif
    
    TIMERSTOP(covariance_kernel)

    // transfer covariance back to host
    TIMERSTART(cov_D2H)
    cudaMemcpy(cov, Cov, sizeof(float)*rows*cols*rows*cols,
               cudaMemcpyDeviceToHost);                                   CUERR
    TIMERSTOP(cov_D2H)

    // write mean image to disk
    TIMERSTART(write_mean_image_to_disk)
    dump_bitmap(cov, rows*cols, rows*cols, "imgs/celebA_covariance.bmp");
    TIMERSTOP(write_mean_image_to_disk)

    // get rid of the memory
    cudaFreeHost(data);                                                   CUERR
    cudaFreeHost(cov);                                                    CUERR
    cudaFree(Data);                                                       CUERR
    cudaFree(Mean);                                                       CUERR
    cudaFree(Cov);                                                        CUERR

}
