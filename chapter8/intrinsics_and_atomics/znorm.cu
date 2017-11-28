#include "../include/cbf_generator.hpp"
#include "../include/hpc_helpers.hpp"

typedef uint64_t index_t;
typedef uint8_t label_t;
typedef float value_t;

__forceinline__ __device__
double cuda_rsqrt(const double& value) {
    return rsqrt(value);
}

__forceinline__ __device__
float cuda_rsqrt(const float& value) {
    return rsqrtf(value);
}

template <
    typename index_t,
    typename value_t> __global__
void znorm_kernel(
    value_t * Subject,       // pointer to the subject
    index_t num_entries,     // number of time series (m)
    index_t num_features) {  // number of time ticks (n)

    // get thread and block identifiers
    const index_t blid = blockIdx.x;
    const index_t thid = threadIdx.x;
    const index_t base = blid*num_features;

    // 1. coalesced loading of entries
    value_t v = Subject[base+thid];
    value_t x = v; // copy for later

    // 2a. perform a warp reduction (sum stored in thread zero)
    for (index_t offset = num_features/2; offset > 0; offset /= 2)
        x += __shfl_down(x, offset, num_features);

    // 2b. perform the first broadcast
    value_t mu = __shfl(x, 0)/num_features;

    // define the square residues
    value_t y = (v-mu)*(v-mu);

    // 3a. perform a warp reduction (sum stored in thread zero)
    for (index_t offset = num_features/2; offset > 0; offset /= 2)
        y += __shfl_down(y, offset, num_features);

    // 3b. perform the second broadcast
    value_t sigma = __shfl(y, 0)/(num_features-1);

    // 4. write result back
    Subject[base+thid] = (v-mu)*cuda_rsqrt(sigma);
}

int main () {

    constexpr index_t num_features = 32;
    constexpr index_t num_entries = 1UL << 20;

    // small letters for hosts, capital letters for device
    value_t * data = nullptr, * Data = nullptr;
    label_t * labels = nullptr;

    // malloc memory
    cudaMallocHost(&data, sizeof(value_t)*num_entries*num_features);      CUERR
    cudaMalloc    (&Data, sizeof(value_t)*num_entries*num_features);      CUERR
    cudaMallocHost(&labels, sizeof(label_t)*num_entries);                 CUERR

    // create CBF data set on host
    TIMERSTART(generate_data)
    generate_cbf(data, labels, num_entries, num_features);
    TIMERSTOP(generate_data)

    TIMERSTART(copy_data_to_device)
    cudaMemcpy(Data, data, sizeof(value_t)*num_entries*num_features, H2D);CUERR
    TIMERSTOP(copy_data_to_device)


    TIMERSTART(z_norm)
    znorm_kernel<<<num_entries, 32>>>(Data, num_entries, num_features);   CUERR
    TIMERSTOP(z_norm)

    TIMERSTART(copy_data_to_host)
    cudaMemcpy(data, Data, sizeof(value_t)*num_entries*num_features, D2H);CUERR
    TIMERSTOP(copy_data_to_host)


    value_t accum = 0, accum2=0;
    for (index_t i = 0; i < 32; i++) {
        accum  += data[i];
        accum2 += data[i]*data[i];
    }

    std::cout << accum << " " << accum2 << std::endl;

    // get rid of the memory
    cudaFreeHost(labels);
    cudaFreeHost(data);
    cudaFree(Data);
}
