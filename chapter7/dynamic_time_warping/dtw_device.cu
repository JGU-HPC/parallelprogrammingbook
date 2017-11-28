#include "../include/cbf_generator.hpp"
#include "../include/hpc_helpers.hpp"
#include "../include/binary_IO.hpp"

typedef uint64_t index_t;
typedef uint8_t label_t;
typedef float value_t;

__constant__ value_t cQuery[12*1024];
texture<value_t, 1, cudaReadModeElementType> tSubject;

template <
    typename index_t,
    typename value_t> __global__
void DTW_naive_kernel(
    value_t * Query,
    value_t * Subject,
    value_t * Dist,
    value_t * Cache,
    index_t num_entries,
    index_t num_features) {

    const index_t thid = blockDim.x*blockIdx.x+threadIdx.x;
    const index_t lane = num_features+1;
    const index_t base = thid*num_features;

    if (thid < num_entries) {

        // set penalty to the correct position in memory
        value_t * penalty = Cache + thid*2*lane;

        // init penalty matrix
        penalty[0] = 0;
        for (index_t index = 0; index < lane; index++)
            penalty[index+1] = INFINITY;

        for (index_t row = 1; row < lane; row++) {

            const value_t q_value = Query[row-1];
            const index_t target_row = row & 1;
            const index_t source_row = !target_row;

            if (row == 2)
                penalty[target_row*lane] = INFINITY;

            for (index_t col = 1; col < lane; col++) {

                const value_t diag = penalty[source_row*lane+col-1];
                const value_t abve = penalty[source_row*lane+col-0];
                const value_t left = penalty[target_row*lane+col-1];

                const value_t s_value = Subject[base+col-1];
                const value_t residue = q_value - s_value;

                penalty[target_row*lane+col] = residue * residue
                                             + min(diag, min(abve, left));
            }
        }

        const index_t last_row = num_features & 1;
        Dist[thid] = penalty[last_row*lane+num_features];
    }
}

template <
    typename index_t,
    typename value_t,
    index_t const_num_features> __global__
void DTW_static_kernel(
    value_t * Query,
    value_t * Subject,
    value_t * Dist,
    index_t num_entries,
    index_t num_features) {

    const index_t thid = blockDim.x*blockIdx.x+threadIdx.x;
    const index_t lane = num_features+1;
    const index_t base = thid*num_features;

    if (thid < num_entries) {

        // set penalty to the correct position in memory
        value_t penalty[2*(const_num_features+1)];

        // init penalty matrix
        penalty[0] = 0;
        for (index_t index = 0; index < lane; index++)
            penalty[index+1] = INFINITY;

        for (index_t row = 1; row < lane; row++) {

            const value_t q_value = Query[row-1];
            const index_t target_row = row & 1;
            const index_t source_row = !target_row;

            if (row == 2)
                penalty[target_row*lane] = INFINITY;

            for (index_t col = 1; col < lane; col++) {

                const value_t diag = penalty[source_row*lane+col-1];
                const value_t abve = penalty[source_row*lane+col-0];
                const value_t left = penalty[target_row*lane+col-1];

                const value_t s_value = Subject[base+col-1];
                const value_t residue = q_value - s_value;

                penalty[target_row*lane+col] = residue * residue
                                             + min(diag, min(abve, left));
            }
        }

        const index_t last_row = num_features & 1;
        Dist[thid] = penalty[last_row*lane+num_features];
    }
}

template <
    typename index_t,
    typename value_t> __global__
void DTW_interleaved_kernel(
    value_t * Query,
    value_t * Subject,
    value_t * Dist,
    value_t * Cache,
    index_t num_entries,
    index_t num_features) {

    const index_t thid = blockDim.x*blockIdx.x+threadIdx.x;
    const index_t lane = num_features+1;
    const index_t base = thid*num_features;

    auto iota = [&] (const index_t& index) { return index*num_entries+thid; };

    if (thid < num_entries) {

        // set penalty to the correct position in memory
        value_t * penalty = Cache;

        // init penalty matrix
        penalty[iota(0)] = 0;
        for (index_t index = 0; index < lane; index++)
            penalty[iota(index+1)] = INFINITY;

        for (index_t row = 1; row < lane; row++) {

            const value_t q_value = Query[row-1];
            const index_t target_row = row & 1;
            const index_t source_row = !target_row;

            if (row == 2)
                penalty[iota(target_row*lane)] = INFINITY;

            for (index_t col = 1; col < lane; col++) {

                const value_t diag = penalty[iota(source_row*lane+col-1)];
                const value_t abve = penalty[iota(source_row*lane+col-0)];
                const value_t left = penalty[iota(target_row*lane+col-1)];

                const value_t s_value = Subject[base+col-1];
                const value_t residue = q_value - s_value;

                penalty[iota(target_row*lane+col)] = residue * residue
                                                   + min(diag, min(abve, left));
            }
        }

        const index_t last_row = num_features & 1;
        Dist[thid] = penalty[iota(last_row*lane+num_features)];
    }
}

template <
    typename index_t,
    typename value_t> __global__
void DTW_shared_kernel(
    value_t * Query,
    value_t * Subject,
    value_t * Dist,
    index_t num_entries,
    index_t num_features) {

    const index_t thid = blockDim.x*blockIdx.x+threadIdx.x;
    const index_t lane = num_features+1;
    const index_t base = thid*num_features;

    extern __shared__ value_t Cache[];

    if (thid < num_entries) {

        // set penalty to the correct position in memory
        value_t * penalty = Cache + 2*threadIdx.x*lane;

        // init penalty matrix
        penalty[0] = 0;
        for (index_t index = 0; index < lane; index++)
            penalty[index+1] = INFINITY;

        for (index_t row = 1; row < lane; row++) {

            const value_t q_value = Query[row-1];
            const index_t target_row = row & 1;
            const index_t source_row = !target_row;

            if (row == 2)
                penalty[target_row*lane] = INFINITY;

            for (index_t col = 1; col < lane; col++) {

                const value_t diag = penalty[source_row*lane+col-1];
                const value_t abve = penalty[source_row*lane+col-0];
                const value_t left = penalty[target_row*lane+col-1];

                const value_t s_value = Subject[base+col-1];
                const value_t residue = q_value - s_value;

                penalty[target_row*lane+col] = residue * residue
                                             + min(diag, min(abve, left));
            }
        }

        const index_t last_row = num_features & 1;
        Dist[thid] = penalty[last_row*lane+num_features];
    }
}

template <
    typename index_t,
    typename value_t> __global__
void DTW_shared_opt_kernel(
    value_t * Query,
    value_t * Subject,
    value_t * Dist,
    index_t num_entries,
    index_t num_features) {

    const index_t thid = blockDim.x*blockIdx.x+threadIdx.x;
    const index_t lane = num_features+1;
    const index_t base = thid*num_features;

    extern __shared__ value_t Cache[];

    if (thid < num_entries) {

        // set penalty to the correct position in memory
        value_t * penalty = Cache + 2*threadIdx.x*lane;

        // init penalty matrix
        penalty[0] = 0;
        for (index_t index = 0; index < lane; index++)
            penalty[index+1] = INFINITY;

        for (index_t row = 1; row < lane; row++) {

            const value_t q_value = cQuery[row-1];
            const index_t target_row = row & 1;
            const index_t source_row = !target_row;

            if (row == 2)
                penalty[target_row*lane] = INFINITY;

            for (index_t col = 1; col < lane; col++) {

                const value_t diag = penalty[source_row*lane+col-1];
                const value_t abve = penalty[source_row*lane+col-0];
                const value_t left = penalty[target_row*lane+col-1];

                const value_t s_value = Subject[base+col-1];
                const value_t residue = q_value - s_value;

                penalty[target_row*lane+col] = residue * residue
                                             + min(diag, min(abve, left));
            }
        }

        const index_t last_row = num_features & 1;
        Dist[thid] = penalty[last_row*lane+num_features];
    }
}

template <
    typename index_t,
    typename value_t> __global__
void DTW_wavefront_kernel(
    value_t * Query,
    value_t * Subject,
    value_t * Dist,
    index_t num_entries,
    index_t num_features) {

    const index_t blid = blockIdx.x;
    const index_t thid = threadIdx.x;
    const index_t lane = num_features+1;
    const index_t base = blid*num_features;

    //if (thid == 0)
    //    printf("%lu %lu\n", blid, base);

    extern __shared__ value_t Cache[];
    value_t * penalty = Cache;

    // initialize penalty matrix
    for (index_t l = thid; l < lane; l += blockDim.x) {
        penalty[0*lane+l] = INFINITY;
        penalty[1*lane+l] = INFINITY;
        penalty[2*lane+l] = INFINITY;
    }

    penalty[0*lane+0] = 0;

    __syncthreads();

    // relax diagonals
    for (index_t k = 2; k < 2*lane-1; k++) {
        const index_t target_row = k % 3;
        const index_t before_row = target_row == 2 ? 0 : target_row+1;
        const index_t source_row = before_row == 2 ? 0 : before_row+1;

        for (index_t l = thid; l < lane; l += blockDim.x) {
            const index_t i = k-l;
            const index_t j = l;

            const bool outside = k <= l || j == 0 || i >= lane;

            const value_t residue = outside ? INFINITY :
                                    Query[i-1]-Subject[base+j-1];

            penalty[target_row*lane+l] = outside ? INFINITY : residue*residue +
                                         min(penalty[before_row*lane+l-1],
                                         min(penalty[source_row*lane+l+0],
                                             penalty[source_row*lane+l-1]));
        }

        __syncthreads();
    }

    const index_t last_diag = (2*num_features) % 3;
    Dist[blid] = penalty[last_diag*lane+num_features];
}

template <
    typename index_t,
    typename value_t> __global__
void DTW_wavefront_const_kernel(
    value_t * Subject,
    value_t * Dist,
    index_t num_entries,
    index_t num_features) {

    const index_t blid = blockIdx.x;
    const index_t thid = threadIdx.x;
    const index_t lane = num_features+1;
    const index_t base = blid*num_features;

    //if (thid == 0)
    //    printf("%lu %lu\n", blid, base);

    extern __shared__ value_t Cache[];
    value_t * penalty = Cache;

    // initialize penalty matrix
    for (index_t l = thid; l < lane; l += blockDim.x) {
        penalty[0*lane+l] = INFINITY;
        penalty[1*lane+l] = INFINITY;
        penalty[2*lane+l] = INFINITY;
    }

    penalty[0*lane+0] = 0;

    __syncthreads();

    // relax diagonals
    for (index_t k = 2; k < 2*lane-1; k++) {
        const index_t target_row = k % 3;
        const index_t before_row = target_row == 2 ? 0 : target_row+1;
        const index_t source_row = before_row == 2 ? 0 : before_row+1;

        for (index_t l = thid; l < lane; l += blockDim.x) {
            const index_t i = k-l;
            const index_t j = l;

            const bool outside = k <= l || j == 0 || i >= lane;

            const value_t residue = outside ? INFINITY :
                                    cQuery[i-1]-Subject[base+j-1];

            penalty[target_row*lane+l] = outside ? INFINITY : residue*residue +
                                         min(penalty[before_row*lane+l-1],
                                         min(penalty[source_row*lane+l+0],
                                             penalty[source_row*lane+l-1]));
        }

        __syncthreads();
    }

    const index_t last_diag = (2*num_features) % 3;
    Dist[blid] = penalty[last_diag*lane+num_features];
}


template <
    typename index_t,
    typename value_t> __global__
void DTW_wavefront_tex_kernel(
    value_t * Query,
    value_t * Dist,
    index_t num_entries,
    index_t num_features) {

    const index_t blid = blockIdx.x;
    const index_t thid = threadIdx.x;
    const index_t lane = num_features+1;
    const index_t base = blid*num_features;

    //if (thid == 0)
    //    printf("%lu %lu\n", blid, base);

    extern __shared__ value_t Cache[];
    value_t * penalty = Cache;

    // initialize penalty matrix
    for (index_t l = thid; l < lane; l += blockDim.x) {
        penalty[0*lane+l] = INFINITY;
        penalty[1*lane+l] = INFINITY;
        penalty[2*lane+l] = INFINITY;
    }

    penalty[0*lane+0] = 0;

    __syncthreads();

    // relax diagonals
    for (index_t k = 2; k < 2*lane-1; k++) {
        const index_t target_row = k % 3;
        const index_t before_row = target_row == 2 ? 0 : target_row+1;
        const index_t source_row = before_row == 2 ? 0 : before_row+1;

        for (index_t l = thid; l < lane; l += blockDim.x) {
            const index_t i = k-l;
            const index_t j = l;

            const bool outside = k <= l || j == 0 || i >= lane;

            const value_t residue = outside ? INFINITY :
                                    Query[i-1]-tex1Dfetch(tSubject, base+j-1);

            penalty[target_row*lane+l] = outside ? INFINITY : residue*residue +
                                         min(penalty[before_row*lane+l-1],
                                         min(penalty[source_row*lane+l+0],
                                             penalty[source_row*lane+l-1]));
        }

        __syncthreads();
    }

    const index_t last_diag = (2*num_features) % 3;
    Dist[blid] = penalty[last_diag*lane+num_features];
}

template <
    typename index_t,
    typename value_t>
value_t dtw(
    value_t * query,
    value_t * subject,
    index_t num_features) {

    const index_t lane = num_features+1;
    value_t * penalty = new value_t[2*lane];

    for (index_t index = 0; index < lane; index++)
        penalty[index+1] = INFINITY;
    penalty[0] = 0;

    for (index_t row = 1; row < lane; row++) {

        const value_t q_value = query[row-1];
        const index_t target_row = row & 1;
        const index_t source_row = !target_row;

        if (row == 2)
            penalty[target_row*lane] = INFINITY;

        for (index_t col = 1; col < lane; col++) {

            const value_t diag = penalty[source_row*lane+col-1];
            const value_t abve = penalty[source_row*lane+col+0];
            const value_t left = penalty[target_row*lane+col-1];

            const value_t residue = q_value-subject[col-1];

            penalty[target_row*lane+col] = residue*residue +
                                            min(diag, min(abve, left));
        }
    }

    const index_t last_row = num_features & 1;
    const value_t result = penalty[last_row*lane+num_features];
    delete [] penalty;

    return result;
}

#include <omp.h>
template <
    typename index_t,
    typename value_t>
void host_dtw(
    value_t * query,
    value_t * subject,
    value_t * dist,
    index_t num_entries,
    index_t num_features) {

    # pragma omp parallel for
    for (index_t entry = 0; entry < num_entries; entry++)
        dist[entry] = dtw(query, subject+entry*num_features, num_features);
}

int main () {

    constexpr index_t num_features = 128;
    constexpr index_t num_entries = 1UL << 20;

    // small letters for hosts, capital letters for device
    value_t * data = nullptr, * dist = nullptr,
            * Data = nullptr, * Dist = nullptr, * Cache = nullptr;
    label_t * labels = nullptr;

    // malloc memory
    cudaMallocHost(&data, sizeof(value_t)*num_entries*num_features);      CUERR
    cudaMallocHost(&dist, sizeof(value_t)*num_entries);                   CUERR
    cudaMallocHost(&labels, sizeof(label_t)*num_entries);                 CUERR


    cudaMalloc(&Data, sizeof(value_t)*num_entries*num_features);          CUERR
    cudaMalloc(&Dist, sizeof(value_t)*num_entries);                       CUERR
    cudaMalloc(&Cache, sizeof(value_t)*num_entries*2*(num_features+1));   CUERR

    // create CBF data set on host
    TIMERSTART(generate_data)
    generate_cbf(data, labels, num_entries, num_features);
    TIMERSTOP(generate_data)

    // transfer data to device
    TIMERSTART(transfer_data_H2D)
    cudaMemcpy(Data, data, sizeof(value_t)*num_entries*num_features,
               cudaMemcpyHostToDevice);                                   CUERR
    cudaMemset(Dist, 0, sizeof(value_t)*num_entries);                     CUERR
    cudaMemset(Cache, 0, sizeof(value_t)*num_entries*2*(num_features+1)); CUERR
    TIMERSTOP(transfer_data_H2D)

    index_t threads = 32;

    cudaMemset(Dist, 0, sizeof(value_t)*num_entries);                     CUERR
    TIMERSTART(DTW_naive_kernel)
    DTW_naive_kernel<<<SDIV(num_entries, threads), threads>>>
                    (Data, Data, Dist, Cache, num_entries, num_features); CUERR
    TIMERSTOP(DTW_naive_kernel)


    cudaMemset(Dist, 0, sizeof(value_t)*num_entries);                     CUERR
    TIMERSTART(DTW_static_kernel)
    DTW_static_kernel<index_t, value_t, num_features>
                     <<<SDIV(num_entries, threads), threads>>>
                     (Data, Data, Dist, num_entries, num_features);       CUERR
    TIMERSTOP(DTW_static_kernel)

    cudaMemset(Dist, 0, sizeof(value_t)*num_entries);                     CUERR
    TIMERSTART(DTW_interleaved_kernel)
    DTW_interleaved_kernel<<<SDIV(num_entries, threads), threads>>>
                    (Data, Data, Dist, Cache, num_entries, num_features); CUERR
    TIMERSTOP(DTW_interleaved_kernel)

    cudaMemset(Dist, 0, sizeof(value_t)*num_entries);                     CUERR
    TIMERSTART(DTW_shared_kernel)
    index_t shared_memory = 2*(num_features+1)*threads*sizeof(value_t);
    DTW_shared_kernel<<<SDIV(num_entries, threads), threads, shared_memory>>>
                     (Data, Data, Dist, num_entries, num_features);       CUERR
    TIMERSTOP(DTW_shared_kernel)

    cudaMemset(Dist, 0, sizeof(value_t)*num_entries);                     CUERR
    TIMERSTART(DTW_shared_opt_kernel)
    shared_memory = 2*(num_features+1)*threads*sizeof(value_t);
    DTW_shared_opt_kernel<<<SDIV(num_entries, threads), threads, shared_memory>>>
                     (Data, Data, Dist, num_entries, num_features);       CUERR
    TIMERSTOP(DTW_shared_opt_kernel)

    cudaMemset(Dist, 0, sizeof(value_t)*num_entries);                     CUERR
    TIMERSTART(DTW_wavefront_kernel)
    shared_memory = 3*(num_features+1)*sizeof(value_t);
    DTW_wavefront_kernel<<<num_entries, threads, shared_memory>>>
                         (Data, Data, Dist, num_entries, num_features);   CUERR
    TIMERSTOP(DTW_wavefront_kernel)


    cudaMemcpyToSymbol(cQuery, Data, sizeof(value_t)*num_features);       CUERR

    cudaMemset(Dist, 0, sizeof(value_t)*num_entries);                     CUERR
    TIMERSTART(DTW_wavefront_const_kernel)
    shared_memory = 3*(num_features+1)*sizeof(value_t);
    DTW_wavefront_const_kernel<<<num_entries, threads, shared_memory>>>
                         (Data, Dist, num_entries, num_features);         CUERR
    TIMERSTOP(DTW_wavefront_const_kernel)

    cudaBindTexture(0, tSubject, Data,
                    sizeof(value_t)*num_entries*num_features);            CUERR

    cudaMemset(Dist, 0, sizeof(value_t)*num_entries);                     CUERR
    TIMERSTART(DTW_wavefront_tex_kernel)
    shared_memory = 3*(num_features+1)*sizeof(value_t);
    DTW_wavefront_tex_kernel<<<num_entries, threads, shared_memory>>>
                         (Data, Dist, num_entries, num_features);               CUERR
    TIMERSTOP(DTW_wavefront_tex_kernel)

    TIMERSTART(transfer_dist_D2H)
    cudaMemcpy(dist, Dist, sizeof(value_t)*num_entries,
               cudaMemcpyDeviceToHost);                                   CUERR
    TIMERSTOP(transfer_dist_D2H)

    TIMERSTART(DTW_openmp)
    host_dtw(data, data, dist, num_entries, num_features);
    TIMERSTOP(DTW_openmp)


    for (index_t index = 0; index < 10; index++)
        std::cout << index_t(labels[index]) << " " << dist[index] << std::endl;


    // get rid of the memory
    cudaFreeHost(labels);
    cudaFreeHost(data);
    cudaFreeHost(dist);
    cudaFree(Cache);
    cudaFree(Data);
    cudaFree(Dist);

}
