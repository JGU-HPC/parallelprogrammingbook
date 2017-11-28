#include "../include/cbf_generator.hpp"
#include "../include/hpc_helpers.hpp"
#include "../include/binary_IO.hpp"

typedef uint64_t index_t;
typedef uint8_t label_t;
typedef float value_t;

template <
    typename index_t,
    typename value_t>
value_t plain_dtw(
    value_t * query,
    value_t * subject,
    index_t num_features) {

    // for convenient indexing
    const index_t lane = num_features+1;

    // allocate the matrix of M
    value_t * penalty = new value_t[lane*lane];

    // initialize the matrix M
    for (index_t index = 1; index < lane-1; index++) {
        penalty[index] = INFINITY;
        penalty[index*lane] = INFINITY;
    }
    penalty[0] = 0;

    // traverse graph in row-major order
    for (index_t row = 1; row < lane; row++) {

        const value_t q_value = query[row-1];

        for (index_t col = 1; col < lane; col++) {

            // determine contribution from incoming edges
            const value_t diag = penalty[(row-1)*lane+col-1];
            const value_t abve = penalty[(row-1)*lane+col+0];
            const value_t left = penalty[(row+0)*lane+col-1];

            // compute residue between query and subject
            const value_t residue = q_value-subject[col-1];

            // relax node
            penalty[row*lane+col] = residue*residue +
                                    min(diag, 
                                    min(abve, left));
        }
    }

    // report the lower right cell and free memory
    const value_t result = penalty[lane*lane-1];
    delete [] penalty;

    return result;
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
    value_t * data = nullptr, * dist = nullptr;
    label_t * labels = nullptr;

    // malloc memory
    cudaMallocHost(&data, sizeof(value_t)*num_entries*num_features);      CUERR
    cudaMallocHost(&dist, sizeof(value_t)*num_entries);                   CUERR
    cudaMallocHost(&labels, sizeof(label_t)*num_entries);                 CUERR

    // create CBF data set on host
    TIMERSTART(generate_data)
    generate_cbf(data, labels, num_entries, num_features);
    TIMERSTOP(generate_data)

  
    TIMERSTART(DTW_openmp)
    host_dtw(data, data, dist, num_entries, num_features);
    TIMERSTOP(DTW_openmp)


    for (index_t index = 0; index < 10; index++)
        std::cout << index_t(labels[index]) << " " << dist[index] << std::endl;


    // get rid of the memory
    cudaFreeHost(labels);
    cudaFreeHost(data);
    cudaFreeHost(dist);
}
