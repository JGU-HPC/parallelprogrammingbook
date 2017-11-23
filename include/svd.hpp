#ifndef HPC_BOOK_SVD_HPP
#define HPC_BOOK_SVD_HPP

#include <cusolverDn.h>
#include "hpc_helpers.hpp"

bool svd_device(
    float * M,
    float * U,
    float * S,
    float * V,
    int height,
    int width,
    bool verbose=false) {

    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    int temp_storage_bytes = 0;
    if (cusolverDnSgesvd_bufferSize(handle, width, height, &temp_storage_bytes))
        return 1;

    float * temp_storage = nullptr;
    if (cudaMalloc(&temp_storage, sizeof(float)*temp_storage_bytes))
        return 1;

    if (verbose)
        std::cout << "CUSOLVER: allocated " << temp_storage_bytes
                  << " bytes of temporary storage." << std::endl;

    int * devInfo;
    if(cudaMalloc(&devInfo, sizeof(int)))
        return 1;

    if (cusolverDnSgesvd(handle, 'A', 'A', height, width,
                         M, height, S, U, height, V, width,
                         temp_storage, temp_storage_bytes, nullptr, devInfo ))
        return 1;

    if (verbose)
        std::cout << "CUSOLVER: computed SVD." << std::endl;

    if (cusolverDnDestroy(handle))
        return 1;
    if (cudaFree(temp_storage))
        return 1;

    if (verbose)
        std::cout << "CUSOLVER: freed " << temp_storage_bytes
                  << " bytes of temporary storage." << std::endl;

    return 0;
}

#endif
