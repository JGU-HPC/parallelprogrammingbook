#include "../include/hpc_helpers.hpp"

__device__ __forceinline__
int atomicUpdateResultBoundedByTwo(
    int* address,
    int value) {

    // get the source value stored at address
    int source = *address, expected;

    do {
        // we expect source
        expected = source;

        // compute our custom binary operation
        int target = expected+value+expected*value;

        // check the constraint
        if (target < 0 || target >= 10)
            return source;

        // try to swap the values
        source = atomicCAS(address, expected, target);

    // (expected == source) on success
    } while (expected != source);

    return source;
}

__global__
void apply_kernel(int * source_address, int value) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        atomicUpdateResultBoundedByTwo(source_address, value);

}


int main () {
    int * data = nullptr;
    cudaMallocHost(&data, sizeof(int));                               CUERR

    *data = 0;
    apply_kernel<<<1, 1>>> (data, 10);

    cudaDeviceSynchronize();

    std::cout << * data << std::endl;

    cudaFree(data);
}
