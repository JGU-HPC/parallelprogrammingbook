#include <cstdint>
#include <iostream>

__global__ void iota_kernel(float * input, uint64_t size) {

    uint64_t thid = blockIdx.x*blockDim.x+threadIdx.x;
    for (uint64_t i = thid; i < size; i += gridDim.x*blockDim.x)
        input[i] = i;
}

int main () {

    uint64_t size = 1UL << 20;
    float * input = nullptr;
    cudaMallocHost(&input, sizeof(float)*size);
    iota_kernel<<<1024, 1024>>>(input, size);

    cudaDeviceSynchronize();

    for (uint64_t i = 0; i < 20; i++)
        std::cout << input[i] << std::endl;
}
