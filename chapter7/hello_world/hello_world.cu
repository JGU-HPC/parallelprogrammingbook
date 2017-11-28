#include <stdio.h>                      // printf

__global__ void hello_kernel() {

    // calculate global thread identifier, note blockIdx.x=0 here
    const auto thid = blockDim.x*blockIdx.x + threadIdx.x;

    // print a greeting message
    printf("Hello from thread %d!\n", thid);
}

// compile with: nvcc hello_world.cu -std=c++11 -O3
// output:
// Hello from thread 0!
// Hello from thread 1!
// Hello from thread 2!
// Hello from thread 3!

int main (int argc, char * argv[]) {

    // set the ID of the CUDA device
    cudaSetDevice(0);

    // invoke kernel using 4 threads executed in 1 thread block
    hello_kernel<<<1, 4>>>();

    // synchronize the GPU preventing premature termination
    cudaDeviceSynchronize();
}
