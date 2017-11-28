NVCC= nvcc
NVCCFLAGS= -O2 -std=c++14 -arch=sm_61

all: single_gpu multi_gpu streamed_gpu multi_streamed_gpu

single_gpu: single_gpu.cu
	$(NVCC) $(NVCCFLAGS) single_gpu.cu -o single_gpu

multi_gpu: multi_gpu.cu
	$(NVCC) $(NVCCFLAGS) multi_gpu.cu -o multi_gpu

streamed_gpu: streamed_gpu.cu
	$(NVCC) $(NVCCFLAGS) streamed_gpu.cu -o streamed_gpu

multi_streamed_gpu: multi_streamed_gpu.cu
	$(NVCC) $(NVCCFLAGS) multi_streamed_gpu.cu -o multi_streamed_gpu

clean:
	rm -f single_gpu
	rm -f multi_gpu
	rm -f streamed_gpu
	rm -f multi_streamed_gpu
