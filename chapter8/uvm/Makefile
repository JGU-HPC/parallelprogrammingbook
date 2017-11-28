NVCC= nvcc
NVCCFLAGS= -O2 -std=c++11 -arch=sm_61

all: uvm_minimal_example

uvm_minimal_example: uvm_minimal_example.cu
	$(NVCC) $(NVCCFLAGS) uvm_minimal_example.cu -o uvm_minimal_example

clean:
	rm -f uvm_minimal_example
