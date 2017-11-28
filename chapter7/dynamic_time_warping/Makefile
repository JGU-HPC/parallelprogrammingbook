NVCC= nvcc
NVCCFLAGS= -O2 -std=c++14 -arch=sm_61
CXXFLAGS= -Xcompiler="-fopenmp -march=native"

all: dtw_host  dtw_device

dtw_host: dtw_host.cu
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) dtw_host.cu -o dtw_host

dtw_device: dtw_device.cu
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) dtw_device.cu -o dtw_device

clean:
	rm -f dtw_host
	rm -f dtw_device
