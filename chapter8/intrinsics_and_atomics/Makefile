NVCC= nvcc
NVCCFLAGS= -O2 -std=c++14 -arch=sm_61

all: znorm atomics cas

znorm: znorm.cu
	$(NVCC) $(NVCCFLAGS) znorm.cu -o znorm

atomics: atomics.cu
	$(NVCC) $(NVCCFLAGS) atomics.cu -o atomics

cas: cas.cu
	$(NVCC) $(NVCCFLAGS) cas.cu -o cas

clean:
	rm -f znorm
	rm -f atomics
	rm -f cas
