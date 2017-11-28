NVCC=nvcc

all: hello_world

hello_world: hello_world.cu
	$(NVCC) -O2 -std=c++11 hello_world.cu -o hello_world

clean:
	rm -f hello_world
