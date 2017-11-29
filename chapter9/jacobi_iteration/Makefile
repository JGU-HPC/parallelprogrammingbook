CXX = g++
CXXFLAGS = -O2 -std=c++11
MPICXX= mpic++
MPICXXFLAGS= $(CXXFLAGS)

all: jacobi_seq jacobi_1D_block_simple jacobi_1D_block jacobi_1D_nonblock

jacobi_seq: jacobi_seq.cpp
	$(CXX) $(CXXFLAGS) jacobi_seq.cpp -o jacobi_seq

jacobi_1D_block_simple: jacobi_1D_block_simple.cpp
	$(MPICXX) $(MPICXXFLAGS) jacobi_1D_block_simple.cpp -o jacobi_1D_block_simple

jacobi_1D_block: jacobi_1D_block.cpp
	$(MPICXX) $(MPICXXFLAGS) jacobi_1D_block.cpp -o jacobi_1D_block

jacobi_1D_nonblock: jacobi_1D_nonblock.cpp
	$(MPICXX) $(MPICXXFLAGS) jacobi_1D_nonblock.cpp -o jacobi_1D_nonblock

clean:
	rm -rf jacobi_seq
	rm -rf jacobi_1D_block_simple
	rm -rf jacobi_1D_block
	rm -rf jacobi_1D_nonblock
