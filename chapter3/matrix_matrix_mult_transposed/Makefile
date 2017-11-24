CXX=g++
CXXFLAGS=-O2 -std=c++14 -Wall

all: matrix_mult_seq matrix_mult_omp

matrix_mult_seq: matrix_mult.cpp
	$(CXX) $(CXXFLAGS) matrix_mult.cpp -o matrix_mult_seq

matrix_mult_omp: matrix_mult.cpp
	$(CXX) $(CXXFLAGS) matrix_mult.cpp -fopenmp -o matrix_mult_omp

clean:
	rm -f matrix_mult_seq
	rm -f matrix_mult_omp
