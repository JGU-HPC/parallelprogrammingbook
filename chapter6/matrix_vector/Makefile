CXX= g++
CXXFLAGS= -std=c++14 -O2 -fopenmp

all: matrix_vector

matrix_vector: matrix_vector.cpp
	$(CXX) matrix_vector.cpp $(CXXFLAGS) -o matrix_vector

clean:
	rm -rf matrix_vector
