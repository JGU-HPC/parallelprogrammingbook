CXX= g++
CXXFLAGS= -std=c++14 -O2 -fopenmp

all: vector_add vector_add_scoped

vector_add: vector_add.cpp
	$(CXX) vector_add.cpp $(CXXFLAGS) -o vector_add

vector_add_scoped: vector_add_scoped.cpp
	$(CXX) vector_add_scoped.cpp $(CXXFLAGS) -o vector_add_scoped

clean:
	rm -rf vector_add
	rm -rf vector_add_scoped
