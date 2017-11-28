CXX= g++-6
CXXFLAGS= -std=c++14 -O2 -fopenmp -mavx -march=native

all: custom_reduction avx_reduction string_reduction

custom_reduction: custom_reduction.cpp
	$(CXX) custom_reduction.cpp $(CXXFLAGS) -o custom_reduction

avx_reduction: avx_reduction.cpp
	$(CXX) avx_reduction.cpp $(CXXFLAGS) -o avx_reduction

string_reduction: string_reduction.cpp
	$(CXX) string_reduction.cpp $(CXXFLAGS) -o string_reduction

clean:
	rm -rf custom_reduction
	rm -rf avx_reduction
	rm -rf string_reduction
