CXX= g++-6
CXXFLAGS= -std=c++14 -O2 -fopenmp

all: softmax

softmax: softmax.cpp
	$(CXX) softmax.cpp $(CXXFLAGS) -o softmax

clean:
	rm -rf softmax
