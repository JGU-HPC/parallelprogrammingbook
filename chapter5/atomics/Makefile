CXX= g++
CXXFLAGS= -std=c++14 -O2 -pthread -latomic -march=native

all: query_atomics atomic_counting atomic_max arbitrary_atomics universal_atomics

query_atomics: query_atomics.cpp
	$(CXX) query_atomics.cpp $(CXXFLAGS) -o query_atomics

atomic_counting: atomic_counting.cpp
	$(CXX) atomic_counting.cpp $(CXXFLAGS) -o atomic_counting

atomic_max: atomic_max.cpp
	$(CXX) atomic_max.cpp $(CXXFLAGS) -o atomic_max

arbitrary_atomics: arbitrary_atomics.cpp
	$(CXX) arbitrary_atomics.cpp $(CXXFLAGS) -o arbitrary_atomics

universal_atomics: universal_atomics.cpp
	$(CXX) universal_atomics.cpp $(CXXFLAGS) -o universal_atomics

clean:
	rm -rf arbitrary_atomics
	rm -rf atomic_counting
	rm -rf atomic_max
	rm -rf arbitrary_atomics
	rm -rf universal_atomics
