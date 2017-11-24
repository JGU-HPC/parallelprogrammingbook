CXX= g++
CXXFLAGS= -std=c++14 -O2 -pthread

all: hello_world

hello_world: hello_world.cpp
	$(CXX) hello_world.cpp $(CXXFLAGS) -o hello_world

clean:
	rm -rf hello_world
