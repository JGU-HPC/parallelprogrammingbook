CXX= g++
CXXFLAGS= -std=c++14 -O2 -pthread

all: main_basic main_basic_tree

main_basic: main_basic.cpp
	$(CXX) main_basic.cpp $(CXXFLAGS) -o main_basic

main_basic_tree: main_basic_tree.cpp
	$(CXX) main_basic_tree.cpp $(CXXFLAGS) -o main_basic_tree

clean:
	rm -rf main_basic
	rm -rf main_basic_tree
