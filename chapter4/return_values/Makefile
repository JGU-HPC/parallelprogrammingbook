CXX= g++
CXXFLAGS= -std=c++14 -O2 -pthread

all: traditional promise_future packaged_task async

traditional: traditional.cpp
	$(CXX) traditional.cpp $(CXXFLAGS) -o traditional

promise_future: promise_future.cpp
	$(CXX) promise_future.cpp $(CXXFLAGS) -o promise_future

packaged_task: packaged_task.cpp
	$(CXX) packaged_task.cpp $(CXXFLAGS) -o packaged_task

async: async.cpp
	$(CXX) async.cpp $(CXXFLAGS) -o async

clean:
	rm -rf traditional
	rm -rf promise_future
	rm -rf packaged_task
	rm -rf async
