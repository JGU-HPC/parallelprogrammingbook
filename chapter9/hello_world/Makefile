MPICXX= mpic++
MPICXXFLAGS= -O2 -std=c++11

all: hello_world

hello_world: hello_world.cpp
	$(MPICXX) $(MPICXXFLAGS) hello_world.cpp -o hello_world

clean:
	rm -rf hello_world
