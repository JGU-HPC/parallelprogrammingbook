UPCXX= /opt/upcxx/bin/upc++                   # install UPC++ and specify binary
UPCXXFLAGS= -O2 -std=c++11                    # as usual
GASNETRUN= /opt/gasnet/bin/gasnetrun_mpi -n 4 # install gasnet and choose backend

all: hello_world

hello_world: hello_world.cxx
	$(UPCXX) $(UPCXXFLAGS) hello_world.cxx -o hello_world

hello_world_run: hello_world
	$(GASNETRUN) hello_world

clean:
	rm -rf hello_world
