MPICXX= mpic++
MPICXXFLAGS= -O2 -std=c++11

all: ping_pong_ring ping_pong_ring_nonblock

ping_pong_ring: ping_pong_ring.cpp
	$(MPICXX) $(MPICXXFLAGS) ping_pong_ring.cpp -o ping_pong_ring

ping_pong_ring_nonblock: ping_pong_ring_nonblock.cpp
	$(MPICXX) $(MPICXXFLAGS) ping_pong_ring_nonblock.cpp -o ping_pong_ring_nonblock

clean:
	rm -rf ping_pong_ring
	rm -rf ping_pong_nonblock
