MPICXX= mpic++
MPICXXFLAGS= -O2 -std=c++11

all: primes_serialized_comm primes

primes_serialized_comm: primes_serialized_comm.cpp
	$(MPICXX) $(MPICXXFLAGS) primes_serialized_comm.cpp -o primes_serialized_comm

primes: primes.cpp
	$(MPICXX) $(MPICXXFLAGS) primes.cpp -o primes

clean:
	rm -rf primes_serialized_comm
	rm -rf primes
