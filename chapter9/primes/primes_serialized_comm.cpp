#include <stdlib.h>

#include "mpi.h"

int main (int argc, char *argv[]){
	// Initialize MPI
	MPI::Init(argc,argv);

	// Get the number of processes
	int numP=MPI::COMM_WORLD.Get_size();

	// Get the ID of the process
	int myId=MPI::COMM_WORLD.Get_rank();

	if(argc < 2){
		// Only the first process prints the output message
		if(!myId){
			std::cout << "ERROR: The syntax of the program is "
                                  << argv[0] << " n" << std::endl;
		}
		MPI::COMM_WORLD.Abort(1);
	}

	int n;

	if(!myId){
		n = atoi(argv[1]);
	}

	// Barrier to synchronize the processes before measuring time
	MPI::COMM_WORLD.Barrier();

	// Measure the current time
	double start = MPI::Wtime();

	// Send the value of n to all processes
	MPI::COMM_WORLD.Bcast(&n, 1, MPI::INT, 0);

	if(n < 1){
		// Only the first process prints the output message
		if(!myId){
			std::cout << "ERROR: The parameter 'n' must be higher than 0" << std::endl;
		}
		MPI::COMM_WORLD.Abort(1);
	}

	// Perform the computation of the number of primes between 1 and n in parallel
	int myCount = 0;
	int total;
	bool prime;

	// Each process analyzes only part of the numbers below n
	// Data to perform a block distribution
	int blockSize = (n-1)/numP;
	int myBlockSize = blockSize;
	int myStart = 2+myId*blockSize;

	// For the cases that n is not multiple of numP
	if(myId < (n-1)%numP){
		myBlockSize++;
		myStart += myId;
	} else {
		myStart += (n-1)%numP;
	}

	int myEnd = myStart+myBlockSize;

	for(int i=myStart; i<myEnd; i++){
		prime = true;
		for(int j=2; j<i; j++){
			if((i%j) == 0){
				prime = false;
				break;
			}
		}
		myCount += prime;
	}

	// Reduce the partial counts into 'total' in the process 0
    //MPI::COMM_WORLD.Reduce(&myCount, &total, 1, MPI::INT, MPI::SUM, 0);

MPI::Status st;
  if(!myId){
    total = myCount;
    for(int i=1; i<numP; i++){
      MPI::COMM_WORLD.Recv(&myCount, 1, MPI::INT, MPI::ANY_SOURCE, 0, st);
      total += myCount;
    }
}
  else {
    MPI::COMM_WORLD.Send(&myCount, 1, MPI::INT, 0, 0);
  }  

	// Measure the current time
    // Barrier is not necessary because
	double end = MPI::Wtime();

    if(!myId){
    	std::cout << total << " primes between 1 and " << n << std::endl;
    	std::cout << "Time with " << numP << " processes: " << end-start << " seconds" << std::endl;
    }

	// Terminate MPI
	MPI::Finalize();
	return 0;
}
