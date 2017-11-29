#include "mpi.h"

int main (int argc, char *argv[]){
	// Initialize MPI
	MPI::Init(argc,argv);

	// Get the number of processes
	int numP=MPI::COMM_WORLD.Get_size();

	// Get the ID of the process
	int myId=MPI::COMM_WORLD.Get_rank();

	// Every process prints Hello
	std::cout << "Process " << myId << " of " << numP << ": Hello, world!" << std::endl;

	// Terminate MPI
	MPI::Finalize();
	return 0;
}
