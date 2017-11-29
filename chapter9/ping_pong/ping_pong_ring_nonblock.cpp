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
			std::cout << "ERROR: The syntax of the program is " << argv[0] 
                                  << " num_ping_pong" << std::endl;
		}
		MPI::COMM_WORLD.Abort(1);
	}

	int num_ping_pong = atoi(argv[1]);
	int ping_pong_count = 0;
	int next_id = myId+1, prev_id=myId-1;

	if(next_id >= numP){
		next_id = 0;
	}
	if(prev_id < 0){
		prev_id = numP-1;
	}	

	MPI::Request rq_send, rq_recv;

	while(ping_pong_count < num_ping_pong){
		// First receive the ping and then send the pong
		ping_pong_count++;
		rq_send = MPI::COMM_WORLD.Isend(&ping_pong_count, 1, MPI::INT, next_id, 0);
		std::cout << "Process " << myId << " sends PING number " << ping_pong_count
					<< " to process " << next_id << std::endl;
		rq_recv = MPI::COMM_WORLD.Irecv(&ping_pong_count, 1, MPI::INT, prev_id, 0);
		std::cout << "Process " << myId << " receives PING number " << ping_pong_count
					<< " from process " << prev_id << std::endl;

		rq_recv.Wait();

		rq_send = MPI::COMM_WORLD.Isend(&ping_pong_count, 1, MPI::INT, prev_id, 0);
		std::cout << "Process " << myId << " sends PONG number " << ping_pong_count
					<< " to process " << prev_id << std::endl;
		rq_recv = MPI::COMM_WORLD.Irecv(&ping_pong_count, 1, MPI::INT, next_id, 0);
		std::cout << "Process " << myId << " receives PONG number " << ping_pong_count
					<< " from process " << next_id << std::endl;

		rq_recv.Wait();
	}

	// Terminate MPI
	MPI::Finalize();

	return 0;
}
