#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include "mpi.h"

void readInput(int rows, int cols, float *data){

	// Open the file pointer
	/*FILE* fp = fopen(file.c_str(), "rb");

	// Check if the file exists
	if(fp == NULL){
		std::cout << "ERROR: File " << file << " could not be opened" << std::endl;
		MPI::COMM_WORLD.Abort(1);
	}

	for(int i=0; i<rows*cols; i++){
		if(!fscanf(fp, "%f", &data[i])){
			std::cout << "ERROR: Not enough values in file " << file << std::endl;
			MPI::COMM_WORLD.Abort(1);
		}
	}*/

    // checkerboard
    for(int i=0; i<rows; i++)
        for(int j=0; j<cols; j++)
            data[i*cols+j] = (i+j) % 2;
}

void printOutput(int rows, int cols, float *data){

	FILE *fp = fopen("outSUMMA.txt", "wb");
	// Check if the file was opened
	if(fp == NULL){
		std::cout << "ERROR: Output file outSUMMA.txt could not be opened" << std::endl;
		MPI::COMM_WORLD.Abort(1);
	}

    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++)
        	fprintf(fp, "%lf ", data[i*cols+j]);
        fprintf(fp, "\n");
    }

    fclose(fp);
}

int main (int argc, char *argv[]){
	// Initialize MPI
	MPI::Init(argc,argv);

	// Get the number of processes
	int numP=MPI::COMM_WORLD.Get_size();

	// Get the ID of the process
	int myId=MPI::COMM_WORLD.Get_rank();

	if(argc < 4){
		// Only the first process prints the output message
		if(!myId){
			std::cout << "ERROR: The syntax of the program is ./summa m k n"
					<< std::endl;
		}
		MPI::COMM_WORLD.Abort(1);
	}

	int m = atoi(argv[1]);
	int k = atoi(argv[2]);
	int n = atoi(argv[3]);

    int gridDim = sqrt(numP);
    // Check if a square grid could be created
    if(gridDim*gridDim != numP){
		// Only the first process prints the output message
		if(!myId)
			std::cout << "ERROR: The number of processes must be square"
					<< std::endl;
	
		MPI::COMM_WORLD.Abort(1);
    }

	if((m%gridDim) || (n%gridDim) || (k%gridDim)){
		// Only the first process prints the output message
		if(!myId)
			std::cout << "ERROR: 'm', 'k' and 'n' must be multiple of sqrt(numP)" << std::endl;
		
		MPI::COMM_WORLD.Abort(1);
	}

	if((m < 1) || (n < 1) || (k<1)){
		// Only the first process prints the output message
		if(!myId)
			std::cout << "ERROR: 'm', 'k' and 'n' must be higher than 0" << std::endl;
		
		MPI::COMM_WORLD.Abort(1);
	}

	float *A;
	float *B;
	float *C;

	// Only one process reads the data from the files and broadcasts the data
	if(!myId){
		A = new float[m*k];
		readInput(m, k, A);
		B = new float[k*n];
		readInput(k, n, B);
		C = new float[m*n];
	}

	// The computation is divided by 2D blocks
	int blockRowsA = m/gridDim;
	int blockRowsB = k/gridDim;
	int blockColsB = n/gridDim;

	// Create the datatypes of the blocks
	MPI::Datatype blockAType = MPI::FLOAT.Create_vector(blockRowsA, blockRowsB, k);
	MPI::Datatype blockBType = MPI::FLOAT.Create_vector(blockRowsB, blockColsB, n);
	MPI::Datatype blockCType = MPI::FLOAT.Create_vector(blockRowsA, blockColsB, n);
	blockAType.Commit(); blockBType.Commit(); blockCType.Commit();

	float* myA = new float[blockRowsA*blockRowsB];
	float* myB = new float[blockRowsB*blockColsB];
	float* myC = new float[blockRowsA*blockColsB]();
	float* buffA = new float[blockRowsA*blockRowsB];
	float* buffB = new float[blockRowsB*blockColsB];

	// Measure the current time
	MPI::COMM_WORLD.Barrier();
	double start = MPI::Wtime();

	MPI::Request req;

	// Scatter A and B
	if(!myId){
		for(int i=0; i<gridDim; i++){
			for(int j=0; j<gridDim; j++){
				req = MPI::COMM_WORLD.Isend(A+i*blockRowsA*k+j*blockRowsB, 1, blockAType, i*gridDim+j, 0);
				req = MPI::COMM_WORLD.Isend(B+i*blockRowsB*n+j*blockColsB, 1, blockBType, i*gridDim+j, 0);
			}
		}
	}

	MPI::COMM_WORLD.Recv(myA, blockRowsA*blockRowsB, MPI::FLOAT, 0, 0);
	MPI::COMM_WORLD.Recv(myB, blockRowsB*blockColsB, MPI::FLOAT, 0, 0);

	// Create the communicators
	MPI::Intercomm rowComm = MPI::COMM_WORLD.Split(myId/gridDim, myId%gridDim);
	MPI::Intercomm colComm = MPI::COMM_WORLD.Split(myId%gridDim, myId/gridDim);

	// The main loop
	for(int i=0; i<gridDim; i++){
		// The owners of the block to use must copy it to the buffer
		if(myId%gridDim == i){
			memcpy(buffA, myA, blockRowsA*blockRowsB*sizeof(float));
		}
		if(myId/gridDim == i){
			memcpy(buffB, myB, blockRowsB*blockColsB*sizeof(float));
		}

		// Broadcast along the communicators
		rowComm.Bcast(buffA, blockRowsA*blockRowsB, MPI::FLOAT, i);
		colComm.Bcast(buffB, blockRowsB*blockColsB, MPI::FLOAT, i);

		// The multiplication of the submatrices
		for(int i=0; i<blockRowsA; i++){
			for(int j=0; j<blockColsB; j++){
				for(int l=0; l<blockRowsB; l++){
					myC[i*blockColsB+j] += buffA[i*blockRowsB+l]*buffB[l*blockColsB+j];
				}
			}
		}
	}

	// Only process 0 writes
	// Gather the final matrix to the memory of process 0
	if(!myId){
		for(int i=0; i<blockRowsA; i++)
			memcpy(&C[i*n], &myC[i*blockColsB], blockColsB*sizeof(float));		

		for(int i=0; i<gridDim; i++)
			for(int j=0; j<gridDim; j++)
				if(i || j)
					MPI::COMM_WORLD.Recv(&C[i*blockRowsA*n+j*blockColsB], 1, blockCType, i*gridDim+j, 0);
	} else 
		MPI::COMM_WORLD.Send(myC, blockRowsA*blockColsB, MPI::FLOAT, 0, 0);

	// Measure the current time
	double end = MPI::Wtime();

	if(!myId){
    	std::cout << "Time with " << numP << " processes: " << end-start << " seconds" << std::endl;
    	printOutput(m, n, C);
		delete [] A;
		delete [] B;
		delete [] C;
	}

	MPI::COMM_WORLD.Barrier();

	delete [] myA;
	delete [] myB;
	delete [] myC;
	delete [] buffA;
	delete [] buffB;

	// Terminate MPI
	MPI::Finalize();
	return 0;
}
