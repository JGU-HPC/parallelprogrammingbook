#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>

#include "mpi.h"

void readInput(std::string file, int rows, int cols, float *data){

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

void printOutput(std::string file, int rows, int cols, float *data){

	FILE *fp = fopen(file.c_str(), "wb");
	// Check if the file was opened
	if(fp == NULL){
		std::cout << "ERROR: Output file " << file << " could not be opened" << std::endl;
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

	if(argc < 7){
		// Only the first process prints the output message
		if(!myId){
			std::cout << "ERROR: The syntax of the program is ./matrix-mult-cols inputMatA inputMatB outputMat m k n"
					<< std::endl;
		}
		MPI::COMM_WORLD.Abort(1);
	}

	std::string inputFileA = argv[1];
	std::string inputFileB = argv[2];
	std::string outputFile = argv[3];
	int m = atoi(argv[4]);
	int k = atoi(argv[5]);
	int n = atoi(argv[6]);

	if((m < 1) || (n < 1) || (k<1)){
		// Only the first process prints the output message
		if(!myId){
			std::cout << "ERROR: 'm', 'k' and 'n' must be higher than 0" << std::endl;
		}
		MPI::COMM_WORLD.Abort(1);
	}

	// A is replicated in all the processes
	float *A = new float[m*k];
	float *B;
	float *C;

	// Only one process reads the data from the files and broadcasts the data
	if(!myId){
		readInput(inputFileA, m, k, A);
		B = new float[k*n];
		readInput(inputFileB, k, n, B);
		C = new float[m*n];
	}

	// The computation is divided by rows
	int blockCols = n/numP;

	// Create the datatype column
	MPI::Datatype colTypeB = MPI::FLOAT.Create_vector(k, blockCols, n);
	colTypeB.Commit();
	MPI::Datatype colTypeC = MPI::FLOAT.Create_vector(m, blockCols, n);
	colTypeC.Commit();

	// Measure the current time
	double start = MPI::Wtime();

	// Arrays for the chunk of data to work
	float *myB = new float[k*blockCols];
	float *myC = new float[m*blockCols];

	// Broadcast the input matrix A
	MPI::COMM_WORLD.Bcast(A, m*k, MPI::FLOAT, 0);

	MPI::Status status;
	MPI::Request req;

	// Scatter the input matrix B
	if(!myId){
		for(int i=numP-1; i>=0; i--){
			req = MPI::COMM_WORLD.Isend(&B[i*blockCols], 1, colTypeB, i, 0);
		}
	}

	MPI::COMM_WORLD.Recv(myB, k*blockCols, MPI::FLOAT, 0, 0, status);

	// The multiplication of the submatrices
	for(int i=0; i<m; i++){
		for(int j=0; j<blockCols; j++){
			myC[i*blockCols+j] = 0.0;
			for(int l=0; l<k; l++){
				myC[i*blockCols+j] += A[i*k+l]*myB[l*blockCols+j];
			}
		}
	}

	// Only process 0 writes
	// Gather the final matrix to the memory of process 0
	req = MPI::COMM_WORLD.Isend(myC, m*blockCols, MPI::FLOAT, 0, 0);
	if(!myId){
		for(int i=numP-1; i>=0; i--){
			MPI::COMM_WORLD.Recv(&C[i*blockCols], 1, colTypeC, i, 0, status);
		}
	}

	// Measure the current time
	double end = MPI::Wtime();

	colTypeB.Free();
	colTypeC.Free();

	if(!myId){
    	std::cout << "Time with " << numP << " processes: " << end-start << " seconds" << std::endl;
    	printOutput(outputFile, m, n, C);
		delete [] B;
		delete [] C;
	}

	MPI::COMM_WORLD.Barrier();

	delete [] A;
	delete [] myB;
	delete [] myC;

	// Terminate MPI
	MPI::Finalize();
	return 0;
}
