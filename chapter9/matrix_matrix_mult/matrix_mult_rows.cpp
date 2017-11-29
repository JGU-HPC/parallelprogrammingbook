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
			std::cout << "ERROR: The syntax of the program is ./matrix-mult-rows inputMatA inputMatB outputMat m k n"
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

	float *A;
	// B is replicated in all the processes
	float *B = new float[k*n];
	float *C;

	// Only one process reads the data from the files and broadcasts the data
	if(!myId){
		A = new float[m*k];
		readInput(inputFileA, m, k, A);
		readInput(inputFileB, k, n, B);
		C = new float[m*n];
	}

	// The computation is divided by rows
	int blockRows = m/numP;
	int myRows = blockRows;

	// For the cases that 'rows' is not multiple of numP
	if(myId < m%numP){
		myRows++;
	}

	// Measure the current time
	double start = MPI::Wtime();

	// Arrays for the chunk of data to work
	float *myA = new float[myRows*k];
	float *myC = new float[myRows*n];

	// The process 0 must specify how many rows are sent to each process
	int *sendCounts;
	int *displs;
	if(!myId){
		sendCounts = new int[numP];
		displs = new int[numP];

		displs[0] = 0;

		for(int i=0; i<numP; i++){

			if(i>0){
				displs[i] = displs[i-1]+sendCounts[i-1];
			}

			if(i < m%numP){
				sendCounts[i] = (blockRows+1)*k;
			} else {
				sendCounts[i] = blockRows*k;
			}
		}
	}

	// Scatter the input matrix A
	MPI::COMM_WORLD.Scatterv(A, sendCounts, displs, MPI::FLOAT, myA, myRows*k, MPI::FLOAT, 0);
	// Broadcast the input matrix B
	MPI::COMM_WORLD.Bcast(B, k*n, MPI::FLOAT, 0);

	// The multiplication of the submatrices
	for(int i=0; i<myRows; i++){
		for(int j=0; j<n; j++){
			myC[i*n+j] = 0.0;
			for(int l=0; l<k; l++){
				myC[i*n+j] += myA[i*k+l]*B[l*n+j];
			}
		}
	}

	// Only process 0 writes
	// Gather the final matrix to the memory of process 0
	if(!myId){
		for(int i=0; i<numP; i++){

			if(i>0){
				displs[i] = displs[i-1]+sendCounts[i-1];
			}

			if(i < m%numP){
				sendCounts[i] = (blockRows+1)*n;
			} else {
				sendCounts[i] = blockRows*n;
			}
		}
	}
	MPI::COMM_WORLD.Gatherv(myC, myRows*n, MPI::FLOAT, C, sendCounts, displs, MPI::FLOAT, 0);

	// Measure the current time
	double end = MPI::Wtime();

	if(!myId){
    	std::cout << "Time with " << numP << " processes: " << end-start << " seconds" << std::endl;
    	printOutput(outputFile, m, n, C);
		delete [] A;
		delete [] C;
	}

	delete [] B;
	delete [] myA;
	delete [] myC;

	// Terminate MPI
	MPI::Finalize();
	return 0;
}
