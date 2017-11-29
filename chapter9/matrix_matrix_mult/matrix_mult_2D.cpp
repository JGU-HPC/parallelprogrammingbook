#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include "mpi.h"

struct params{
  int m, k, n;
  float alpha;
};

void readParams(params* p){
	p->m = 3000;
	p->k = 3000;
	p->n = 3000;
	p->alpha = 1.5;
}

void readInput(int m, int k, int n, float *A, float *B){

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
    for(int i=0; i<m; i++)
        for(int j=0; j<k; j++)
            A[i*k+j] = (i+j) % 2;

    for(int i=0; i<k; i++)
        for(int j=0; j<n; j++)
            B[i*n+j] = (i+j) % 2;
}

void printOutput(int rows, int cols, float *data){

	FILE *fp = fopen("out2D.txt", "wb");
	// Check if the file was opened
	if(fp == NULL){
		std::cout << "ERROR: Output file out2D.txt could not be opened" << std::endl;
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
	int gridDim = sqrt(numP);

	// Get the ID of the process
	int myId=MPI::COMM_WORLD.Get_rank();

	if(gridDim*gridDim != numP){
		// Only the first process prints the output message
		if(!myId)
			std::cout << "ERROR: the number of processes must be square" << std::endl;
		
		MPI::COMM_WORLD.Abort(1);
	}
	
	params p;

	// Arguments for the datatype
	int blockLengths[2] = {3, 1};
	MPI::Aint lb, extent;
	MPI::INT.Get_extent(lb, extent);
	MPI::Aint disp[2] = {0, 3*extent};
	MPI::Datatype types[2] = {MPI::INT, MPI::FLOAT}; 

	// Create the datatype for the parameters
	MPI::Datatype paramsType = MPI::INT.Create_struct(2, blockLengths, disp, types);
	paramsType.Commit();

	if(!myId){
		// Process 0 reads the parameters from a configuration file
		readParams(&p);
	}

	// Broadcast of all the parameters using one message with a struct
	MPI::COMM_WORLD.Bcast(&p, 1, paramsType, 0);

	if((p.m < 1) || (p.n < 1) || (p.k<1)){
		// Only the first process prints the output message
		if(!myId)
			std::cout << "ERROR: 'm', 'k' and 'n' must be higher than 0" << std::endl;
		
		MPI::COMM_WORLD.Abort(1);
	}

	if((p.m%gridDim) || (p.n%gridDim)){
		// Only the first process prints the output message
		if(!myId)
			std::cout << "ERROR: 'm', 'n' must be multiple of the grid dimensions" << std::endl;
		
		MPI::COMM_WORLD.Abort(1);
	}

	float *A, *B, *C, *myA, *myB, *myC;
	int blockRows = p.m/gridDim;
	int blockCols = p.n/gridDim;
	MPI::Request req;

	// Only one process reads the data from the files
	if(!myId){
		A = new float[p.m*p.k];
		B = new float[p.k*p.n];
		readInput(p.m, p.k, p.n, A, B);
	}

	MPI::COMM_WORLD.Barrier();
	double start = MPI::Wtime();

	// Create the datatype for a block of rows of A
	MPI::Datatype rowsType = MPI::FLOAT.Create_contiguous(blockRows*p.k);
	rowsType.Commit();

	// Send the rows of A that needs each process
	if(!myId)
		for(int i=0; i<gridDim; i++)
			for(int j=0; j<gridDim; j++)
				req = MPI::COMM_WORLD.Isend(&A[i*blockRows*p.k], 1, rowsType, i*gridDim+j, 0);
			
	myA = new float[blockRows*p.k];
	MPI::COMM_WORLD.Recv(myA, 1, rowsType, 0, 0);				

	// Create the datatype for a block of columns of B
	MPI::Datatype colsType = MPI::FLOAT.Create_vector(p.k, blockCols, p.n);
	colsType.Commit();

	// Send the columns of B that needs each process
	if(!myId)
		for(int i=0; i<gridDim; i++)
			for(int j=0; j<gridDim; j++)
				req = MPI::COMM_WORLD.Isend(&B[blockCols*j], 1, colsType, i*gridDim+j, 0);
		
	myB = new float[p.k*blockCols];	
	MPI::COMM_WORLD.Recv(myB, p.k*blockCols, MPI::FLOAT, 0, 0);

	// Array for the chunk of data to work
	myC = new float[blockRows*blockCols];

	// The multiplication of the submatrices
	for(int i=0; i<blockRows; i++)
		for(int j=0; j<blockCols; j++){
			myC[i*blockCols+j] = 0.0;
			for(int l=0; l<p.k; l++)
				myC[i*blockCols+j] += p.alpha*myA[i*p.k+l]*myB[l*blockCols+j];
		}

	// Only process 0 writes
	// Gather the final matrix to the memory of process 0
	// Create the datatype for a block of columns
	MPI::Datatype block2DType = MPI::FLOAT.Create_vector(blockRows, blockCols, p.n);
	block2DType.Commit();

	if(!myId){
		C = new float[p.m*p.n];
		
		for(int i=0; i<blockRows; i++)
			memcpy(&C[i*p.n], &myC[i*blockCols], blockCols*sizeof(float));		

		for(int i=0; i<gridDim; i++)
			for(int j=0; j<gridDim; j++)
				if(i || j)
					MPI::COMM_WORLD.Recv(&C[i*blockRows*p.n+j*blockCols], 1, block2DType, i*gridDim+j, 0);
	} else 
		MPI::COMM_WORLD.Send(myC, blockRows*blockCols, MPI::FLOAT, 0, 0);

	// Measure the current time and print by process 0
	double end = MPI::Wtime();

	if(!myId){
    		std::cout << "Time with " << numP << " processes: " << end-start << " seconds" << std::endl;
    		printOutput(p.m, p.n, C);
		delete [] A;
		delete [] B;
		delete [] C;
	}

	// Delete the types and arrays
	rowsType.Free();
	colsType.Free();
	block2DType.Free();

	delete [] myA;
	delete [] myB;
	delete [] myC;

	// Terminate MPI
	MPI::Finalize();
	return 0;
}
