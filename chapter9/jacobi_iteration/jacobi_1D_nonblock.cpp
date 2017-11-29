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
            data[i*cols+j] = (i/121+j/121) % 2;
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

	if(argc < 6){
		// Only the first process prints the output message
		if(!myId){
			std::cout << "ERROR: The syntax of the program is ./jacobi-1D-unblock inputFile rows cols outputFile errThreshold"
					<< std::endl;
		}
		MPI::COMM_WORLD.Abort(1);
	}

	std::string inputFile = argv[1];
	int rows = atoi(argv[2]);
	int cols = atoi(argv[3]);
	std::string outputFile = argv[4];
	float errThres = atof(argv[5]);

	if((rows < 1) || (cols < 1)){
		// Only the first process prints the output message
		if(!myId){
			std::cout << "ERROR: The number of rows and columns must be higher than 0" << std::endl;
		}
		MPI::COMM_WORLD.Abort(1);
	}

	float *data;

	// Only one process reads the data from the file and broadcasts the data
	if(!myId){
		data = new float[rows*cols];
		readInput(inputFile, rows, cols, data);
	}

	// The computation is divided by rows
	int blockRows = rows/numP;
	int myRows = blockRows;

	// For the cases that 'rows' is not multiple of numP
	if(myId < rows%numP){
		myRows++;
	}

	// Measure the current time
	double start = MPI::Wtime();

	// Arrays for the chunk of data to work
	float *myData = new float[myRows*cols];
	float *buff = new float[myRows*cols];

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

			if(i < rows%numP){
				sendCounts[i] = (blockRows+1)*cols;
			} else {
				sendCounts[i] = blockRows*cols;
			}
		}
	}

	// Scatter the input matrix
	MPI::COMM_WORLD.Scatterv(data, sendCounts, displs, MPI::FLOAT, myData, myRows*cols, MPI::FLOAT, 0);
	memcpy(buff, myData, myRows*cols*sizeof(float));

	float error = errThres+1.0;
	float myError;

	// Buffers to receive the rows
	float *prevRow = new float[cols];
	float *nextRow = new float[cols];
	MPI::Status status;
	MPI::Request request[4];

	while(error > errThres){
		if(myId > 0){
			// Send the first row to the previous process
			request[0] = MPI::COMM_WORLD.Isend(myData, cols, MPI::FLOAT, myId-1, 0);
			// Receive the previous row from the previous process
			request[1] = MPI::COMM_WORLD.Irecv(prevRow, cols, MPI::FLOAT, myId-1, 0);
		}

		if(myId < numP-1){
			// Send the last row to the next process
			request[2] = MPI::COMM_WORLD.Isend(&myData[(myRows-1)*cols], cols, MPI::FLOAT, myId+1, 0);
			// Receive the next row from the next process
			request[3] = MPI::COMM_WORLD.Irecv(nextRow, cols, MPI::FLOAT, myId+1, 0);
		}

		// Update the main block
		for(int i=1; i<myRows-1; i++){
			for(int j=1; j<cols-1; j++){
				// calculate discrete laplacian by averaging 4-neighbourhood
				buff[i*cols+j]= 0.25f*(myData[(i+1)*cols+j]+myData[i*cols+j-1]+myData[i*cols+j+1]+myData[(i-1)*cols+j]);
			}
		}

		// Update the first row
		if(myId > 0){
			request[1].Wait(status);
			if(myRows > 1){
				for(int j=1; j<cols-1; j++){
					buff[j] = 0.25*(myData[cols+j]+myData[j-1]+myData[j+1]+prevRow[j]);
				}
			}
		}

		// Update the last row
		if(myId < numP-1){
			request[3].Wait(status);
			if(myRows > 1){
				for(int j=1; j<cols-1; j++){
					buff[(myRows-1)*cols+j] = 0.25*(nextRow[j]+myData[(myRows-1)*cols+j-1]+
						myData[(myRows-1)*cols+j+1]+myData[(myRows-2)*cols+j]);
				}
			}
		}

		// Calculate the error of the block
		myError = 0.0;
		for(int i=0; i<myRows; i++){
			for(int j=1; j<cols-1; j++){
                // determine difference between 'data' and 'buff' and add up error
                myError += (myData[i*cols+j]-buff[i*cols+j])*(myData[i*cols+j]-buff[i*cols+j]);
			}
		}

		memcpy(myData, buff, myRows*cols*sizeof(float));

		// Sum the error of all the processes
		// The output is stored in the variable 'error' of all processes
		MPI::COMM_WORLD.Allreduce(&myError, &error, 1, MPI::FLOAT, MPI::SUM);
	}

	// Only process 0 writes
	// Gather the final matrix to the memory of process 0
	MPI::COMM_WORLD.Gatherv(myData, myRows*cols, MPI::FLOAT, data, sendCounts, displs, MPI::FLOAT, 0);

	// Measure the current time
	double end = MPI::Wtime();

	if(!myId){
    	std::cout << "Time with " << numP << " processes: " << end-start << " seconds" << std::endl;
    	printOutput(outputFile, rows, cols, data);
		delete [] data;
	}

	delete [] myData;
	delete [] buff;
	delete [] prevRow;
	delete [] nextRow;

	// Terminate MPI
	MPI::Finalize();
	return 0;
}
