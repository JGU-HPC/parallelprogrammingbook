#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <chrono>

void readInput(std::string file, int rows, int cols, float *data){

	// Open the file pointer
	/*FILE* fp = fopen(file.c_str(), "rb");

	// Check if the file exists
	if(fp == NULL){
		std::cout << "ERROR: File " << file << " could not be opened" << std::endl;
		exit(1);
	}

	for(int i=0; i<rows*cols; i++){
		if(!fscanf(fp, "%f", &data[i])){
			std::cout << "ERROR: Not enough values in file " << file << std::endl;
			exit(1);
		}
	}*/

    // checkerboard
    for(int i=0; i<rows; i++)
        for(int j=0; j<cols; j++)
            data[i*cols+j] = (i/121+j/121) % 2;

    //fclose(fp);
}

void printOutput(std::string file, int rows, int cols, float *data){

	FILE *fp = fopen(file.c_str(), "wb");
	// Check if the file was opened
	if(fp == NULL){
		std::cout << "ERROR: Output file " << file << " could not be opened" << std::endl;
		exit(1);
	}

    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++)
        	fprintf(fp, "%lf ", data[i*cols+j]);
        fprintf(fp, "\n");
    }

    fclose(fp);
}

int main (int argc, char *argv[]){
	if(argc < 6){
		std::cout << "ERROR: The syntax of the program is " << argv[0] 
                          << " inputFile rows cols outputFile errThreshold"
			  << std::endl;
		exit(1);
	}

	std::string inputFile = argv[1];
	int rows = atoi(argv[2]);
	int cols = atoi(argv[3]);
	std::string outputFile = argv[4];
	float errThres = atof(argv[5]);

	if((rows < 1) || (cols < 1)){
		std::cout << "ERROR: The number of rows and columns must be higher than 0" << std::endl;
		exit(1);
	}

	float *data = new float[rows*cols];
	float *buff = new float[rows*cols];
	readInput(inputFile, rows, cols, data);
	memcpy(buff, data, rows*cols*sizeof(float));

	// Measure the current time
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

	float error = errThres + 1.0;

	while(error > errThres){
		for(int i=1; i<rows-1; i++){
			for(int j=1; j<cols-1; j++){
				// calculate discrete laplacian by averaging 4-neighbourhood
				buff[i*cols+j]= 0.25f*(data[(i+1)*cols+j]+data[i*cols+j-1]+data[i*cols+j+1]+data[(i-1)*cols+j]);
			}
		}

		// Determine the error
		error = 0.0;
		for(int i=1; i<rows-1; i++){
			for(int j=1; j<cols-1; j++){
                // determine difference between 'data' and 'buff' and add up error
                error += (data[i*cols+j]-buff[i*cols+j])*(data[i*cols+j]-buff[i*cols+j]);
			}
		}

		memcpy(data, buff, rows*cols*sizeof(float));
	}

    end = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsed_seconds = end-start;

	std::cout << "Sequential Jacobi with dimensions " << rows << "x" << cols << " in " << elapsed_seconds.count()
			<< " seconds" << std::endl;

	printOutput(outputFile, rows, cols, data);

	delete [] data;
	delete [] buff;

	return 0;
}
