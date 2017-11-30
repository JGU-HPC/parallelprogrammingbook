#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>

#include <upcxx.h>
#include <timer.h>

void readInput(int m, int n, float *A, float *x){

    // checkerboard
	for(int i=0; i<m; i++)
        for(int j=0; j<n; j++)
        	A[i*n+j] = (i+j) % 2;

    for(int i=0; i<n; i++)
        x[i] = i;
}

void printOutput(int n, float *data, bool firstCall){
	FILE *fp;
	if(firstCall){
		fp = fopen("outMatVec.txt", "wb");
		// Check if the file was opened
		if(fp == NULL){
			std::cout << "ERROR: Output file outMatVec.txt could not be opened" << std::endl;
			exit(1);
		}
	} else {
		fp = fopen("outMatVecPriv.txt", "wb");
		// Check if the file was opened
		if(fp == NULL){
			std::cout << "ERROR: Output file outMatVecPriv.txt could not be opened" << std::endl;
			exit(1);
		}
	}

	for(int i=0; i<n; i++)
        fprintf(fp, "%lf ", data[i]);

    fprintf(fp, "\n");
    fclose(fp);
}
	

int main (int argc, char *argv[]){
	// Initialize UPC++
	upcxx::init(&argc, &argv);

	int numP = upcxx::ranks();
	int myId = upcxx::myrank();

	if(argc < 3){
		// Only the first process prints the output message
		if(!MYTHREAD){
			std::cout << "ERROR: The syntax of the program is "
                                  << argv[0] << " m n " << std::endl;
		}
		exit(1);
	}

	size_t m=atoi(argv[1]);
	size_t n=atoi(argv[2]);

	if((m < 1) || (n < 1)){
		// Only the first process prints the output message
		if(!myId)
			std::cout << "ERROR: 'm' and 'n' must be higher than 0" << std::endl;
		
		exit(1);
	}

    if(m % numP){
		// Only the first process prints the output message
		if(!myId)
			std::cout << "ERROR: 'm' must be multiple of the number of processes" << std::endl;
		
		exit(1);
    }

	upcxx::shared_var<upcxx::global_ptr<float>> globalA, globalx, globaly;
	upcxx::global_ptr<float> A, x, y;

    if(!myId){
        // Allocate shared memory with affinity to process 0 to store the whole matrices
        A = upcxx::allocate<float>(0, m*n);
        x = upcxx::allocate<float>(0, n);
        y = upcxx::allocate<float>(0, m);
        readInput(m, n, (float *)A, (float *)x);
		globalA = A;
		globalx = x;
		globaly = y;
    }

    size_t blockRows = m/numP;

	// To measure time
	upcxx::timer t;

	// Barrier to guarantee that 'A' and 'x' are initialized
	upcxx::barrier();
	t.start();

	A = globalA;
	x = globalx;	
	y = globaly;

    // First option, directly access in computation to shared memory
    for(size_t i=myId*blockRows; i<(myId+1)*blockRows; i++){
        y[i] = 0;
        for(size_t j=0; j<n; j++){
            y[i] += A[i*n+j]*x[j];
		}
    }
    t.stop();

    if(!myId){
        std::cout << "Time with " << numP << " processes accessing the elements on demand: " << t.secs() << " seconds" << std::endl;
        printOutput(m, (float *)y, true);
    }

	upcxx::barrier();
	t.start();

    // Second option, use buffers and work over private memory
    // Allocate memory for buffers 
	upcxx::global_ptr<float> privA = upcxx::allocate<float>(myId, blockRows*n);
	upcxx::global_ptr<float> privX = upcxx::allocate<float>(myId, n);
	upcxx::global_ptr<float> privY = upcxx::allocate<float>(myId, blockRows);

    upcxx::copy(A+blockRows*n*myId, privA, blockRows*n);
	upcxx::copy(x, privX, n);
	
    for(size_t i=0; i<blockRows; i++){
        privY[i] = 0;
        for(size_t j=0; j<n; j++)
            privY[i] += privA[i*n+j]*privX[j];
    }

	upcxx::copy(privY, y+myId*blockRows, blockRows);

	// To guarantee that all processes have copied the results into the memory with affinity to process 0
	upcxx::barrier();
	t.stop();
    if(!myId){
        std::cout << "Time with " << numP << " processes accessing copying the elements to private memory: " << t.secs() << " seconds" << std::endl;
        printOutput(m, (float *)y, false);
    }

	upcxx::deallocate(privA);
	upcxx::deallocate(privX);
	upcxx::deallocate(privY);

    if(!myId){
        upcxx::deallocate(A);
        upcxx::deallocate(x);
        upcxx::deallocate(y);
    }

	// Terminate UPC++
	upcxx::finalize();
	return 0;
}
