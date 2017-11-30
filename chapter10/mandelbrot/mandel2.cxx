#include <upcxx.h>	

// Output array
upcxx::shared_var<upcxx::global_ptr<int>> outImage;

// Array to know the busy threads
upcxx::shared_array<bool> busyTh;

void printMandel(int *image, int rows, int cols){
		
	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++)
			std::cout << image[i*cols+j] << " ";
		std::cout << std::endl;
	}		
}


int mandel(int i, int j, int rows, int cols, int maxIter){
	float zReal = 0.0, zImag = 0.0, cReal, cImag, temp, lengthsq;

	cReal = -2.0+j*4.0/rows;
 	cImag = 2.0-i*4.0/cols;
	int k = 0;

	do { // Iterate for pixel color
		temp = zReal*zReal-zImag*zImag+cReal;
		zImag = 2.0*zReal*zImag+cImag;
		zReal = temp;
		lengthsq = zReal*zReal+zImag*zImag;
		k++;
	} while (lengthsq<4.0 && k < maxIter);

        if(k>=maxIter) 
		return 0;

        return k;
} 


void mandelRow(int iterRow, int th, int rows, int cols, int maxIter){
	int rowRes[cols];

	for(int j=0; j<cols; j++){
		rowRes[j] = mandel(iterRow, j, rows, cols, maxIter);
	}

	// Copy the partial result
	upcxx::copy<int>(rowRes, (upcxx::global_ptr<int>) &(outImage.get())[iterRow*cols], cols);

	busyTh[th] = false;
}

int main (int argc, char *argv[]){

	// Initialize UPC++
	upcxx::init(&argc, &argv);

	int numT = upcxx::ranks();
	int myId = upcxx::myrank();

	if(numT == 1){
		std::cout << "ERROR: More than 1 thread is required for this master-slave approach"
			<< std::endl;
		exit(1);
	}

	if(argc < 4){
		// Only the first process prints the output message
		if(!MYTHREAD){
			std::cout << "ERROR: The syntax of the program is "
                                  << argv[0] << " rows cols maxIter" << std::endl;
		}
		exit(1);
	}

	int rows = atoi(argv[1]);
	int cols = atoi(argv[2]);
	int maxIter = atoi(argv[3]);

	if(rows < 0){
		// Only the first process prints the output message
		if(!myId)
			std::cout << "ERROR: 'rows' must be higher than 0" << std::endl;
		exit(1);
	}

	if(cols < 0){
		// Only the first process prints the output message
		if(!myId)
			std::cout << "ERROR: 'cols' must be higher than 0" << std::endl;
		exit(1);
	}

	if(maxIter < 0){
		// Only the first process prints the output message
		if(!myId)
			std::cout << "ERROR: 'maxIter' must be higher than 0" << std::endl;
		exit(1);
	}

	if(rows%numT){
		// Only the first process prints the output message
		if(!myId)
			std::cout << "ERROR: 'n' must multiple of the number of processes" << std::endl;
		exit(1);
	}

	// Initialize the lazy array
	// All elements with affinity to Thread 0
	busyTh.init(numT);
	busyTh[myId] = false;

	// To guarantee that busyTh is initialized
	upcxx::barrier();

	// Thread 0 is the master
	if(!myId){
		outImage.put(upcxx::allocate(0, rows*cols*sizeof(int)));
		int nextTh = 1;

		// While there are more rows
		for(int i=0; i<rows; i++){
			// Check whether any thread has finished
			while(busyTh[nextTh]){
				nextTh++;
				if(nextTh == numT){
					nextTh = 1;
				}
			}
			busyTh[nextTh] = true;

			upcxx::async(nextTh)(mandelRow, i, nextTh, rows, cols, maxIter);
			upcxx::advance();
		}

		// Wait for the last row of each thread
		upcxx::async_wait();

		printMandel((int *) outImage.get(), rows, cols);
		// Deallocate the local memory
		upcxx::deallocate<int>(outImage.get());
	}

	// Terminate UPC++
	upcxx::finalize();
	return 0;
}
