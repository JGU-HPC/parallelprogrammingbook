#include <upcxx.h>	

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

int main (int argc, char *argv[]){

	// Initialize UPC++
	upcxx::init(&argc, &argv);

	int numT = upcxx::ranks();
	int myId = upcxx::myrank();

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

	// Output array
	int blockRows = rows/numT;
	int myImage[blockRows*cols];
	upcxx::shared_var<upcxx::global_ptr<int>> outImage;

	// Only the owner allocates the array to gather the output
	if(!myId){
		outImage.put(upcxx::allocate(0, rows*cols*sizeof(int)));
	}

	// To guarantee that memory is allocated
	upcxx::barrier();

	// Mandel computation of the block of rows
	for(int i=0; i<blockRows; i++)
		for(int j=0; j<cols; j++)
			myImage[i*cols+j] = mandel(i+myId*blockRows, j, rows, cols, maxIter);

	// Copy the partial result
	upcxx::copy<int>(myImage, (upcxx::global_ptr<int>) &(outImage.get())[myId*blockRows*cols], blockRows*cols);

	// All threads must have finished their local computation
	upcxx::barrier();

	if(!myId){
		printMandel((int *) outImage.get(), rows, cols);
		// Deallocate the local memory
		upcxx::deallocate<int>(outImage.get());
	}	

	// Terminate UPC++
	upcxx::finalize();
	return 0;
}
