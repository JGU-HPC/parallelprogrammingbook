#include <upcxx.h>	

upcxx::shared_array<upcxx::atomic<int>> histogram;

void readImage(int rows, int cols, int *image){

	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){
			image[i*cols+j] = (i*j)%256;
		}

	}
}

void printHistogram(){

	for(int i=0; i<256; i++)
		std::cout << ((upcxx::atomic<int>) histogram[i]).load() << " ";

	std::cout << std::endl;
}

int main (int argc, char *argv[]){
	// Initialize UPC++
	upcxx::init(&argc, &argv);

	int numT = upcxx::ranks();
	int myId = upcxx::myrank();

	if(argc < 3){
		// Only the first process prints the output message
		if(!MYTHREAD){
			std::cout << "ERROR: The syntax of the program is "
                                  << argv[0] << " rows cols" << std::endl;
		}
		exit(1);
	}

	int rows = atoi(argv[1]);
	int cols = atoi(argv[2]);

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

	if(rows%numT){
		// Only the first process prints the output message
		if(!myId)
			std::cout << "ERROR: 'n' must multiple of the number of processes" << std::endl;
		exit(1);
	}

	// Create the array of global pointers
	upcxx::shared_array<upcxx::global_ptr<int>> p(numT);
	
	// Each thread allocates the memory of its subspace
	int blockRows = rows/numT;
	p[myId] = upcxx::allocate(myId, blockRows*cols*sizeof(int));

	// Thread 0 reads the image and copies the fragments
	if(!myId){
		int *block = new int[blockRows*cols];
		int *block2 = new int[blockRows*cols];
		upcxx::event e;

		readImage(blockRows, cols, block);

		for(int i=0; i<numT-1; i++){
			upcxx::async_copy<int>(block, p[i], blockRows*cols, &e);

			// Overlap the copy with reading the next fragment
			// We cannot use "block" for the next fragment because it has not been sent 
			readImage(blockRows, cols, block2);

			// The previous copy must have finished to reuse its buffer
			e.wait();
			int *aux = block;
			block = block2;
			block2 = aux;
		}

		// The last copy does not overlap
		upcxx::copy<int>(block, p[numT-1], blockRows*cols);

		delete block;
		delete block2;
	}

	// Threads must wait until Thread 0 has copied the fragments of the text
	upcxx::barrier();

	// Privatize the pointer
	int *myImage = (int *) (upcxx::global_ptr<int>) p[myId];

	// Check whether it is really local
	if(!((upcxx::global_ptr<int>) p[myId]).is_local())
		std::cout << "Thread " << myId << " not accessing local memory" << std::endl;

std::cout << "To init histogram" << std::endl;

	// Initialize the histogram
	histogram.init(256);
	for(int i=myId; i<256; i+=numT){
		std::cout << "Before, histogram[" << i << "] = " << histogram[i].get().load() << std::endl;
		//((upcxx::atomic<int>) histogram[i]).store(1);
		histogram[i].get().store(1);
		std::cout << "After, histogram[" << i << "] = " << histogram[i].get().load() << std::endl;
	}

std::cout << "histogram initialized" << std::endl;

	// Threads must wait until the histogram has been initialized
	upcxx::barrier();

	// Examine the local image
	/*for(int i=0; i<blockRows*cols; i++)
		// Atomic add
		((upcxx::atomic<int>) histogram[myImage[i]]).fetch_add(1);*/

	// All threads must have finished their local computation
	upcxx::barrier();

	if(!myId)
		printHistogram();

	// Deallocate the local memory
	upcxx::deallocate<int>(p[myId]);

	// Terminate UPC++
	upcxx::finalize();
	return 0;
}
