#include <upcxx.h>	

void readImage(int rows, int cols, int *image){

	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){
			image[i*cols+j] = (i*j)%256;
		}

	}
}

void printHistogram(upcxx::shared_array<int> h){

	for(int i=0; i<256; i++)
		std::cout << h[i] << " ";

	std::cout << std::endl;
}

upcxx::shared_array<upcxx::shared_lock> locks;

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

	// Declare the histogram
	upcxx::shared_array<int> histogram(256);
	for(int i=myId; i<256; i+=numT)
		histogram[i] = 0;

	// Initialize the locks
	locks.init(256);
	for(int i=myId; i<256; i+=numT)
		new (locks[i].raw_ptr()) upcxx::shared_lock(myId);

	// Threads must wait until all locks and histogram have been initialized
	upcxx::barrier();

	// Examine the local image
	for(int i=0; i<blockRows*cols; i++){
		// Close the lock to access the shared array
		((upcxx::shared_lock) locks[myImage[i]]).lock();

		histogram[myImage[i]] = histogram[myImage[i]]+1;

		// Open the lock again
		((upcxx::shared_lock) locks[myImage[i]]).unlock();
	}

	// All threads must have finished their local computation
	upcxx::barrier();

	if(!myId)
		printHistogram(histogram);

	// Deallocate the local memory
	upcxx::deallocate<int>(p[myId]);

	// Terminate UPC++
	upcxx::finalize();
	return 0;
}
