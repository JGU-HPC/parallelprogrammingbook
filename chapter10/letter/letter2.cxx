#include <upcxx.h>	

void readText(int n, char *text){

	int i;
	for(i=0; i<n/4; i++){
		text[i*4] = 'A';
		text[i*4+1] = 'C';
		text[i*4+2] = 'G';
		text[i*4+3] = 'T';
	}

	if(n%4){
		text[i*4] = 'A';
		if((n%4) > 1){
			text[i*4+1] = 'C';
			if((n%4) > 2){
				text[i*4+2] = 'G';
			}
		}
	}
}

int main (int argc, char *argv[]){
	// Initialize UPC++
	upcxx::init(&argc, &argv);

	int numT = upcxx::ranks();
	int myId = upcxx::myrank();

	if(argc < 3){
		// Only the first process prints the output message
		if(!MYTHREAD){
			std::cout << "ERROR: The syntax of the program is ./letter l n"
					<< std::endl;
		}
		exit(1);
	}

	char l = *argv[1];
	int n = atoi(argv[2]);

	if(n < 0){
		// Only the first process prints the output message
		if(!myId)
			std::cout << "ERROR: 'n' must be higher than 0" << std::endl;
		
		exit(1);
	}

	if(n%numT){
		// Only the first process prints the output message
		if(!myId)
			std::cout << "ERROR: 'n' must multiple of the number of processes" << std::endl;
		
		exit(1);
	}

	// Create the array of global pointers
	upcxx::shared_array<upcxx::global_ptr<char>> p(numT);
	
	// Each thread allocates the memory of its subspace
	int blockFactor = n/numT;
	p[myId] = upcxx::allocate(myId, blockFactor*sizeof(char));

	// Thread 0 reads the text and copy the fragments
	if(!myId){
		char *text = new char[blockFactor];
		char *text2 = new char[blockFactor];
		upcxx::event e;

		readText(blockFactor, text);

		for(int i=0; i<numT-1; i++){
			upcxx::async_copy<char>(text, p[i], blockFactor, &e);

			// Overlap the copy with reading the next fragment
			// We cannot use text for teh next fragment because it has not been sent 
			readText(blockFactor, text2);
			char *aux = text;
			text = text2;
			text2 = aux;

			// The previous copy must have finished to reuse its buffer
			e.wait();
		}

		// The last copy does not overlap
		upcxx::copy<char>(text, p[numT-1], blockFactor);

		delete text;
		delete text2;
	}

	// Threads must wait until Thread 0 has copied the fragments of the text
	upcxx::barrier();

	// Privatize the pointer
	int myNumOcc = 0;
	char *myText = (char *) (upcxx::global_ptr<char>) p[myId];

	// Check whether it is really local
	if(!((upcxx::global_ptr<char>) p[myId]).is_local())
		std::cout << "Thread " << myId << " not accessing local memory" << std::endl;

	// Find the local occurrences
	for(int i=0; i<blockFactor; i++)
		if(myText[i] == l)
			myNumOcc++;

	// Reduce number of occurrences
	int numOcc;
	upcxx::reduce(&myNumOcc, &numOcc, 1, 0, UPCXX_SUM, UPCXX_INT);

	if(!myId){
		std::cout << "Letter " << l << " found " << numOcc << " in the text " << std::endl;
	}

	// Deallocate the local memory
	upcxx::deallocate<char>(p[myId]);

	// Terminate UPC++
	upcxx::finalize();
	return 0;
}
