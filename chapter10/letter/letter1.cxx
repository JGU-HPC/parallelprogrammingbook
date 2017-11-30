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
			std::cout << "ERROR: The syntax of the program is "
                                  << argv[0] << " l n" << std::endl;
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
		char *text = new char[100];
		readText(n, text);

		for(int i=0; i<numT; i++)
			upcxx::copy<char>(&text[blockFactor*i], p[i], blockFactor);

		delete text;
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

	// Put the local occurrences accessible to all threads
	upcxx::shared_array<int> occs(numT);
	occs[myId] = myNumOcc;

	// All threads must have put accessible the local occurrences
	upcxx::barrier();

	if(!myId){
		int numOcc = myNumOcc;
		for(int i=1; i<numT; i++)
			numOcc += occs[i];

		std::cout << "Letter " << l << " found " << numOcc << " in the text " << std::endl;
	}

	// Deallocate the local memory
	upcxx::deallocate<char>(p[myId]);

	// Terminate UPC++
	upcxx::finalize();
	return 0;
}
