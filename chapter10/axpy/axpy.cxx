#include <upcxx.h>
#include <timer.h>

void printOutput(int n, upcxx::shared_array<float, 2> data){
	FILE *fp = fopen("outAXPY.txt", "wb");
	// Check if the file was opened
	if(fp == NULL){
		std::cout << "ERROR: Output file outAXPY.txt could not be opened" << std::endl;
		exit(1);
	}

	float aux;
	for(int i=0; i<n; i++){
		aux = data[i];
        	fprintf(fp, "%lf ", aux);
	}
        fprintf(fp, "\n");

    	fclose(fp);
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
                                  << argv[0] << " n alpha" << std::endl;
		}
		exit(1);
	}

	int n = atoi(argv[1]);
	float alpha = atof(argv[2]);

	if(n < 1){
		// Only the first process prints the output message
		if(!myId)
			std::cout << "ERROR: 'n' must be higher than 0" << std::endl;
		
		exit(1);
	}

	if(n%2){
		// Only the first process prints the output message
		if(!myId)
			std::cout << "ERROR: The blocks (of size 2) must be complete" << std::endl;
		
		exit(1);
	}

	// Declare the shared arrays
	upcxx::shared_array<float, 2> x(n);
	upcxx::shared_array<float, 2> y(n);

	// To measure time
	upcxx::timer t;
	upcxx::barrier();
	t.start();

	// Example accessing memory without affinity
	// Initialize arrays
	for(int i=myId; i<n; i+=numT){
		x[i] = i;
		y[i] = numT;
	}

	// Compute axpy
	for(int i=myId; i<n; i+=numT)
		y[i] += alpha*x[i];

	upcxx::barrier();

	t.stop();
	if(!myId){
		std::cout << "Time with " << numT << " processes using global arrays and without affinity: " << t.secs() << " seconds" << std::endl;
		//printOutput(n, y);
	}  

	// Example accessing memory with affinity
	// Initialize arrays
	upcxx::barrier();
	t.reset();
	t.start();
	for(int i=2*myId; i<n; i+=2*numT){
		x[i] = i;
		y[i] = numT;
		x[i+1] = i+1;
		y[i+1] = numT;
	}

	// Compute axpy
	for(int i=2*myId; i<n; i+=2*numT){
		y[i] += alpha*x[i];
		y[i+1] += alpha*x[i+1];
	}

	upcxx::barrier();
	t.stop();
	if(!myId){
		std::cout << "Time with " << numT << " processes using global arrays and affinity: " << t.secs() << " seconds" << std::endl;
		//printOutput(n, y);
	}  

	// Example with privatization
	float *privX = (float *) &x[myId*2];
	float *privY = (float *) &y[myId*2];
	upcxx::barrier();
	t.reset();
	t.start();

	// Calculate the amount of data
	int myBlocks = n/(2*numT);
	int myIni = 2*myId;
	// When 'n' is not multiple of 'numT'
	if(!myId < (n/2)%numT)
		myBlocks++;

	// Initialize arrays
	for(int i=0; i<myBlocks; i++){
		privX[2*i] = myIni+2*numT*i;
		privX[2*i+1] = myIni+2*numT*i+1;
		privY[2*i] = numT;
		privY[2*i+1] = numT;
	}

	// Compute axpy
	for(int i=0; i<myBlocks*2; i++){	
		privY[i] += alpha*privX[i];
	}

	upcxx::barrier();
	t.stop();
	if(!myId){
		std::cout << "Time with " << numT << " processes using private pointers: " << t.secs() << " seconds" << std::endl;
		printOutput(n, y);
	}

	// Terminate UPC++
	upcxx::finalize();
	return 0;
}
