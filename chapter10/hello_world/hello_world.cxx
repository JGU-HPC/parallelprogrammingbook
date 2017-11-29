#include <upcxx.h>

int main (int argc, char *argv[]){
	// Initialize UPC++
	upcxx::init(&argc, &argv);

	// Every process prints Hello
	std::cout << "Thread " << upcxx::myrank() << " of " << upcxx::ranks() << ": Hello, world!" << std::endl;

	// Terminate UPC++
	upcxx::finalize();
	return 0;
}
