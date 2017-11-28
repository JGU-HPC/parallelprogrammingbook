#include <iostream>

int main() {
    // run the statement after the pragma in the current team
    #pragma omp parallel
    std::cout << "Hello world!" << std::endl;
}
