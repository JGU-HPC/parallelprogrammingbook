#include <iostream>
#include <vector>
#include <string>
#include <omp.h>

int main () {

    std::string result("SIMON SAYS_");
    std::vector<std::string> data {"p", "a", "r", "a", "l", "l",
                                   "e", "l", " ", "p", "r", "o",
                                   "g", "r", "a", "m", "m", "i",
                                   "n", "g", " ", "i", "s", " ",
                                   "f", "u", "n", "!"};

    #pragma omp declare reduction(custom_op : std::string : \
        omp_out = omp_out+omp_in)                           \
        initializer (omp_priv=std::string(""))

    # pragma omp parallel for reduction(custom_op:result) num_threads(2)
    for (uint64_t i = 0; i < data.size(); i++)
        result = result+data[i];

    std::cout << result << std::endl;


}
