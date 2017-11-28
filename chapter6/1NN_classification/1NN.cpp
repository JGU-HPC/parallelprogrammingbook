#include <iostream>   // std::cout
#include <limits>     // std::numeric_limits
#include <vector>     // std::vector

// hpc_helpers contains the TIMERSTART and TIMERSTOP macros
// and the no_init_t template that disables implicit type
// initialization
#include "../include/hpc_helpers.hpp"
// binary_IO contains the load_binary function to load
// and store binary data from and to a file
#include "../include/binary_IO.hpp"

template <typename value_t,
          typename index_t>
void all_vs_all(value_t* test,
                value_t* train,
                value_t* delta,
                index_t num_test,
                index_t num_train,
                index_t num_features,
                bool parallel) {

    // coarse-grained parallelism
    #pragma omp parallel for collapse(2) if(parallel)
    for (index_t i = 0; i < num_test; i++)
        for (index_t j = 0; j < num_train; j++) {
            value_t accum = value_t(0);
            // fine-grained parallelism
            // #pragma omp parallel for reduction(+:accum) 
            for (index_t k = 0; k < num_features; k++) {
                const value_t residue = test [i*num_features+k]
                                      - train[j*num_features+k];
                accum += residue*residue;
            }
            delta[i*num_train+j] = accum;
        }
}

template <typename label_t,
          typename value_t,
          typename index_t>
value_t accuracy(label_t* label_test,
                 label_t* label_train,
                 value_t* delta,
                 index_t num_test,
                 index_t num_train,
                 index_t num_classes,
                 bool parallel) {

    index_t counter = index_t(0);

    #pragma omp parallel for reduction(+:counter) if(parallel)
    for (index_t i = 0; i < num_test; i++) {

        // the initial distance is float::max
        // the initial index j_star is some dummy value
        value_t bsf = std::numeric_limits<value_t>::max();
        index_t jst = std::numeric_limits<index_t>::max();

        // find training sample with smallest distance
        for (index_t j = 0; j < num_train; j++) {
            const value_t value = delta[i*num_train+j];
            if (value < bsf) {
                bsf = value;
                jst = j;
            }
        }

        // compare predicted label with original label
        bool match = true;
        for (index_t k = 0; k < num_classes; k++)
            match &= label_test [i  *num_classes+k] ==
                     label_train[jst*num_classes+k];

        counter += match;
    }

    return value_t(counter)/value_t(num_test);
}

int main(int argc, char* argv[]) {
 
   // run parallelized when any command line argument given
    const bool parallel = argc > 1;

    std::cout << "running "
              << (parallel ? "in parallel" : "sequentially")
              << std::endl;

    // the shape of the data matrices
    const uint64_t num_features = 28*28;
    const uint64_t num_classes = 10;
    const uint64_t num_entries = 65000;
    const uint64_t num_train = 55000;
    const uint64_t num_test = num_entries-num_train;

    // memory for the data matrices and all-pair matrix
    std::vector<float> input(num_entries*num_features);
    std::vector<float> label(num_entries*num_classes);
    std::vector<float> delta(num_test*num_train);

    // get the images and labels from disk
    load_binary(input.data(), input.size(), "./data/X.bin");
    load_binary(label.data(), label.size(), "./data/Y.bin");

    TIMERSTART(all_vs_all)
    const uint64_t inp_off = num_train * num_features;
    all_vs_all(input.data() + inp_off,
               input.data(),
               delta.data(),
               num_test, num_train,
               num_features, parallel);
    TIMERSTOP(all_vs_all)

    TIMERSTART(classify)
    const uint64_t lbl_off = num_train * num_classes;
    auto acc = accuracy(label.data() + lbl_off,
                        label.data(),
                        delta.data(),
                        num_test, num_train,
                        num_classes, parallel);
    TIMERSTOP(classify)

    std::cout << "test accuracy: " << acc << std::endl;
}
