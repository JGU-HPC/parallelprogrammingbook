#include "../include/hpc_helpers.hpp"  // timers
#include "../include/binary_IO.hpp"    // load images

#include <limits>   // numerical limits of data types
#include <vector>   // std::vector
#include <cmath>    // std::max

template <
    typename value_t,
    typename index_t>
void softmax_regression(
    value_t * input,
    value_t * output,
    value_t * weights,
    value_t * bias,
    index_t   n_input,
    index_t   n_output) {

    for (index_t j = 0; j < n_output; j++) {
        value_t accum = value_t(0);
        for (index_t k = 0; k < n_input; k++)
            accum += weights[j*n_input+k]*input[k];
        output[j] = accum + bias[j];
    }

    value_t norm = value_t(0);
    value_t mu = std::numeric_limits<value_t>::lowest();

    // compute mu = max(z_j)
    for (index_t index = 0; index < n_output; index++)
        mu = std::max(mu, output[index]);

    // compute y_j = exp(z_j-mu)
    for (index_t j = 0; j < n_output; j++)
        output[j] = std::exp(output[j]-mu);

    // compute Z = sum_j z_j
    for (index_t j = 0; j < n_output; j++)
        norm += output[j];

    // compute z_j/Z
    for (index_t j = 0; j < n_output; j++)
        output[j] /= norm;
}

template <
    typename value_t,
    typename index_t>
index_t argmax(
    value_t * neurons,
    index_t   n_units) {

    index_t arg = 0;
    value_t max = std::numeric_limits<value_t>::lowest();

    for (index_t j = 0; j < n_units; j++) {
        const value_t val = neurons[j];
        if (val > max) {
            arg = j;
            max = val;
        }
    }

    return arg;
}

template <
    typename value_t,
    typename index_t>
value_t accuracy(
    value_t * input,
    value_t * label,
    value_t * weights,
    value_t * bias,
    index_t   num_entries,
    index_t   num_features,
    index_t   num_classes) {

    index_t counter = index_t(0);

    # pragma omp parallel for reduction(+: counter)
    for (index_t i= 0; i < num_entries; i++) {

        value_t output[num_classes];
        const uint64_t input_off = i*num_features;
        const uint64_t label_off = i*num_classes;

        softmax_regression(input+input_off, output, weights,
                           bias, num_features, num_classes);

        counter +=  argmax(output, num_classes) ==
                    argmax(label+label_off, num_classes);
    }

    return value_t(counter)/value_t(num_entries);
}

template <
    typename value_t,
    typename index_t>
void train(
    value_t * input,
    value_t * label,
    value_t * weights,
    value_t * bias,
    index_t   num_entries,
    index_t   num_features,
    index_t   num_classes,
    index_t   num_iters=32,
    value_t   epsilon=1E-1) {

    value_t * grad_bias    = new value_t[num_classes];
    value_t * grad_weights = new value_t[num_features*num_classes];

    # pragma omp parallel
    for (uint64_t index = 0; index < num_iters; index++){

        // zero the gradients
        # pragma omp single
        for (index_t j = 0; j < num_classes; j++)
            grad_bias[j] = value_t(0);

        # pragma omp for collapse(2)
        for (index_t j = 0; j < num_classes; j++)
            for (index_t k = 0; k < num_features; k++)
                grad_weights[j*num_features+k] = value_t(0);

        // compute softmax contributions
        # pragma omp for \
            reduction(+:grad_bias[0:num_classes]) \
            reduction(+:grad_weights[0:num_classes*num_features])
        for (index_t i = 0; i < num_entries; i++) {

            const index_t inp_off = i*num_features;
            const index_t out_off = i*num_classes;

            value_t * output = new value_t[num_classes];
            softmax_regression(input+inp_off,
                               output,
                               weights,
                               bias,
                               num_features,
                               num_classes);

            for (index_t j = 0; j < num_classes; j++) {

                const index_t out_ind = out_off+j;
                const value_t lbl_res = output[j]-label[out_ind];

                grad_bias[j] += lbl_res;

                const index_t wgt_off = j*num_features;
                for (index_t k = 0; k < num_features; k++) {

                    const index_t wgt_ind = wgt_off+k;
                    const index_t inp_ind = inp_off+k;
                    grad_weights[wgt_ind] += lbl_res*input[inp_ind];
                }
            }
            delete [] output;
        }

        // adjust bias vector
        # pragma omp single
        for (index_t j = 0; j < num_classes; j++)
            bias[j] -= epsilon*grad_bias[j]/num_entries;

        // adjust weight matrix
        # pragma omp for collapse(2)
        for (index_t j = 0; j < num_classes; j++)
            for (index_t k = 0; k < num_features; k++)
                weights[j*num_features+k] -=
                    epsilon*grad_weights[j*num_features+k]/num_entries;
    }

    delete [] grad_bias;
    delete [] grad_weights;
}


int main() {

    const uint64_t num_features = 28*28;
    const uint64_t num_classes = 10;
    const uint64_t num_entries = 65000;

    std::vector<float> input(num_entries*num_features);
    std::vector<float> label(num_entries*num_classes);

    std::vector<float> weights(num_classes*num_features);
    std::vector<float> bias(num_classes);

    load_binary(input.data(), input.size(), "./data/X.bin");
    load_binary(label.data(), label.size(), "./data/Y.bin");
    //load_binary(weights.data(), weights.size(), "./data/A.bin");
    //load_binary(bias.data(), bias.size(), "./data/b.bin");

    while(true) {

        TIMERSTART(training)
        train(input.data(),
              label.data(),
              weights.data(),
              bias.data(),
              55000UL,
              num_features,
              num_classes);
        TIMERSTOP(training)

        const uint64_t off_inp = 55000*num_features;
        const uint64_t off_lbl = 55000*num_classes;

        TIMERSTART(accuracy)
        auto acc = accuracy(input.data()+off_inp,
                            label.data()+off_lbl,
                            weights.data(),
                            bias.data(),
                            10000UL,
                            num_features,
                            num_classes);
        TIMERSTOP(accuracy)

        std::cout << "accuracy_test: " << acc << std::endl;
    }
}
