#ifndef CBF_GENERATOR_HPP
#define CBF_GENERATOR_HPP

#include <random>
#include <cstdint>

template <
    typename index_t,
    typename value_t,
    typename label_t>
void generate_cbf(
    value_t * data,
    label_t * labels,
    index_t num_entries,
    index_t num_features) {

    std::mt19937 engine(42);
    std::uniform_int_distribution<index_t> lower_dist(0.125*num_features,
                                                      0.250*num_features);
    std::uniform_int_distribution<index_t> delta_dist(0.250*num_features,
                                                      0.750*num_features);
    std::uniform_real_distribution<value_t> normal_dist(0, 1);

    // create the labels (0: Cylinder, 1:Bell, 2:Funnel)
    for (index_t entry = 0; entry < num_entries; entry++)
        labels[entry] = entry % 3;

    for (index_t entry = 0; entry < num_entries; entry++) {

        const index_t a   = lower_dist(engine);
        const index_t bma = delta_dist(engine);
        const value_t amp = normal_dist(engine)+6;

        // Cylinder
        if (labels[entry] == 0) {
            for (index_t index = 0; index < num_features; index++) {
                const value_t value = (index >= a && index < a+bma) ? amp : 0;
                data[entry*num_features+index] = value+normal_dist(engine);
            }
        }

        // Bell
        if (labels[entry] == 1) {
            for (index_t index = 0; index < num_features; index++) {
                const value_t delta = value_t(index)-value_t(a);
                const value_t value = (index >= a && index < a+bma) ?
                                      amp*delta/bma : 0;
                data[entry*num_features+index] = value+normal_dist(engine);
            }
        }

        // Funnel
        if (labels[entry] == 2) {
            for (index_t index = 0; index < num_features; index++) {
                const value_t delta = value_t(a+bma)-value_t(index);
                const value_t value = (index >= a && index < a+bma) ?
                                      amp*delta/bma : 0;
                data[entry*num_features+index] = value+normal_dist(engine);
            }
        }
    }
}


#endif
