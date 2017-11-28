#include <algorithm>      // std::sort
#include <iostream>       // std::cout
#include <vector>         // std::vector
#include <atomic>         // std::atomic
#include <random>         // std::uniform_int_distribution
#include "threadpool.hpp" // work sharing thread pool

template <
    typename value_t_,
    typename weight_t_>
struct generic_tuple_t {

    value_t_  value;
    weight_t_ weight;

    // expose types
    typedef value_t_ value_t;
    typedef value_t_ weight_t;

    generic_tuple_t(
        value_t_  value_,
        weight_t_ weight_) : value (value_ ),
                             weight(weight_) {}
};

template <
    typename bmask_t_,
    typename value_t_>
struct state_t {

    bmask_t_ bmask=0;
    value_t_ value=0;

    // expose template parameters
    typedef bmask_t_ bmask_t;
    typedef value_t_ value_t;

    // non-default constructors are not allowed
    // when wrapped with std::atomic<state_t>
};

// shortcuts for convenience
typedef uint64_t index_t;
typedef uint32_t bmask_t;
typedef uint32_t value_t;
typedef uint32_t weight_t;
typedef generic_tuple_t<value_t, weight_t> tuple_t;

// the global state encoding the mask and value
std::atomic<state_t<bmask_t, value_t>> global_state;
const value_t capacity (1500);
const index_t num_items (32);
std::vector<tuple_t> tuples;

// our work-sharing thread pool
ThreadPool TP(4);

// initializes Knapsack problem
template <
    typename tuple_t,
    typename index_t>
void init_tuples(
    std::vector<tuple_t>&  tuples,
    index_t num_entries) {

    // recover the types stored in tuple_t
    typedef typename tuple_t::value_t  value_t;
    typedef typename tuple_t::weight_t weight_t;

    // C++11 random number generator
    std::mt19937 engine(0); // mersenne twister
    std::uniform_int_distribution<value_t>  rho_v(80, 100);
    std::uniform_int_distribution<weight_t> rho_w(80, 100);

    // generate pairs of values and weights
    for (index_t index = 0; index < num_entries; index++)
        tuples.emplace_back(rho_v(engine), rho_w(engine));

    // sort two pairs by value/weight density
    auto predicate = [] (const auto& lhs,
                         const auto& rhs) -> bool {
        return lhs.value*rhs.weight > rhs.value*lhs.weight;
    };

    std::sort(tuples.begin(), tuples.end(), predicate);
}

template <
    typename tuple_t,
    typename bmask_t>
void atomic_update(
    tuple_t tuple,
    bmask_t bmask) {

    typedef typename tuple_t::value_t value_t;

    auto g_state = global_state.load();
    auto l_value = tuple.value;
    state_t<bmask_t, value_t> target;

    do {

        // exit if solution is not optimal
        if (g_state.value > l_value)
            return;

        // construct the desired target
        target.value = l_value;
        target.bmask = bmask;

    } while (!global_state.compare_exchange_weak(g_state, target));
}

template <
    typename index_t,
    typename tuple_t>
typename tuple_t::value_t dantzig_bound(
    index_t height,
    tuple_t tuple) {

    auto predicate = [&] (const index_t& i) {
        return i < num_items &&
               tuple.weight < capacity;
    };

    // greedily pack items until backpack full
    for (index_t i = height; predicate(i); i++) {
        tuple.value  += tuples[i].value;
        tuple.weight += tuples[i].weight;
    }

    return tuple.value;
}

template <
    typename index_t,
    typename tuple_t,
    typename bmask_t>
void traverse(
    index_t height,  // height of the binary tree
    tuple_t tuple,   // weight and value up to height
    bmask_t bmask) {  // binary mask up to height

    // check whether item packed or not
    const bool bit  = (bmask >> height) % 2;
    tuple.weight += bit*tuples[height].weight;
    tuple.value  += bit*tuples[height].value;

    // check versus maximum capacity
    if (tuple.weight > capacity)
        return; // my backpack is full

    // update global lower bound if needed
    atomic_update(tuple, bmask);

    // calculate local Danzig upper bound
    // and compare with global upper bound
    auto bsf = global_state.load().value;
    if (dantzig_bound(height+1, tuple) < bsf)
       return;

    // if everything was fine generate new candidate
    if (height+1 < num_items) {
        traverse(height+1, tuple, bmask+(1<<(height+1)));
        traverse(height+1, tuple, bmask);
    }
}

int main () {

    // initialize tuples with random values
    init_tuples(tuples, num_items);

    // traverse left and right branch
    TP.spawn(traverse<index_t, tuple_t, bmask_t>,
             0, tuple_t(0, 0), 0);
    TP.spawn(traverse<index_t, tuple_t, bmask_t>,
             0, tuple_t(0, 0), 1);

    // wait for all tasks to be finished
    TP.wait_and_stop();

    // report the final solution
    auto g_state = global_state.load();
    std::cout << "value " << g_state.value << std::endl;

    auto bmask = g_state.bmask;
    for (index_t i = 0; i < num_items; i++) {
        std::cout << bmask % 2 << " ";
        bmask >>= 1;
    }
    std::cout << std::endl;
}
