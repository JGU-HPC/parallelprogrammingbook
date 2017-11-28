#include <iostream>
#include <omp.h>

template <
    typename value_t>
struct binop {

    constexpr static value_t neutral = 0;

    inline value_t operator()(
        const value_t& lhs,
        const value_t& rhs) const {

        const value_t ying = std::abs(lhs);
        const value_t yang = std::abs(rhs);

        return ying > yang ? lhs : rhs;
    }
};

int main () {

    const uint64_t num_iters = 1UL << 20;
    int64_t result = binop<int64_t>::neutral;

    #pragma omp declare reduction(custom_op : int64_t : \
    omp_out = binop<int64_t>()(omp_out, omp_in))        \
    initializer (omp_priv=binop<int64_t>::neutral)


    # pragma omp parallel for reduction(custom_op:result)
    for (uint64_t i = 0; i < num_iters; i++)
        result = binop<int64_t>()(result, i&1 ? -i : i);

    std::cout << result << std::endl;


}
