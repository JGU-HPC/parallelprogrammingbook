#include <iostream>
#include <atomic>

template <
    typename x_value_t,
    typename y_value_t,
    typename z_value_t>
struct state_t {

    x_value_t x;
    y_value_t y;
    z_value_t z;
};

template <
    typename R,
    typename S,
    typename T>
void status() { // report size and if lock-free
    typedef std::atomic<state_t<R,S,T>> atomic_state_t;
    std::cout << sizeof(atomic_state_t) << "\t"
              << atomic_state_t().is_lock_free()
              << std::endl;
}

int main () {

    std::cout << "size\tlock_free?" << std::endl;

    status<uint8_t,  uint8_t,  uint8_t >();
    status<uint16_t, uint8_t,  uint8_t >();
    status<uint16_t, uint16_t, uint8_t >();
    status<uint32_t, uint16_t, uint16_t>();
    status<uint32_t, uint32_t, uint16_t>();
    status<uint64_t, uint32_t, uint32_t>();
}
