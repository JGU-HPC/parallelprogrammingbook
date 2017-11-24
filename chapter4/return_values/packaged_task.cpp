#include <iostream>
#include <cstdint>
#include <vector>
#include <thread>
#include <future>

template <
    typename Func,      // <-- type of function func
    typename ... Args,  // <-- type of arguments arg0,arg1,...
    typename Rtrn=typename std::result_of<Func(Args...)>::type>
auto create_task(       // ^-- type of return value func(args)
    Func &&    func,
    Args && ...args) -> std::packaged_task<Rtrn(void)> {

    // basically build an auxilliary function aux(void)
    // without arguments returning func(arg0,arg1,...)
    auto aux = std::bind(std::forward<Func>(func),
                         std::forward<Args>(args)...);


    // create a task wrapping the auxilliary function:
    // task() executes aux(void) := func(arg0,arg1,...)
    auto task = std::packaged_task<Rtrn(void)>(aux);

    // the return value of aux(void) is assigned to a
    // future object accesible via task.get_future()
    return task;
}

uint64_t fibo(uint64_t n) {

    uint64_t a_0 = 0;
    uint64_t a_1 = 1;

    for (uint64_t index = 0; index < n; index++) {
        const uint64_t tmp = a_0; a_0 = a_1; a_1 += tmp;
    }

    return a_0;
}

int main(int argc, char * argv[]) {

    const uint64_t num_threads = 32;

    std::vector<std::thread> threads;
    std::vector<std::future<uint64_t>> results;

    for (uint64_t id = 0; id < num_threads; id++) {
        auto task = create_task(fibo, id);
        results.emplace_back(task.get_future());
        threads.emplace_back(std::move(task));
    }

    for (auto& result: results)
        std::cout << result.get() << std::endl;

    for (auto& thread: threads)
        thread.detach();
}
