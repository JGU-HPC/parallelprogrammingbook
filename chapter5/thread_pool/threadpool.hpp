#ifndef THREADPOOL_HPP
#define THREADPOOL_HPP

#include <cstdint>
#include <future>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

class ThreadPool {

private:

    // storage for threads and tasks
    std::vector<std::thread> threads;
    std::queue<std::function<void(void)>> tasks;

    // primitives for signaling
    std::mutex mutex;
    std::condition_variable cv, cv_wait;

    // the state of the thread, pool
    bool stop_pool;
    std::atomic<uint32_t> active_threads;
    const uint32_t capacity;

    // custom task factory
    template <
        typename     Func,
        typename ... Args,
        typename Rtrn=typename std::result_of<Func(Args...)>::type>
    auto make_task(
        Func &&    func,
        Args && ...args) -> std::packaged_task<Rtrn(void)> {

        auto aux = std::bind(std::forward<Func>(func),
                             std::forward<Args>(args)...);

        return std::packaged_task<Rtrn(void)>(aux);
    }
    
    // will be executed before execution of a task
    void before_task_hook() {
        active_threads++;
    }
    
    // will be executed after execution of a task
    void after_task_hook() {
        active_threads--;
                    
        if (active_threads == 0 && tasks.empty()) {
            stop_pool = true;
            cv_wait.notify_one();
        }
    }

public:
    ThreadPool(
        uint64_t capacity_) :
        stop_pool(false),     // pool is running
        active_threads(0),    // no work to be done
        capacity(capacity_) { // remember size
        
        // this function is executed by the threads
        auto wait_loop = [this] ( ) -> void {

            // wait forever
            while (true) {

                // this is a placeholder task
                std::function<void(void)> task;

                { // lock this section for waiting
                    std::unique_lock<std::mutex>
                        unique_lock(mutex);

                    // actions must be performed on
                    // wake-up if (i) the thread pool
                    // has been stopped or (ii) there
                    // are still tasks to be processed
                    auto predicate = [this] ( ) -> bool {
                        return  (stop_pool) ||
                               !(tasks.empty());
                    };

                    // wait to be waken up on
                    // aforementioned conditions
                    cv.wait(unique_lock, predicate);

                    // exit if thread pool stopped
                    // and no tasks to be performed
                    if (stop_pool && tasks.empty())
                        return;

                    // else extract task from queue
                    task = std::move(tasks.front());
                    tasks.pop();                    
                    before_task_hook();
                } // here we release the lock

                // execute the task in parallel
                task();

                {   // adjust the thread counter
                    std::lock_guard<std::mutex>
                        lock_guard(mutex);
                    after_task_hook();
                } // here we release the lock
            }
        };

        // initially spawn capacity many threads
        for (uint64_t id = 0; id < capacity; id++)
            threads.emplace_back(wait_loop);
    }

    ~ThreadPool() {

        { // acquire a scoped lock
            std::lock_guard<std::mutex>
                lock_guard(mutex);

            // and subsequently alter
            // the global state to stop
            stop_pool = true;
        } // here we release the lock

        // signal all threads
        cv.notify_all();

        // finally join all threads
        for (auto& thread : threads)
            thread.join();
    }

    template <
        typename     Func,
        typename ... Args,
        typename Pair=Func(Args...),
        typename Rtrn=typename std::result_of<Pair>::type>
    auto enqueue(
        Func &&     func,
        Args && ... args) -> std::future<Rtrn> {

        // create the task, get the future
        // and wrap task in a shared pointer
        auto task = make_task(func, args...);
        auto future = task.get_future();
        auto task_ptr = std::make_shared<decltype(task)>
                        (std::move(task));

        {   // lock the scope
            std::lock_guard<std::mutex>
                lock_guard(mutex); 
                        
            // you cannot reuse pool after being stopped    
            if(stop_pool)
                throw std::runtime_error(
                    "enqueue on stopped ThreadPool"
                );

            // wrap the task in a generic void
            // function void -> void
            auto payload = [task_ptr] ( ) -> void {
                // basically call task()
                task_ptr->operator()();
            };

            // append the task to the queue
            tasks.emplace(payload);
        }

        // tell one thread to wake-up
        cv.notify_one();

        return future;
    }

    // public member function template
    template <
        typename     Func,
        typename ... Args>
    void spawn(
        Func &&     func,
        Args && ... args) {

        // enqueue if idling threads
        if (active_threads < capacity)
            enqueue(func, args...);
        // else process sequential
        else
            func(args...);
    }

    void wait_and_stop() {
        
        // wait for pool being set to stop
        std::unique_lock<std::mutex>
            unique_lock(mutex);
        
        auto predicate = [&] () -> bool {
            return stop_pool;
        };
        
        cv_wait.wait(unique_lock, predicate);
    }
};

#endif
