#include <iostream>            // std::cout
#include <thread>              // std::thread
#include <mutex>               // std::mutex
#include <chrono>              // std::this_thread::sleep_for
#include <condition_variable>  // std::condition_variable

// convenient time formats (C++14 required)
using namespace std::chrono_literals;

int main() {

    std::mutex mutex;
    std::condition_variable cv;
    bool is_ping = true; // globally shared state

    auto ping = [&] ( ) -> void {
        while (true) {

            // wait to be signalled
            std::unique_lock<std::mutex> unique_lock(mutex);
            cv.wait(unique_lock,[&](){return is_ping;});

            // print "ping" to the command line
            std::this_thread::sleep_for(1s);
            std::cout << "ping" << std::endl;

            // alter state and notify other thread
            is_ping = !is_ping;
            cv.notify_one();
        }
    };

    auto pong = [&] ( ) -> void {
        while (true) {
            // wait to be signalled
            std::unique_lock<std::mutex> unique_lock(mutex);
            cv.wait(unique_lock,[&](){return !is_ping;});

            // print "pong" to the command line
            std::this_thread::sleep_for(1s);
            std::cout << "pong" << std::endl;

            // alter state and notify other thread
            is_ping = !is_ping;
            cv.notify_one();
        }
    };

    std::thread ping_thread(ping);
    std::thread pong_thread(pong);
    ping_thread.join();
    pong_thread.join();
}
