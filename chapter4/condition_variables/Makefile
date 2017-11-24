CXX= g++
CXXFLAGS= -std=c++14 -O2 -pthread

all: alarm_clock ping_pong one_shot_alarm_clock

alarm_clock: alarm_clock.cpp
	$(CXX) alarm_clock.cpp $(CXXFLAGS) -o alarm_clock

one_shot_alarm_clock: one_shot_alarm_clock.cpp
	$(CXX) one_shot_alarm_clock.cpp $(CXXFLAGS) -o one_shot_alarm_clock

ping_pong: ping_pong.cpp
	$(CXX) ping_pong.cpp $(CXXFLAGS) -o ping_pong

clean:
	rm -rf alarm_clock
	rm -rf one_shot_alarm_clock
	rm -rf ping_pong
