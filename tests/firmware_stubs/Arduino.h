#pragma once
#include <cstdint>
#include <cstdio>
#include <deque>
#include <string>

extern uint32_t testMillis;
inline uint32_t millis() { return testMillis; }
inline void delay(uint32_t amount) { testMillis += amount; }

struct TestSerial {
  std::deque<char> input;
  std::string output;
  void begin(unsigned) {}
  int available() { return int(input.size()); }
  char read() { char c = input.front(); input.pop_front(); return c; }
  void println(const char *line) { output += std::string(line) + "\n"; }
  template <typename... Args> void printf(const char *format, Args... args) {
    char buffer[512];
    std::snprintf(buffer, sizeof(buffer), format, args...);
    output += buffer;
  }
};
extern TestSerial Serial;
