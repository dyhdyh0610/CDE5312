#pragma once
#include "Arduino.h"
#include <array>
#include <vector>

struct TwoWire {
  uint8_t address = 0, channel = 255, failedChannel = 255;
  std::vector<uint8_t> bytes;
  std::array<uint8_t, 8> actual{};
  void begin(int, int) {}
  void setClock(unsigned) {}
  void setTimeOut(unsigned) {}
  void beginTransmission(uint8_t a) { address = a; bytes.clear(); }
  void write(uint8_t value) { bytes.push_back(value); }
  int endTransmission() {
    if (address == 0x70) {
      if (!bytes.empty()) {
        channel = 255;
        for (uint8_t i = 0; i < 8; i++) if (bytes[0] == (1 << i)) channel = i;
      }
      return 0;
    }
    if (address == 0x5A && channel < 8 && channel != failedChannel) {
      if (bytes.size() == 2 && bytes[0] == 2) actual[channel] = bytes[1];
      return 0;
    }
    return 2;
  }
};
extern TwoWire Wire;
