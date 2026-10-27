#pragma once
#include "Wire.h"
#define DRV2605_REG_RTPIN 0x02
#define DRV2605_MODE_REALTIME 0x05
struct Adafruit_DRV2605 {
  bool begin(TwoWire *) { return true; }
  void useERM() {}
  void setRealtimeValue(uint8_t) {}
  void setMode(uint8_t) {}
};
