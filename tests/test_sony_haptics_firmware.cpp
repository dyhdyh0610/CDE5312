#include "Arduino.h"
#include "Wire.h"
#include <iostream>
#include <stdexcept>

uint32_t testMillis = 0;
TestSerial Serial;
TwoWire Wire;
#include "../arduino/sony_haptics/sony_haptics.ino"

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
bool stopped() {
  return Wire.actual[0] == 0 && Wire.actual[4] == 0 && Wire.actual[2] == 0 && Wire.actual[3] == 0;
}

int main() {
  try {
    setup();
    require(stopped(), "startup must be silent");
    require(Serial.output.find("READY HAPTIC1 15") != std::string::npos, "all four channels must be ready");
    processCommand("D1,1,1,1,0,0");
    require(Wire.actual[0] == 40 && Wire.actual[4] == 40, "upper-right mapping");
    require(Serial.output.find("ACK 1") != std::string::npos, "command acknowledgement");
    processCommand("D1,2,0,0,1,1");
    require(Wire.actual[0] == 0 && Wire.actual[4] == 0 && Wire.actual[2] == 40 && Wire.actual[3] == 40,
            "direction change must clear old channels");
    testMillis += 749;
    loop();
    require(automatic && !stopped(), "valid command remains active for 749ms");
    testMillis++;
    loop();
    require(!automatic && stopped(), "750ms watchdog must stop every channel");

    processCommand("D1,3,1,0,0,0");
    testMillis += 500;
    processCommand("D1,4,1,0,0,0");
    testMillis += 500;
    loop();
    require(automatic, "fresh full frames renew watchdog");
    processCommand("PING");
    testMillis += 250;
    loop();
    require(stopped(), "PING must not keep old motion alive");

    for (const char *bad : {"D1,0,1,0,0,0", "D1,1,-1,0,0,0", "D1,1,2,0,0,0",
                            "D1,4294967296,1,0,0,0", "D1,1,1,0,0", "D1,1,1,0,0,0,junk",
                            "D1,1,1,0,1,0", "D1,1,0,1,0,1"}) {
      processCommand("D1,5,1,0,0,0");
      processCommand(bad);
      require(stopped() && !automatic, "invalid frame must stop, not preserve motion");
    }
    processCommand("D1,4294967295,0,1,0,0");
    require(Wire.actual[4] == 40, "maximum uint32 sequence is supported");
    processCommand("0");
    require(stopped(), "manual emergency stop");

    processCommand("D1,6,1,0,0,0");
    for (char c : std::string("D1,7,1")) Serial.input.push_back(c);
    testMillis += 750;
    loop();
    require(stopped(), "partial serial line cannot extend watchdog");
    Serial.input.push_back('\n');
    loop();
    processCommand("D1,8,1,0,0,0");
    for (int i = 0; i < 85; i++) Serial.input.push_back('9');
    Serial.input.push_back('\n');
    loop();
    require(stopped(), "overflowing line must stop all motors");

    processCommand("D1,9,1,0,0,0");
    Wire.failedChannel = 4;
    processCommand("D1,10,0,1,0,0");
    require(!automatic && Wire.actual[0] == 0, "I2C failure must cancel motion and stop reachable motors");
    Wire.failedChannel = 255;
    stopTest();
    ready[1] = false;
    processCommand("D1,11,0,1,0,0");
    require(stopped(), "missing module must not start other modules");
    ready[1] = true;

    processCommand("2");
    processCommand("one");
    for (uint8_t i = 0; i < 4; i++) {
      loop();
      require(Wire.actual[channels[i]] == 70, "original manual test order/strength");
      testMillis += 500;
      loop();
      testMillis += 500;
    }
    require(stopped() && !running, "manual test completes with every motor off");
    testMillis = 0xFFFFFF00;
    processCommand("D1,12,1,0,0,0");
    testMillis += 749;
    loop();
    require(automatic, "millis wrap must not stop prematurely");
    testMillis++;
    loop();
    require(stopped(), "millis wrap must not disable watchdog");
    std::cout << "Firmware protocol, directions, watchdog, I2C errors and manual tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
