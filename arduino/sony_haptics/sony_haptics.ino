#include <Wire.h>
#include <Adafruit_DRV2605.h>
#include <stdint.h>
#include <string.h>

// Fixed installation order: UP, RIGHT, DOWN, LEFT.
const uint8_t TCA_ADDR = 0x70;
const uint8_t DRV_ADDR = 0x5A;
const uint8_t channels[4] = {0, 4, 2, 3};
const uint32_t FRAME_TIMEOUT_MS = 750;
const uint32_t ON_MS = 500;
const uint32_t OFF_MS = 500;

Adafruit_DRV2605 drv[4];
bool ready[4] = {false, false, false, false};
uint8_t levels[4] = {0, 0, 0, 0};
uint8_t strength = 40, targetPulses = 1, completedPulses = 0, currentMotor = 0;
bool running = false, motorOn = false, automatic = false;
uint32_t phaseStarted = 0, lastFrameAt = 0;
char inputLine[80];
uint8_t inputLength = 0;
bool inputOverflow = false;

bool selectChannel(uint8_t channel) {
  Wire.beginTransmission(TCA_ADDR);
  Wire.write(uint8_t(1 << channel));
  return Wire.endTransmission() == 0;
}

bool deviceFound(uint8_t address) {
  Wire.beginTransmission(address);
  return Wire.endTransmission() == 0;
}

bool writeLevel(uint8_t motor, uint8_t value) {
  if (!selectChannel(channels[motor])) return false;
  // Checked RTP writes: Adafruit's setRealtimeValue() has no success result.
  Wire.beginTransmission(DRV_ADDR);
  Wire.write(uint8_t(DRV2605_REG_RTPIN));
  Wire.write(value);
  if (Wire.endTransmission() != 0) return false;
  levels[motor] = value;
  return true;
}

bool stopAll() {
  bool success = true;
  for (uint8_t i = 0; i < 4; i++) {
    if (ready[i] && !writeLevel(i, 0)) success = false;
  }
  motorOn = false;
  return success;
}

void stopTest() {
  running = false;
  automatic = false;
  if (!stopAll()) Serial.println("ERR I2C_STOP Disconnect motor power; check wiring.");
}

void channelError() {
  stopTest();
  Serial.println("ERR I2C Communication failed; motors stopped where reachable.");
}

void printReady() {
  uint8_t mask = 0;
  for (uint8_t i = 0; i < 4; i++) if (ready[i]) mask |= uint8_t(1 << i);
  Serial.printf("READY HAPTIC1 %u\n", mask);
}

bool findMotor(uint8_t start) {
  for (uint8_t i = start; i < 4; i++) {
    if (ready[i]) { currentMotor = i; return true; }
  }
  return false;
}

void startTest(uint8_t pulses) {
  running = false;
  automatic = false;
  if (!stopAll()) { channelError(); return; }
  if (!findMotor(0)) { Serial.println("ERR NOT_READY No motors available."); return; }
  targetPulses = pulses;
  completedPulses = 0;
  motorOn = false;
  running = true;
  phaseStarted = millis() - OFF_MS;
  Serial.printf("START strength=%u pulses=%u order=UP,RIGHT,DOWN,LEFT\n", strength, pulses);
}

// Strict unsigned decimal parser; reject negatives, overflow and trailing text.
bool readNumber(const char *&cursor, uint32_t maximum, char delimiter, uint32_t &value) {
  if (*cursor < '0' || *cursor > '9') return false;
  value = 0;
  while (*cursor >= '0' && *cursor <= '9') {
    uint8_t digit = uint8_t(*cursor++ - '0');
    if (digit > maximum || value > (maximum - digit) / 10) return false;
    value = value * 10 + digit;
  }
  if (*cursor != delimiter) return false;
  if (delimiter) cursor++;
  return true;
}

bool applyFrame(const char *command) {
  const char *cursor = command + 3;
  uint32_t sequence, values[4];
  if (!readNumber(cursor, UINT32_MAX, ',', sequence) || sequence == 0) return false;
  for (uint8_t i = 0; i < 4; i++) {
    if (!readNumber(cursor, 1, i == 3 ? '\0' : ',', values[i])) return false;
  }
  if ((values[0] && values[2]) || (values[1] && values[3])) return false;
  running = false;
  automatic = false;
  for (uint8_t i = 0; i < 4; i++) {
    if (values[i] && !ready[i]) {
      stopTest();
      Serial.println("ERR NOT_READY Requested module is unavailable.");
      return true;
    }
  }
  // Turn off old directions before enabling the new ones.
  for (uint8_t i = 0; i < 4; i++) {
    if (ready[i] && values[i] == 0 && !writeLevel(i, 0)) { channelError(); return true; }
  }
  for (uint8_t i = 0; i < 4; i++) {
    if (ready[i] && values[i] && !writeLevel(i, strength)) { channelError(); return true; }
  }
  lastFrameAt = millis();
  automatic = true;
  Serial.printf("ACK %lu\n", (unsigned long)sequence);
  return true;
}

void processCommand(const char *command) {
  if (!command[0]) return;
  if (strcmp(command, "PING") == 0) { printReady(); return; }
  if (strcmp(command, "0") == 0) { stopTest(); Serial.println("STOP"); return; }
  if (strncmp(command, "D1,", 3) == 0) {
    if (!applyFrame(command)) { stopTest(); Serial.println("ERR COMMAND Invalid D1 frame."); }
    return;
  }
  // Preserve the original manual test commands. Close the Mac app first.
  if (strcmp(command, "1") == 0 || strcmp(command, "2") == 0 || strcmp(command, "3") == 0) {
    strength = command[0] == '1' ? 40 : command[0] == '2' ? 70 : 100;
    if (running && motorOn && !writeLevel(currentMotor, strength)) { channelError(); return; }
    Serial.printf("Strength set to %u\n", strength);
  } else if (strcmp(command, "one") == 0) {
    startTest(1);
  } else if (strcmp(command, "two") == 0) {
    startTest(3);
  } else if (strcmp(command, "three") == 0) {
    startTest(5);
  } else {
    stopTest();
    Serial.println("ERR COMMAND Use PING / D1,seq,up,right,down,left / 1,2,3 / one,two,three / 0.");
  }
}

void handleInput() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (!inputOverflow) {
        inputLine[inputLength] = '\0';
        processCommand(inputLine);
      }
      inputLength = 0;
      inputOverflow = false;
    } else if (!inputOverflow) {
      if (inputLength < sizeof(inputLine) - 1 && c >= 32 && c <= 126) {
        inputLine[inputLength++] = c;
      } else {
        inputLength = 0;
        inputOverflow = true;
        stopTest();
        Serial.println("ERR COMMAND Invalid or oversized line.");
      }
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Wire.begin(21, 22);
  Wire.setClock(100000);
  Wire.setTimeOut(50);
  if (!deviceFound(TCA_ADDR)) {
    Serial.println("ERR TCA9548A_NOT_FOUND address=0x70");
    return;
  }
  Wire.beginTransmission(TCA_ADDR);
  Wire.write(uint8_t(0));
  Wire.endTransmission();
  for (uint8_t i = 0; i < 4; i++) {
    if (!selectChannel(channels[i]) || !deviceFound(DRV_ADDR) || !drv[i].begin(&Wire)) continue;
    drv[i].useERM();  // Same motor type as the supplied sketch. Use useLRA() for LRA hardware.
    drv[i].setRealtimeValue(0);
    drv[i].setMode(DRV2605_MODE_REALTIME);
    ready[i] = true;
    if (!writeLevel(i, 0)) ready[i] = false;
    Serial.printf("INFO motor=%u channel=%u ready=%u\n", i + 1, channels[i], ready[i]);
  }
  stopTest();
  printReady();
}

void loop() {
  handleInput();
  uint32_t now = millis();
  if (automatic) {
    // PING/partial/invalid lines cannot keep an old direction alive.
    if (now - lastFrameAt >= FRAME_TIMEOUT_MS) {
      stopTest();
      Serial.println("TIMEOUT STOP");
    }
    return;
  }
  if (!running) return;
  if (motorOn) {
    if (now - phaseStarted < ON_MS) return;
    if (!writeLevel(currentMotor, 0)) { channelError(); return; }
    motorOn = false;
    completedPulses++;
    phaseStarted = now;
    if (completedPulses >= targetPulses) {
      if (!findMotor(currentMotor + 1)) { running = false; Serial.println("DONE"); return; }
      completedPulses = 0;
    }
    return;
  }
  if (now - phaseStarted < OFF_MS) return;
  if (!writeLevel(currentMotor, strength)) { channelError(); return; }
  motorOn = true;
  phaseStarted = now;
}
