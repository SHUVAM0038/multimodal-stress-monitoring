/*
 * Multimodal Stress Monitoring - ESP32 sensor node
 * -------------------------------------------------
 * Reads blood pressure + pulse from a serial BP module and computes SpO2
 * from a MAX30102 (red + IR PPG) using Maxim's reference algorithm, then
 * streams one line per second to the Python app over USB serial:
 *
 *     BP:<sys>/<dia> HR:<pulse> SpO2:<spo2>
 *
 * SpO2 is printed as 0 when there is no finger on the sensor or the
 * algorithm has no valid estimate yet - the Python side treats 0 as
 * "no reading", not as low oxygen.
 *
 * Wiring (ESP32):
 *   BP module TX/RX -> GPIO16 (RX2) / GPIO17 (TX2), 9600 baud
 *   MAX30102        -> I2C SDA GPIO21, SCL GPIO22
 *
 * RESEARCH PROTOTYPE ONLY - not a medical device.
 */

#include <Wire.h>
#include "MAX30105.h"          // SparkFun MAX3010x library (works with MAX30102)
#include "spo2_algorithm.h"

#define BP_RX 16
#define BP_TX 17

const long FINGER_IR_THRESHOLD = 50000;   // IR level that means a finger is present
const unsigned long PRINT_MS   = 1000;

MAX30105 sensor;

// 100 samples at the sensor's 25 Hz output (100 Hz / 4x averaging) = 4 s window,
// matching the sample rate the Maxim algorithm expects.
#define BUFFER_SIZE 100
uint32_t irBuffer[BUFFER_SIZE];
uint32_t redBuffer[BUFFER_SIZE];
int bufferIndex = 0;

int32_t spo2 = 0;
int8_t  validSpO2 = 0;
int32_t ppgHR = 0;          // HR estimate from PPG (not used; BP module pulse is used)
int8_t  validPpgHR = 0;
bool    fingerPresent = false;

String bpLine = "";
int systolic = 0, diastolic = 0, pulse = 0;

unsigned long lastPrint = 0;

// ---------------- BP module (UART) ----------------
// Line format: "success,SYS,DIA,PULSE" or "SYS,DIA,PULSE"
void readBP()
{
  while (Serial2.available()) {
    char c = Serial2.read();

    if (c != '\n' && c != '\r') { bpLine += c; continue; }
    if (bpLine.length() <= 5)    { bpLine = ""; continue; }

    String data = bpLine;
    bpLine = "";
    if (data.startsWith("success")) data = data.substring(data.indexOf(',') + 1);

    int first  = data.indexOf(',');
    int second = data.indexOf(',', first + 1);
    if (first <= 0 || second <= first) continue;   // malformed line

    systolic  = data.substring(0, first).toInt();
    diastolic = data.substring(first + 1, second).toInt();
    pulse     = data.substring(second + 1).toInt();
  }
}

// ---------------- MAX30102 -> SpO2 ----------------
// Reads only NEW samples from the sensor FIFO so the algorithm sees the
// true sample rate, and recomputes SpO2 every time the buffer fills.
void readMAX()
{
  sensor.check();
  while (sensor.available()) {
    uint32_t red = sensor.getFIFORed();
    uint32_t ir  = sensor.getFIFOIR();
    sensor.nextSample();

    fingerPresent = ir > FINGER_IR_THRESHOLD;
    if (!fingerPresent) {            // finger removed: discard partial window
      bufferIndex = 0;
      validSpO2 = 0;
      continue;
    }

    redBuffer[bufferIndex] = red;
    irBuffer[bufferIndex]  = ir;

    if (++bufferIndex >= BUFFER_SIZE) {
      bufferIndex = 0;
      maxim_heart_rate_and_oxygen_saturation(irBuffer, BUFFER_SIZE, redBuffer,
                                             &spo2, &validSpO2, &ppgHR, &validPpgHR);
    }
  }
}

void setup()
{
  Serial.begin(115200);                          // to Python
  Serial2.begin(9600, SERIAL_8N1, BP_RX, BP_TX); // BP module
  Wire.begin(21, 22);

  if (!sensor.begin(Wire)) {
    Serial.println("MAX30102 not found!");
    while (true) delay(1000);
  }
  // brightness, sampleAvg, ledMode(2 = red+IR), sampleRate, pulseWidth, adcRange
  sensor.setup(60, 4, 2, 100, 411, 4096);
}

void loop()
{
  readBP();
  readMAX();

  if (millis() - lastPrint >= PRINT_MS) {      // non-blocking: no delay()
    lastPrint = millis();
    int shownSpO2 = (fingerPresent && validSpO2) ? spo2 : 0;

    Serial.print("BP:");    Serial.print(systolic);
    Serial.print("/");      Serial.print(diastolic);
    Serial.print(" HR:");   Serial.print(pulse);
    Serial.print(" SpO2:"); Serial.println(shownSpO2);
  }
}
