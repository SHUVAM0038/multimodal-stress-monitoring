# Multimodal Real-Time Depression & Stress Monitoring System

A low-cost system that combines **facial-emotion recognition** from an ordinary webcam with **physiological signals** (blood pressure, heart rate, SpO2) from an ESP32 sensor node, to support continuous monitoring of patients in **rehabilitation centres** and flag persistent depressive patterns to staff.

> ⚠️ **Research prototype only.** This is an academic project. It is not a diagnostic tool, has not been clinically validated, and must not be used to make decisions about anyone's mental health.

![Happy expression overridden to Stress by physiological data](docs/images/demo_happy_stress.png)

*The face reads as "Happy", but an elevated heart rate (95 bpm) makes the fused state **Stress**. The body overrides the face. This is the reason for combining modalities.*

---

## Why this project

In rehabilitation centres, depression is usually assessed through self-reports, interviews and periodic observation. These are subjective, infrequent and easy to mask. Each automated approach has a blind spot when used alone:

- **Facial expressions** can be masked, and are thrown off by lighting, head pose and occlusion.
- **Physiological signals** are objective but can't tell *which* emotion is behind a change. Stress, excitement and exercise look alike.

Fusing the two lets each cover the other's weakness, and tracking the result over time separates a persistent pattern from a bad moment.

## How it works

![Block diagram](docs/images/block_diagram.png)

### 1. Two parallel data streams
- **Vision (`python/emotion_monitor.py`):** a Haar cascade finds the face in each webcam frame. The largest face is cropped, converted to grayscale, resized to 64×64 and passed to a pre-trained CNN that outputs one of 7 FER-2013 emotions (*Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral*).
- **Physiology (`firmware/sensor_node.ino`):** an ESP32 reads systolic/diastolic BP and pulse from a digital BP monitor over UART, and computes SpO2 from the MAX30102's red and IR PPG signals using Maxim's reference algorithm. Once per second it sends `BP:118/72 HR:81 SpO2:97` to the laptop over USB serial.

### 2. Per-frame fusion (`python/fusion.py`)

Each frame's emotion is combined with the latest vitals into a fused state, shown live on screen:

| Condition | Fused state |
|---|---|
| SpO2 measured and below 90 % | **Low SpO2** (overrides everything) |
| *Happy* or *Angry* face + HR > 90, or BP > 130/85 | **Stress**: the body overrides the face |
| *Sad* face + HR < 60 | **Low Mood** |
| *Sad* face + HR > 90 **and** BP > 130/85 | **Depression** indication |
| Anything else | **Normal** |

Missing readings are never treated as abnormal. An SpO2 or HR of `0` means *no reading*, not low oxygen or a low pulse.

### 3. Batch scoring over 100 frames
Single frames are noisy, so the decision that matters is made per **batch of 100 face frames**. Each emotion carries a valence weight:

| Happy | Neutral | Surprise | Disgust | Sad | Fear | Angry |
|---|---|---|---|---|---|---|
| +2 | +1 | 0 | −1 | −2 | −2 | −3 |

The batch score is the sum of these weights:

| Score | Final emotion for the batch |
|---|---|
| > +50 | Overall Happy |
| 1 to +50 | Mostly Positive |
| −50 to 0 | Neutral |
| < −50 | **Highly Stressed** |

The ±50 cut-offs are ±0.5 per frame, so they scale correctly if the batch size is changed.

### 4. Logging
Each batch is appended to an Excel log, `logs/session_<timestamp>.xlsx`. It records the time, final emotion, score, the most common fused state in the batch, mean BP/HR/SpO2 over the batch (ignoring missing readings) and the alert flag. This gives a trend record per session.

### 5. Alerting
If **10 consecutive batches** are *Highly Stressed*, a critical alert is shown on screen and logged. One non-stressed batch resets the streak, which keeps one-off bad moments from triggering false alarms.

## Demo

| Neutral | Sad | Angry |
|---|---|---|
| ![Neutral](docs/images/demo_neutral.png) | ![Sad](docs/images/demo_sad.png) | ![Angry](docs/images/demo_angry.png) |

> **Note on these screenshots:** they were taken before two fixes. The SpO2 shown was a **placeholder value** from the earlier firmware, not computed from the sensor. An SpO2 of 0 (finger off the sensor) was also wrongly shown as "Low SpO2", as in the Sad and Angry frames. The current firmware computes SpO2 from the PPG signal, and the fusion logic now treats 0 as "no reading".

## Designed for accessibility and usability

The system needs only a standard webcam, a low-cost ESP32 sensor node and an ordinary laptop. No clinical monitor or specialised camera is required, so it can be deployed wherever a computer is available, including smaller rehabilitation centres. The interface is a single window that shows the emotion, vitals, current state and alert status at a glance.

## Repository layout

```
├── firmware/sensor_node.ino     ESP32: BP over UART + SpO2 from MAX30102
├── python/
│   ├── emotion_monitor.py       camera + model + serial + display + logging
│   ├── fusion.py                fusion rules, batch scoring, alert logic
│   └── requirements.txt
├── models/                      place emotion_model.hdf5 here
├── tests/test_fusion.py         unit tests for the fusion / alert logic
└── docs/images/                 block diagram and demo screenshots
```

## Running it

**ESP32**
1. Arduino IDE with ESP32 board support and the **SparkFun MAX3010x** library.
2. Upload `firmware/sensor_node.ino`. Check the Serial Monitor at 115200 baud: SpO2 should read about 95–100 with a finger on the sensor and 0 without.
3. Close the Serial Monitor before starting the Python app, since only one program can use the port.

**Python**
```bash
pip install -r python/requirements.txt
cd python
python emotion_monitor.py --port COM8 --model ../models/emotion_model.hdf5
```
- Linux/Mac ports look like `/dev/ttyUSB0`.
- `--no-serial` runs camera-only for testing.
- `--batch` and `--alert-after` change the batch size and the number of batches before an alert.
- Press **q** to quit.

**Tests**
```bash
pytest tests
```

## Hardware

| Item | Purpose | ESP32 pins |
|---|---|---|
| ESP32 | Reads sensors, streams vitals | – |
| Digital BP monitor (serial output) | Systolic, diastolic, pulse | RX2 GPIO16 / TX2 GPIO17, 9600 baud |
| MAX30102 | Red + IR PPG → SpO2 | I²C SDA 21 / SCL 22 |
| Webcam | Face capture | USB to laptop |

## Limitations

- Fusion rules, emotion weights and thresholds are hand-set, not learned from labelled clinical data.
- The batch score and alert are driven by the facial emotions. The vitals shape the per-frame fused state and are logged with each batch, but don't yet change the batch decision.
- The emotion model was trained on general-population faces (FER-2013), not on patients, and accuracy drops with poor lighting, head turns and occlusion.
- Seven discrete emotions are a coarse proxy for a complex clinical condition. A *Low Mood* indication is not a diagnosis.
- BP is updated only when a cuff reading is taken, so it lags behind the camera.
- Tested by the project team only. No patient trials.

## Ethics & data handling

- The demo images show a project team member, used with consent.
- No patient data was collected. Session logs contain physiological data, so `logs/` is excluded from the repository. Any real deployment would need informed consent, ethics approval and secure storage.

## Future work

- Feed the fused state (not only the face) into the batch decision and alert.
- Learn the fusion rules from data, and adapt thresholds per person.
- Buzzer and Wi-Fi notifications from the ESP32, plus a clinician dashboard.
- Move the sensors into a wearable for continuous, unobtrusive monitoring.
- Add HRV from the PPG signal, and further modalities such as voice or EEG.

## Team

Built at Vellore Institute of Technology.

- **Shuvam Mitra**
- **Shivani Anandaraj**
- **Tarun Eashwar**
