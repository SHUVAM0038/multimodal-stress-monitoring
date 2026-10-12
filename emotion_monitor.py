"""
Multimodal real-time depression / stress monitor.

Webcam -> Haar-cascade face detection -> CNN emotion (FER-2013, 7 classes)
ESP32  -> BP / HR / SpO2 over serial
Per frame : emotion + vitals -> fused state (shown on screen)
Per batch : 100 face frames -> valence score -> final emotion -> Excel log
Alert     : 10 consecutive "Highly Stressed" batches

Usage:
    python emotion_monitor.py --port COM8
    python emotion_monitor.py --no-serial        # camera only, for testing

Press q to quit. RESEARCH PROTOTYPE ONLY - not a diagnostic tool.
"""

import argparse
import os
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from keras.models import load_model

from fusion import Batch, ConsecutiveAlert, Vitals, determine_state, parse_esp32_line

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
INPUT_SIZE = 64                     # model expects 64x64 grayscale


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="COM8", help="ESP32 serial port (e.g. COM8, /dev/ttyUSB0)")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--no-serial", action="store_true", help="run without the ESP32")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--model", default="../models/emotion_model.hdf5")
    p.add_argument("--batch", type=int, default=100, help="face frames per batch")
    p.add_argument("--alert-after", type=int, default=10,
                   help="consecutive 'Highly Stressed' batches before an alert")
    p.add_argument("--log-dir", default="../logs")
    return p.parse_args()


def open_serial(args):
    if args.no_serial:
        return None
    import serial
    try:
        ser = serial.Serial(args.port, args.baud, timeout=0)
        time.sleep(2)                        # ESP32 resets when the port opens
        return ser
    except serial.SerialException as e:
        print(f"[warn] Could not open {args.port} ({e}). Continuing without vitals.")
        return None


def read_vitals(ser, current: Vitals) -> Vitals:
    """Drain all waiting serial lines and keep the most recent valid one."""
    if ser is None:
        return current
    while ser.in_waiting:
        line = ser.readline().decode(errors="ignore").strip()
        parsed = parse_esp32_line(line)
        if parsed is not None:
            current = parsed
    return current


def classify_face(model, gray, box):
    x, y, w, h = box
    face = cv2.resize(gray[y:y + h, x:x + w], (INPUT_SIZE, INPUT_SIZE))
    face = (face / 255.0).reshape(1, INPUT_SIZE, INPUT_SIZE, 1)
    return EMOTIONS[int(np.argmax(model.predict(face, verbose=0)))]


def fmt(value, unit=""):
    return f"{value}{unit}" if value > 0 else "--"


def main():
    args = parse_args()
    model = load_model(args.model, compile=False)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                    "haarcascade_frontalface_default.xml")
    ser = open_serial(args)
    cap = cv2.VideoCapture(args.camera)

    batch = Batch(size=args.batch)
    alert = ConsecutiveAlert(threshold=args.alert_after)

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"session_{datetime.now():%Y%m%d_%H%M%S}.xlsx")
    results = []
    print(f"[info] Logging batches to {log_path}")

    vitals = Vitals()
    final_emotion, alert_on = "Collecting...", False

    try:
        while True:
            vitals = read_vitals(ser, vitals)

            ok, frame = cap.read()
            if not ok:
                print("[error] Camera frame not available")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.3, 5)

            emotion, state = "No Face", "--"
            if len(faces):
                # largest face = the person being monitored
                box = max(faces, key=lambda f: f[2] * f[3])
                emotion = classify_face(model, gray, box)
                state = determine_state(emotion, vitals)
                batch.add(emotion, state, vitals)      # only frames with a face count

                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # ---- end of batch: score, log, check alert ----
            if batch.full():
                final_emotion, score = batch.final_emotion()
                alert_on = alert.update(final_emotion)
                mv = batch.mean_vitals()
                results.append({
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Final Emotion": final_emotion,
                    "Score": score,
                    "Dominant State": batch.dominant_state(),
                    "BP": mv.bp_text,
                    "HR": mv.hr,
                    "SpO2": mv.spo2,
                    "Alert": alert_on,
                })
                pd.DataFrame(results).to_excel(log_path, index=False)

                print(f"\n===== FINAL EMOTION ({args.batch} frames): {final_emotion} "
                      f"(score {score}, streak {alert.streak}/{alert.threshold}) =====")
                if alert_on:
                    print("CRITICAL ALERT: consistent depressive pattern detected")
                batch.clear()

            # ---- overlay ----
            line1 = (f"Emotion: {emotion} | BP:{vitals.bp_text} | HR:{fmt(vitals.hr)} | "
                     f"SpO2:{fmt(vitals.spo2, '%')} | State:{state}")
            line2 = (f"Last batch: {final_emotion} | Stressed streak: "
                     f"{alert.streak}/{alert.threshold}")
            cv2.putText(frame, line1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, line2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if alert_on:
                cv2.rectangle(frame, (0, frame.shape[0] - 40),
                              (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
                cv2.putText(frame, "ALERT: persistent depressive pattern",
                            (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)

            cv2.imshow("AI Emotion + Physiological Monitoring", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if ser is not None:
            ser.close()


if __name__ == "__main__":
    main()
