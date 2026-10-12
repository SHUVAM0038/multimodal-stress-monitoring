"""
Fusion, batch scoring and alerting logic.

Kept free of camera / serial / model code so it can be unit-tested on its own
(see tests/test_fusion.py).

Per camera frame:
    emotion (CNN) + vitals (ESP32) --determine_state()--> fused state (on screen)
Per batch of BATCH_SIZE face frames:
    emotion valence score --batch_category()--> final emotion for the batch
    ConsecutiveAlert -> alert after N consecutive "Highly Stressed" batches
"""

from collections import Counter
from dataclasses import dataclass

# ---------------- Fusion thresholds ----------------
HR_HIGH = 90          # bpm
HR_LOW = 60           # bpm
SYS_HIGH = 130        # mmHg
DIA_HIGH = 85         # mmHg
SPO2_LOW = 90         # %

# ---------------- Per-frame fused states ----------------
NORMAL = "Normal"
STRESS = "Stress"
LOW_MOOD = "Low Mood"
DEPRESSION = "Depression"
LOW_SPO2 = "Low SpO2"

# ---------------- Batch scoring ----------------
# Valence weight per emotion: positive emotions push the score up,
# negative ones push it down.
EMOTION_WEIGHTS = {
    "Happy": 2,
    "Neutral": 1,
    "Surprise": 0,
    "Disgust": -1,
    "Sad": -2,
    "Fear": -2,
    "Angry": -3,
}

OVERALL_HAPPY = "Overall Happy"
MOSTLY_POSITIVE = "Mostly Positive"
BATCH_NEUTRAL = "Neutral"
HIGHLY_STRESSED = "Highly Stressed"


@dataclass
class Vitals:
    systolic: int = 0
    diastolic: int = 0
    hr: int = 0
    spo2: int = 0              # 0 = no finger / no valid reading

    def has_bp(self) -> bool:
        return self.systolic > 0 and self.diastolic > 0

    def has_hr(self) -> bool:
        return self.hr > 0

    def has_spo2(self) -> bool:
        return self.spo2 > 0

    @property
    def bp_text(self) -> str:
        return f"{self.systolic}/{self.diastolic}" if self.has_bp() else "--"


def determine_state(emotion: str, v: Vitals) -> str:
    """Rule-based fusion of one facial-emotion label with the latest vitals.

    A missing reading (0) is never treated as abnormal.
    """
    if v.has_spo2() and v.spo2 < SPO2_LOW:
        return LOW_SPO2

    high_hr = v.has_hr() and v.hr > HR_HIGH
    low_hr = v.has_hr() and v.hr < HR_LOW
    high_bp = v.has_bp() and (v.systolic > SYS_HIGH or v.diastolic > DIA_HIGH)

    # A positive or angry face with a racing heart or raised BP: the body
    # overrides the face.
    if emotion in ("Happy", "Angry") and (high_hr or high_bp):
        return STRESS

    if emotion == "Sad" and low_hr:
        return LOW_MOOD

    if emotion == "Sad" and high_hr and high_bp:
        return DEPRESSION

    return NORMAL


def batch_score(emotions) -> int:
    return sum(EMOTION_WEIGHTS.get(e, 0) for e in emotions)


def batch_category(score: int, n_frames: int = 100) -> str:
    """Map a batch valence score to the batch's final emotion.

    The cut-off is +/-0.5 per frame, i.e. +/-50 for the standard 100-frame
    batch, so the categories mean the same thing at any batch size.
    """
    cutoff = 0.5 * n_frames
    if score > cutoff:
        return OVERALL_HAPPY
    if score > 0:
        return MOSTLY_POSITIVE
    if score < -cutoff:
        return HIGHLY_STRESSED
    return BATCH_NEUTRAL


class Batch:
    """Collects face frames until the batch is full, then summarises them."""

    def __init__(self, size: int = 100):
        self.size = size
        self.emotions: list[str] = []
        self.states: list[str] = []
        self.vitals: list[Vitals] = []

    def add(self, emotion: str, state: str, vitals: Vitals) -> None:
        self.emotions.append(emotion)
        self.states.append(state)
        self.vitals.append(vitals)

    def full(self) -> bool:
        return len(self.emotions) >= self.size

    def final_emotion(self):
        score = batch_score(self.emotions)
        return batch_category(score, len(self.emotions)), score

    def dominant_state(self) -> str:
        return Counter(self.states).most_common(1)[0][0]

    def mean_vitals(self) -> Vitals:
        """Mean of each vital over the batch, ignoring missing (0) readings."""
        def mean(values):
            vals = [x for x in values if x > 0]
            return round(sum(vals) / len(vals)) if vals else 0
        return Vitals(
            systolic=mean(v.systolic for v in self.vitals),
            diastolic=mean(v.diastolic for v in self.vitals),
            hr=mean(v.hr for v in self.vitals),
            spo2=mean(v.spo2 for v in self.vitals),
        )

    def clear(self) -> None:
        self.emotions.clear()
        self.states.clear()
        self.vitals.clear()


class ConsecutiveAlert:
    """Alert once `threshold` consecutive batches equal `trigger`."""

    def __init__(self, threshold: int = 10, trigger: str = HIGHLY_STRESSED):
        self.threshold = threshold
        self.trigger = trigger
        self.streak = 0

    def update(self, batch_result: str) -> bool:
        self.streak = self.streak + 1 if batch_result == self.trigger else 0
        return self.streak >= self.threshold


def parse_esp32_line(line: str):
    """Parse 'BP:118/72 HR:81 SpO2:97' into Vitals. Returns None if malformed."""
    if not line.startswith("BP:"):
        return None
    try:
        fields = dict(part.split(":", 1) for part in line.split())
        sys_, dia = (int(x) for x in fields["BP"].split("/"))
        return Vitals(systolic=sys_, diastolic=dia,
                      hr=int(fields["HR"]), spo2=int(fields["SpO2"]))
    except (ValueError, KeyError):
        return None
