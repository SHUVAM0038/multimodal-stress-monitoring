import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from fusion import (BATCH_NEUTRAL, DEPRESSION, HIGHLY_STRESSED, LOW_MOOD,
                    LOW_SPO2, MOSTLY_POSITIVE, NORMAL, OVERALL_HAPPY, STRESS,
                    Batch, ConsecutiveAlert, Vitals, batch_category,
                    batch_score, determine_state, parse_esp32_line)

CALM = Vitals(118, 76, 72, 97)
RACING = Vitals(123, 78, 95, 96)        # values from the "Happy -> Stress" screenshot
HYPERTENSIVE_RACING = Vitals(138, 88, 98, 97)


# ---------------- per-frame fusion ----------------
def test_happy_face_with_racing_heart_is_stress():
    assert determine_state("Happy", RACING) == STRESS


def test_happy_face_with_calm_vitals_is_normal():
    assert determine_state("Happy", CALM) == NORMAL


def test_angry_face_with_raised_bp_is_stress():
    assert determine_state("Angry", Vitals(135, 80, 75, 97)) == STRESS


def test_neutral_face_is_not_overridden():
    assert determine_state("Neutral", RACING) == NORMAL


def test_sad_with_low_heart_rate_is_low_mood():
    assert determine_state("Sad", Vitals(110, 70, 55, 97)) == LOW_MOOD


def test_sad_with_high_hr_and_high_bp_is_depression():
    assert determine_state("Sad", HYPERTENSIVE_RACING) == DEPRESSION


def test_sad_with_high_hr_but_normal_bp_is_normal():
    assert determine_state("Sad", RACING) == NORMAL


def test_real_low_spo2_overrides_everything():
    assert determine_state("Happy", Vitals(118, 76, 72, 86)) == LOW_SPO2


# Bug fixes vs. the original script
def test_spo2_zero_means_no_reading_not_low_oxygen():
    assert determine_state("Sad", Vitals(123, 78, 95, 0)) == NORMAL


def test_missing_heart_rate_is_not_low_mood():
    assert determine_state("Sad", Vitals(0, 0, 0, 97)) == NORMAL


def test_missing_vitals_never_count_as_stress():
    assert determine_state("Happy", Vitals()) == NORMAL


# ---------------- batch scoring ----------------
def test_batch_categories_match_thresholds():
    assert batch_category(51) == OVERALL_HAPPY
    assert batch_category(50) == MOSTLY_POSITIVE
    assert batch_category(1) == MOSTLY_POSITIVE
    assert batch_category(0) == BATCH_NEUTRAL
    assert batch_category(-50) == BATCH_NEUTRAL
    assert batch_category(-51) == HIGHLY_STRESSED


def test_thresholds_scale_with_batch_size():
    assert batch_category(-20, n_frames=10) == HIGHLY_STRESSED   # 10 Sad frames
    assert batch_category(-20, n_frames=100) == BATCH_NEUTRAL


def test_mostly_sad_batch_is_highly_stressed():
    emotions = ["Sad"] * 60 + ["Neutral"] * 40          # -120 + 40 = -80
    assert batch_score(emotions) == -80
    assert batch_category(batch_score(emotions)) == HIGHLY_STRESSED


def test_mostly_happy_batch_is_overall_happy():
    emotions = ["Happy"] * 50 + ["Neutral"] * 50        # 100 + 50 = 150
    assert batch_category(batch_score(emotions)) == OVERALL_HAPPY


def test_batch_fills_and_summarises():
    b = Batch(size=3)
    for _ in range(3):
        b.add("Sad", LOW_MOOD, Vitals(110, 70, 55, 0))
    assert b.full()
    assert b.dominant_state() == LOW_MOOD
    assert b.mean_vitals().spo2 == 0                    # all missing -> 0, not averaged


def test_mean_vitals_ignores_missing_readings():
    b = Batch(size=2)
    b.add("Neutral", NORMAL, Vitals(120, 80, 70, 98))
    b.add("Neutral", NORMAL, Vitals(120, 80, 70, 0))
    assert b.mean_vitals().spo2 == 98


# ---------------- alerting ----------------
def test_alert_needs_consecutive_stressed_batches():
    a = ConsecutiveAlert(threshold=3)
    assert not a.update(HIGHLY_STRESSED)
    assert not a.update(HIGHLY_STRESSED)
    assert a.update(HIGHLY_STRESSED)


def test_streak_resets_on_other_batch():
    a = ConsecutiveAlert(threshold=3)
    a.update(HIGHLY_STRESSED)
    a.update(HIGHLY_STRESSED)
    a.update(BATCH_NEUTRAL)
    assert not a.update(HIGHLY_STRESSED)
    assert a.streak == 1


# ---------------- serial parsing ----------------
def test_parse_valid_line():
    assert parse_esp32_line("BP:118/72 HR:81 SpO2:97") == Vitals(118, 72, 81, 97)


def test_parse_rejects_garbage():
    assert parse_esp32_line("MAX30102 not found!") is None
    assert parse_esp32_line("BP:abc HR:81 SpO2:97") is None
