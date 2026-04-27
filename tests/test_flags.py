"""CV-derived deformity flag computation from supine landmarks."""
import math
import pytest
from anthroheight.records import Landmark
from anthroheight import flags


def lm(name, x, y, c=0.95):
    return Landmark(name=name, x_px=x, y_px=y, confidence=c)


def test_no_deformities_clean_supine():
    """Patient lying straight: head, shoulders, hips collinear; knees straight."""
    landmarks = [
        lm("nose", 100, 50), lm("left_shoulder", 100, 200), lm("right_shoulder", 100, 220),
        lm("left_hip", 100, 600), lm("right_hip", 100, 620),
        lm("left_knee", 100, 900), lm("left_ankle", 100, 1200),
        lm("right_knee", 100, 920), lm("right_ankle", 100, 1220),
        lm("left_elbow", 200, 350), lm("left_wrist", 200, 500),
    ]
    f = flags.compute_flags(landmarks)
    assert not f.kyphosis_suspected
    assert not f.scoliosis_suspected
    assert not f.lower_limb_contracture_suspected
    assert not f.upper_limb_contracture_suspected


def test_kyphosis_detected_when_head_offset_from_body_axis():
    """Kyphosis raises the head; in overhead view head landmark sits laterally
    offset from the shoulder-hip axis."""
    landmarks = [
        lm("nose", 250, 50),    # 150 px lateral to body axis at x=100
        lm("left_shoulder", 100, 200), lm("right_shoulder", 100, 220),
        lm("left_hip", 100, 600), lm("right_hip", 100, 620),
        lm("left_knee", 100, 900), lm("left_ankle", 100, 1200),
        lm("right_knee", 100, 920), lm("right_ankle", 100, 1220),
        lm("left_elbow", 200, 350), lm("left_wrist", 200, 500),
    ]
    f = flags.compute_flags(landmarks, head_offset_threshold_px=50.0)
    assert f.kyphosis_suspected


def test_scoliosis_detected_when_axis_breaks():
    """Spine landmarks not collinear (head ≠ shoulder midline ≠ hip midline)."""
    landmarks = [
        lm("nose", 100, 50),
        lm("left_shoulder", 100, 200), lm("right_shoulder", 200, 220),
        lm("left_hip", 200, 600), lm("right_hip", 100, 620),
        lm("left_knee", 100, 900), lm("left_ankle", 100, 1200),
        lm("right_knee", 100, 920), lm("right_ankle", 100, 1220),
        lm("left_elbow", 200, 350), lm("left_wrist", 200, 500),
    ]
    f = flags.compute_flags(landmarks, axis_deviation_threshold_px=30.0)
    assert f.scoliosis_suspected


def test_knee_contracture_flagged_when_joint_angle_bent():
    """Bent knee: hip-knee-ankle forms an angle materially less than 180°."""
    landmarks = [
        lm("nose", 100, 50), lm("left_shoulder", 100, 200), lm("right_shoulder", 100, 220),
        lm("left_hip", 100, 600), lm("right_hip", 100, 620),
        lm("left_knee", 100, 900),
        lm("left_ankle", 250, 1100),    # bent away from straight
        lm("right_knee", 100, 920), lm("right_ankle", 100, 1220),
        lm("left_elbow", 200, 350), lm("left_wrist", 200, 500),
    ]
    f = flags.compute_flags(landmarks, joint_angle_threshold_deg=30.0)
    assert f.lower_limb_contracture_suspected


def test_low_confidence_landmarks_ignored():
    """A landmark below 0.5 confidence should not drive a flag."""
    landmarks = [
        lm("nose", 250, 50, c=0.2),    # would suggest kyphosis but low conf
        lm("left_shoulder", 100, 200), lm("right_shoulder", 100, 220),
        lm("left_hip", 100, 600), lm("right_hip", 100, 620),
        lm("left_knee", 100, 900), lm("left_ankle", 100, 1200),
        lm("right_knee", 100, 920), lm("right_ankle", 100, 1220),
        lm("left_elbow", 200, 350), lm("left_wrist", 200, 500),
    ]
    f = flags.compute_flags(landmarks)
    assert not f.kyphosis_suspected
