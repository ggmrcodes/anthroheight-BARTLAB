"""Surrogate-measurement segment length math."""
import numpy as np
import pytest
from anthroheight.records import Landmark, CalibrationResult
from anthroheight import measure


def _identity_calibration(mm_per_px: float = 1.0) -> CalibrationResult:
    """Identity homography — pixel == mm. Useful for unit math."""
    return CalibrationResult(
        homography_matrix=(np.eye(3) * np.array([[mm_per_px, mm_per_px, 1.0]]).T).tolist(),
        mm_per_px_central=mm_per_px,
        marker_corners_px=[[0.0, 0.0]] * 4,
        reprojection_error_px=0.5,
    )


def _lm(name: str, x: float, y: float, c: float = 0.95) -> Landmark:
    return Landmark(name=name, x_px=x, y_px=y, confidence=c)


def test_knee_height_unilateral_left():
    cal = _identity_calibration(mm_per_px=1.0)
    landmarks = [
        _lm("left_knee", 100.0, 100.0),
        _lm("left_ankle", 100.0, 600.0),   # 500 mm vertical
        _lm("right_knee", 200.0, 100.0, c=0.2),    # too low confidence to use
        _lm("right_ankle", 200.0, 600.0, c=0.2),
    ]
    sm = measure.measure_surrogate("knee_height", landmarks, cal, side_preference="left")
    assert abs(sm.segment_length_mm - 500.0) < 0.5
    assert not sm.bilateral_mean_used


def test_knee_height_bilateral_mean_when_both_visible():
    cal = _identity_calibration(mm_per_px=1.0)
    landmarks = [
        _lm("left_knee", 100.0, 100.0),
        _lm("left_ankle", 100.0, 600.0),    # 500 mm
        _lm("right_knee", 200.0, 100.0),
        _lm("right_ankle", 200.0, 700.0),   # 600 mm
    ]
    sm = measure.measure_surrogate("knee_height", landmarks, cal, side_preference="both")
    assert abs(sm.segment_length_mm - 550.0) < 0.5
    assert sm.bilateral_mean_used
    assert sm.bilateral_asymmetry_mm == pytest.approx(100.0, abs=0.5)


def test_ulna_length():
    cal = _identity_calibration()
    landmarks = [
        _lm("left_elbow", 100.0, 100.0),
        _lm("left_wrist", 100.0, 350.0),    # 250 mm
    ]
    sm = measure.measure_surrogate("ulna", landmarks, cal, side_preference="left")
    assert abs(sm.segment_length_mm - 250.0) < 0.5


def test_demispan_uses_shoulder_midpoint():
    cal = _identity_calibration()
    landmarks = [
        _lm("left_shoulder", 200.0, 200.0),
        _lm("right_shoulder", 300.0, 200.0),   # midpoint = (250, 200)
        _lm("middle_fingertip_left", 1050.0, 200.0),    # 800mm from midpoint
    ]
    sm = measure.measure_surrogate("demispan", landmarks, cal, side_preference="left")
    assert abs(sm.segment_length_mm - 800.0) < 0.5


def test_missing_required_landmark_raises():
    cal = _identity_calibration()
    landmarks = [_lm("left_knee", 0.0, 0.0)]   # ankle missing
    with pytest.raises(measure.MissingLandmarkError):
        measure.measure_surrogate("knee_height", landmarks, cal, side_preference="left")


def test_homography_scaling_applied():
    cal = _identity_calibration(mm_per_px=2.0)
    H = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    cal = CalibrationResult(
        homography_matrix=H.tolist(), mm_per_px_central=2.0,
        marker_corners_px=[[0.0, 0.0]] * 4, reprojection_error_px=0.5,
    )
    landmarks = [
        _lm("left_knee", 100.0, 100.0),
        _lm("left_ankle", 100.0, 350.0),    # 250 px diff
    ]
    sm = measure.measure_surrogate("knee_height", landmarks, cal, side_preference="left")
    # 250 px * 2 mm/px = 500 mm
    assert abs(sm.segment_length_mm - 500.0) < 0.5
