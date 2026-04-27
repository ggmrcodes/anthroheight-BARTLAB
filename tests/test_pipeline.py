"""End-to-end orchestration: image+metadata → MeasurementRecord."""
from datetime import datetime, timezone
import numpy as np
import pytest
from unittest.mock import patch

from tests.fixtures.synthetic_aruco import make_image, BED_CORNERS_MM
from anthroheight.records import (
    Landmark, PatientMetadata, CalibrationResult,
)
from anthroheight import pipeline


def _identity_landmarks() -> list[Landmark]:
    """Knee 500mm above ankle, both visible — produces ~50cm knee-height."""
    return [
        Landmark(name="nose", x_px=600, y_px=100, confidence=0.95),
        Landmark(name="left_shoulder", x_px=550, y_px=300, confidence=0.92),
        Landmark(name="right_shoulder", x_px=650, y_px=300, confidence=0.92),
        Landmark(name="left_hip", x_px=550, y_px=900, confidence=0.90),
        Landmark(name="right_hip", x_px=650, y_px=900, confidence=0.90),
        Landmark(name="left_knee", x_px=550, y_px=1100, confidence=0.95),
        Landmark(name="left_ankle", x_px=550, y_px=1600, confidence=0.93),
        Landmark(name="right_knee", x_px=650, y_px=1100, confidence=0.95),
        Landmark(name="right_ankle", x_px=650, y_px=1600, confidence=0.93),
        Landmark(name="left_elbow", x_px=550, y_px=600, confidence=0.85),
        Landmark(name="left_wrist", x_px=550, y_px=750, confidence=0.80),
        Landmark(name="right_elbow", x_px=650, y_px=600, confidence=0.85),
        Landmark(name="right_wrist", x_px=650, y_px=750, confidence=0.80),
    ]


def test_pipeline_end_to_end_knee_height(tmp_path):
    img, _ = make_image()
    pm = PatientMetadata(
        patient_id="P001", age_years=70, sex="F", ethnicity="white",
        operator_input_flags={}, notes="",
    )
    with patch("anthroheight.pipeline.PoseDetector") as pose_cls:
        mock_pose = pose_cls.return_value
        mock_pose.__enter__.return_value = mock_pose
        mock_pose.__exit__.return_value = None
        mock_pose.detect.return_value = _identity_landmarks()
        rec = pipeline.run(
            image_bgr=img,
            patient=pm,
            chosen_surrogate="knee_height",
            side_preference="both",
            operator_id="op1",
            expected_marker_layout=BED_CORNERS_MM,
            timestamp_iso="2026-04-27T12:00:00Z",
        )
    assert rec.surrogate.surrogate_name == "knee_height"
    assert rec.estimate.formula_id.startswith("chumlea_1985")
    assert 140.0 <= rec.estimate.height_cm <= 200.0
    assert rec.surrogate.bilateral_mean_used


def test_pipeline_records_validation_warnings_for_low_confidence(tmp_path):
    img, _ = make_image()
    pm = PatientMetadata(
        patient_id="P002", age_years=30, sex="M", ethnicity="white",
        operator_input_flags={}, notes="",
    )
    weak = _identity_landmarks()
    weak = [
        lm.model_copy(update={"confidence": 0.55}) if lm.name in ("left_knee", "left_ankle")
        else lm
        for lm in weak
    ]
    with patch("anthroheight.pipeline.PoseDetector") as pose_cls:
        mock_pose = pose_cls.return_value
        mock_pose.__enter__.return_value = mock_pose
        mock_pose.__exit__.return_value = None
        mock_pose.detect.return_value = weak
        rec = pipeline.run(
            image_bgr=img,
            patient=pm,
            chosen_surrogate="knee_height",
            side_preference="left",
            operator_id="op1",
            expected_marker_layout=BED_CORNERS_MM,
            timestamp_iso="2026-04-27T12:00:00Z",
        )
    assert any("confidence" in w.message.lower() for w in rec.validation.warnings)
    # Plus age outside Chumlea range -> validity_warning carried into estimate
    assert any("age" in w.lower() for w in rec.estimate.validity_warnings)
