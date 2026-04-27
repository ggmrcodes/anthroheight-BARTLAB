"""Pydantic record schemas — round-trip and validation."""
import json
import numpy as np
import pytest
from pydantic import ValidationError

from anthroheight.records import (
    PatientMetadata, CalibrationResult, Landmark, DeformityFlags,
    SurrogateMeasurement, HeightEstimate, Warning, ValidationReport,
    MeasurementRecord, Sex, SurrogateName, WarningLevel,
)


def test_patient_metadata_roundtrip():
    m = PatientMetadata(
        patient_id="P001", age_years=42, sex="M",
        ethnicity="white",
        operator_input_flags={"kyphosis": True, "scoliosis": False,
                              "lower_limb_contracture": False,
                              "upper_limb_contracture": False, "amputation": False},
        notes="test",
    )
    assert m.model_dump_json()
    restored = PatientMetadata.model_validate_json(m.model_dump_json())
    assert restored == m


def test_patient_metadata_rejects_negative_age():
    with pytest.raises(ValidationError):
        PatientMetadata(
            patient_id="P", age_years=-1, sex="F", ethnicity="white",
            operator_input_flags={}, notes="",
        )


def test_landmark_clamps_confidence_range():
    with pytest.raises(ValidationError):
        Landmark(name="nose", x_px=10.0, y_px=10.0, confidence=1.5)


def test_calibration_result_serializes_homography():
    H = np.eye(3).tolist()
    c = CalibrationResult(
        homography_matrix=H, mm_per_px_central=0.5,
        marker_corners_px=[[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
        reprojection_error_px=0.7,
    )
    assert json.loads(c.model_dump_json())["homography_matrix"] == H


def test_height_estimate_records_formula_and_see():
    e = HeightEstimate(
        height_cm=172.4, standard_error_cm=3.5,
        formula_id="chumlea_1985_white_male",
        formula_citation="Chumlea WC et al. 1985",
        validity_warnings=[],
    )
    assert e.standard_error_cm > 0


def test_warning_level_enum():
    w = Warning(level="warn", message="cross-check disagree", source="validate")
    assert w.level == "warn"
    with pytest.raises(ValidationError):
        Warning(level="not-a-level", message="x", source="x")


def test_measurement_record_full_roundtrip():
    pm = PatientMetadata(
        patient_id="P001", age_years=70, sex="F", ethnicity="white",
        operator_input_flags={}, notes="",
    )
    cal = CalibrationResult(
        homography_matrix=np.eye(3).tolist(), mm_per_px_central=0.5,
        marker_corners_px=[[0.0, 0.0]] * 4, reprojection_error_px=0.5,
    )
    sm = SurrogateMeasurement(
        surrogate_name="knee_height", segment_length_mm=510.0,
        landmarks_used=[("left_knee", 0.95), ("left_ankle", 0.93)],
        bilateral_mean_used=False, bilateral_asymmetry_mm=None,
    )
    he = HeightEstimate(
        height_cm=160.2, standard_error_cm=3.5,
        formula_id="chumlea_1985_white_female", formula_citation="Chumlea 1985",
        validity_warnings=[],
    )
    rec = MeasurementRecord(
        patient_metadata=pm, capture_timestamp="2026-04-27T12:00:00Z",
        image_path="data/measurements/P001/x.png",
        calibration=cal, landmarks=[],
        flags=DeformityFlags(
            kyphosis_suspected=False, kyphosis_confidence=0.0,
            scoliosis_suspected=False, scoliosis_confidence=0.0,
            lower_limb_contracture_suspected=False,
            lower_limb_contracture_confidence=0.0,
            upper_limb_contracture_suspected=False,
            upper_limb_contracture_confidence=0.0,
        ),
        surrogate=sm, estimate=he,
        validation=ValidationReport(warnings=[]),
        sum_of_segments_mm=None, operator_id="op1",
    )
    rec_json = rec.model_dump_json()
    restored = MeasurementRecord.model_validate_json(rec_json)
    assert restored.estimate.height_cm == 160.2
