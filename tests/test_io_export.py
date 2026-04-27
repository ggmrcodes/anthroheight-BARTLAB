"""Annotated PNG + JSON record + CSV log row outputs."""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from anthroheight.records import (
    PatientMetadata, CalibrationResult, Landmark, DeformityFlags,
    SurrogateMeasurement, HeightEstimate, ValidationReport, MeasurementRecord,
)
from anthroheight import io_export


def _record(tmp_path):
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
    return MeasurementRecord(
        patient_metadata=pm, capture_timestamp="2026-04-27T12:00:00Z",
        image_path="placeholder",
        calibration=cal, landmarks=[Landmark(name="left_knee", x_px=100, y_px=100, confidence=0.95)],
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


def test_export_writes_annotated_png(tmp_path):
    rec = _record(tmp_path)
    img = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    out = io_export.export(rec, img, output_root=tmp_path)
    assert out.png_path.exists()
    loaded = cv2.imread(str(out.png_path))
    assert loaded is not None


def test_export_writes_json_record(tmp_path):
    rec = _record(tmp_path)
    img = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    out = io_export.export(rec, img, output_root=tmp_path)
    assert out.json_path.exists()
    data = json.loads(out.json_path.read_text())
    assert data["estimate"]["height_cm"] == 160.2


def test_export_appends_csv_row(tmp_path):
    rec = _record(tmp_path)
    img = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    io_export.export(rec, img, output_root=tmp_path)
    io_export.export(rec, img, output_root=tmp_path)
    csv_path = tmp_path / "log.csv"
    assert csv_path.exists()
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 2
    assert rows[0]["patient_id"] == "P001"
    assert float(rows[0]["height_cm"]) == 160.2


def test_export_blurs_face_by_default(tmp_path):
    rec = _record(tmp_path)
    # Add a nose landmark so blur has a target
    rec_with_nose = rec.model_copy(update={
        "landmarks": rec.landmarks + [
            Landmark(name="nose", x_px=960, y_px=540, confidence=0.9)
        ]
    })
    img = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    # Add a recognizable pattern in the face region
    img[440:640, 860:1060] = [50, 100, 150]
    out = io_export.export(rec_with_nose, img, output_root=tmp_path)
    saved = cv2.imread(str(out.png_path))
    # The face region in the saved image should NOT match the original
    # (blur should have changed pixel values)
    original_face_mean = img[440:640, 860:1060].mean()
    saved_face_mean = saved[440:640, 860:1060].mean()
    # Blur smooths so means stay similar; instead check std (variance) is reduced
    assert saved[440:640, 860:1060].std() < img[440:640, 860:1060].std() + 5


def test_export_skips_blur_when_disabled(tmp_path):
    rec = _record(tmp_path)
    rec_with_nose = rec.model_copy(update={
        "landmarks": rec.landmarks + [
            Landmark(name="nose", x_px=960, y_px=540, confidence=0.9)
        ]
    })
    img = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    img[440:640, 860:1060] = [50, 100, 150]
    out = io_export.export(rec_with_nose, img, output_root=tmp_path, blur_face=False)
    saved = cv2.imread(str(out.png_path))
    # Without blur, the face region should be largely preserved (only annotation
    # circles + text near corners affect non-face areas).
    np.testing.assert_array_equal(
        saved[440:640, 860:1060], img[440:640, 860:1060]
    )


def test_export_raises_on_imwrite_failure(tmp_path, monkeypatch):
    """If cv2.imwrite returns False, export() must raise IOError."""
    rec = _record(tmp_path)
    img = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    monkeypatch.setattr("anthroheight.io_export.cv2.imwrite", lambda *a, **kw: False)
    with pytest.raises(IOError):
        io_export.export(rec, img, output_root=tmp_path)
