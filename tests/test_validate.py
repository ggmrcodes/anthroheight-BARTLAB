"""Cross-checks, plausibility, symmetry, confidence — produces warnings only."""
import pytest
from anthroheight.records import (
    HeightEstimate, SurrogateMeasurement, CalibrationResult,
)
from anthroheight import validate


def _est(h_cm: float) -> HeightEstimate:
    return HeightEstimate(
        height_cm=h_cm, standard_error_cm=3.5,
        formula_id="test", formula_citation="test", validity_warnings=[],
    )


def _sm(name: str = "knee_height", landmark_confs: tuple[float, ...] = (0.95, 0.95),
        asymmetry: float | None = None) -> SurrogateMeasurement:
    return SurrogateMeasurement(
        surrogate_name=name, segment_length_mm=500.0,
        landmarks_used=[("a", c) for c in landmark_confs],
        bilateral_mean_used=asymmetry is not None,
        bilateral_asymmetry_mm=asymmetry,
    )


def _cal(rep_err: float = 0.5) -> CalibrationResult:
    import numpy as np
    return CalibrationResult(
        homography_matrix=np.eye(3).tolist(), mm_per_px_central=0.5,
        marker_corners_px=[[0.0, 0.0]] * 4, reprojection_error_px=rep_err,
    )


def test_clean_measurement_no_warnings():
    rep = validate.run(_est(170.0), _sm(), _cal(), other_estimates=[])
    assert rep.warnings == []


def test_height_below_plausibility_warns():
    rep = validate.run(_est(80.0), _sm(), _cal(), other_estimates=[])
    assert any(w.level == "warn" and "plausib" in w.message.lower() for w in rep.warnings)


def test_height_above_plausibility_warns():
    rep = validate.run(_est(250.0), _sm(), _cal(), other_estimates=[])
    assert any("plausib" in w.message.lower() for w in rep.warnings)


def test_cross_check_disagreement_warns():
    primary = _est(170.0)
    other = _est(180.0)   # 10 cm disagreement, threshold is 5 cm
    rep = validate.run(primary, _sm(), _cal(), other_estimates=[other])
    assert any("cross-check" in w.message.lower() or "disagree" in w.message.lower()
               for w in rep.warnings)


def test_bilateral_asymmetry_warns():
    rep = validate.run(_est(170.0), _sm(asymmetry=30.0), _cal(), other_estimates=[])
    assert any("symmetr" in w.message.lower() or "asymmet" in w.message.lower()
               for w in rep.warnings)


def test_low_pose_confidence_warns():
    rep = validate.run(_est(170.0), _sm(landmark_confs=(0.6, 0.5)), _cal(),
                       other_estimates=[])
    assert any("confidence" in w.message.lower() for w in rep.warnings)


def test_high_reprojection_error_warns():
    rep = validate.run(_est(170.0), _sm(), _cal(rep_err=3.5), other_estimates=[])
    assert any("calibration" in w.message.lower() or "reprojection" in w.message.lower()
               for w in rep.warnings)
