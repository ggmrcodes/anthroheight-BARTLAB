"""Plausibility / cross-check / symmetry / confidence sanity checks.

Produces a ValidationReport with warnings — never blocks. Operator decides
whether to save.
"""
from __future__ import annotations

from anthroheight.records import (
    CalibrationResult, HeightEstimate, SurrogateMeasurement,
    ValidationReport, Warning,
)


PLAUSIBLE_HEIGHT_MIN_CM = 100.0
PLAUSIBLE_HEIGHT_MAX_CM = 220.0
CROSS_CHECK_THRESHOLD_CM = 5.0
SYMMETRY_THRESHOLD_MM = 20.0
LANDMARK_CONFIDENCE_THRESHOLD = 0.7
REPROJECTION_ERROR_THRESHOLD_PX = 2.0


def run(
    estimate: HeightEstimate,
    surrogate: SurrogateMeasurement,
    calibration: CalibrationResult,
    other_estimates: list[HeightEstimate],
) -> ValidationReport:
    warnings: list[Warning] = []

    # Plausibility
    if estimate.height_cm < PLAUSIBLE_HEIGHT_MIN_CM \
       or estimate.height_cm > PLAUSIBLE_HEIGHT_MAX_CM:
        warnings.append(Warning(
            level="warn",
            message=(f"Height {estimate.height_cm:.1f} cm outside plausible range "
                     f"[{PLAUSIBLE_HEIGHT_MIN_CM}-{PLAUSIBLE_HEIGHT_MAX_CM}]."),
            source="validate.plausibility",
        ))

    # Cross-check vs other surrogates
    for other in other_estimates:
        diff = abs(estimate.height_cm - other.height_cm)
        if diff > CROSS_CHECK_THRESHOLD_CM:
            warnings.append(Warning(
                level="warn",
                message=(f"Cross-check disagreement: {estimate.formula_id}="
                         f"{estimate.height_cm:.1f}cm vs {other.formula_id}="
                         f"{other.height_cm:.1f}cm (diff {diff:.1f}cm "
                         f">{CROSS_CHECK_THRESHOLD_CM}cm)."),
                source="validate.cross_check",
            ))

    # Bilateral symmetry
    if surrogate.bilateral_asymmetry_mm is not None \
       and surrogate.bilateral_asymmetry_mm > SYMMETRY_THRESHOLD_MM:
        warnings.append(Warning(
            level="warn",
            message=(f"Bilateral asymmetry {surrogate.bilateral_asymmetry_mm:.1f}mm "
                     f"exceeds threshold {SYMMETRY_THRESHOLD_MM}mm."),
            source="validate.symmetry",
        ))

    # Pose confidence
    low = [(name, c) for name, c in surrogate.landmarks_used
           if c < LANDMARK_CONFIDENCE_THRESHOLD]
    if low:
        warnings.append(Warning(
            level="warn",
            message=("Low pose confidence on landmarks: "
                     + ", ".join(f"{n}={c:.2f}" for n, c in low) + "."),
            source="validate.pose_confidence",
        ))

    # Calibration quality
    if calibration.reprojection_error_px > REPROJECTION_ERROR_THRESHOLD_PX:
        warnings.append(Warning(
            level="warn",
            message=(f"Calibration reprojection error "
                     f"{calibration.reprojection_error_px:.2f}px exceeds threshold "
                     f"{REPROJECTION_ERROR_THRESHOLD_PX}px."),
            source="validate.calibration",
        ))

    return ValidationReport(warnings=warnings)
