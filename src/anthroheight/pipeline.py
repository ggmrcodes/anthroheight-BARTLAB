"""End-to-end orchestrator: image + metadata → MeasurementRecord."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
import numpy as np

from anthroheight.records import (
    PatientMetadata, MeasurementRecord, SurrogateName,
)
from anthroheight.calibration import calibrate
from anthroheight.pose import PoseDetector
from anthroheight.hand import HandDetector
from anthroheight.flags import compute_flags
from anthroheight.measure import measure_surrogate, SidePreference, MissingLandmarkError
from anthroheight.formulas import chumlea, bassey, must
from anthroheight.validate import run as validate_run


def _apply_formula(surrogate, segment_length_mm, patient: PatientMetadata):
    if surrogate == "knee_height":
        return chumlea.estimate_height(
            knee_mm=segment_length_mm,
            age_years=patient.age_years,
            sex=patient.sex,
            ethnicity=patient.ethnicity,
        )
    if surrogate == "demispan":
        return bassey.estimate_height(demispan_mm=segment_length_mm, sex=patient.sex)
    if surrogate == "ulna":
        return must.estimate_height(
            ulna_mm=segment_length_mm,
            age_years=patient.age_years,
            sex=patient.sex,
        )
    raise ValueError(f"unknown surrogate: {surrogate}")


def run(
    image_bgr: np.ndarray,
    patient: PatientMetadata,
    chosen_surrogate: SurrogateName,
    side_preference: SidePreference,
    operator_id: str,
    expected_marker_layout: dict[int, tuple[float, float]],
    timestamp_iso: Optional[str] = None,
) -> MeasurementRecord:
    """Run calibration → pose → measure → formula → validate. Pure: caller
    owns image-acquisition and io_export."""
    if timestamp_iso is None:
        timestamp_iso = datetime.now(timezone.utc).isoformat()

    cal = calibrate(image_bgr, expected_marker_layout=expected_marker_layout)

    with PoseDetector() as pose_detector:
        landmarks = pose_detector.detect(image_bgr)

        if chosen_surrogate == "demispan":
            with HandDetector() as hand_detector:
                side = "left" if side_preference != "right" else "right"
                ft = hand_detector.detect_middle_fingertip(image_bgr, side=side)
                if ft is not None:
                    landmarks = landmarks + [ft]

    flags_ = compute_flags(landmarks)

    surrogate_meas = measure_surrogate(
        chosen_surrogate, landmarks, cal, side_preference,
    )

    estimate = _apply_formula(chosen_surrogate, surrogate_meas.segment_length_mm, patient)

    # NOTE: other_estimates is empty in v1 — pipeline runs a single surrogate
    # per call. Spec §4.8 cross-check (multiple surrogates per session) is a
    # v2 feature; the operator can call run() multiple times and pass prior
    # estimates manually if cross-check is needed today.
    validation = validate_run(estimate, surrogate_meas, cal, other_estimates=[])

    return MeasurementRecord(
        patient_metadata=patient,
        capture_timestamp=timestamp_iso,
        image_path="",   # set by io_export when saved
        calibration=cal,
        landmarks=landmarks,
        flags=flags_,
        surrogate=surrogate_meas,
        estimate=estimate,
        validation=validation,
        sum_of_segments_mm=None,
        operator_id=operator_id,
    )
