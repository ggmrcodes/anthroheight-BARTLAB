"""End-to-end demo of the anthroheight pipeline on a synthetic overhead image.

Run from repo root:

    python -m demo.demo_synthetic

Produces:
    demo/output/demo.png    — annotated image (height stamp, landmarks, blur)
    demo/output/demo.json   — full Pydantic measurement record
    demo/output/log.csv     — appended summary row

Why synthetic?  The MediaPipe Pose Tasks API needs a downloaded
`pose_landmarker_*.task` file for real inference, and a real overhead supine
photo of a patient cannot be ethically sourced for a demo deck.  This script
exercises the *real* calibration, measure, formulas, validate, and io_export
modules — only the pose model is bypassed by constructing landmarks
directly.  The output is a faithful preview of what the full pipeline will
produce on actual capture data.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np

from anthroheight.calibration import calibrate
from anthroheight.flags import compute_flags
from anthroheight.formulas import chumlea
from anthroheight.io_export import export
from anthroheight.measure import measure_surrogate
from anthroheight.records import (
    Landmark, MeasurementRecord, PatientMetadata,
)
from anthroheight.validate import run as validate_run
from anthroheight.bed_layouts import BED_CORNERS_MM
from tests.fixtures.synthetic_aruco import make_image


# ----- demo patient + camera setup ----------------------------------------
PATIENT = PatientMetadata(
    patient_id="DEMO_001",
    age_years=72,
    sex="F",
    ethnicity="white",
    operator_input_flags={
        "kyphosis": True,        # demo case: mild thoracic kyphosis
        "scoliosis": False,
        "lower_limb_contracture": False,
        "upper_limb_contracture": False,
        "amputation": False,
    },
    notes="demo capture — synthetic image with constructed landmarks",
)

OUTPUT_ROOT = Path(__file__).parent / "output"


def draw_synthetic_supine_body(img: np.ndarray) -> np.ndarray:
    """Overlay an anatomically proportioned stick-figure body on the synthetic
    ArUco bed image so the annotated PNG looks like a real overhead patient
    view.  Coordinates are chosen so that the resulting knee-height in mm
    (after homography) matches a realistic adult female (~48 cm)."""
    out = img.copy()
    # Colors are BGR (OpenCV order). Rendered as warm peach skin + pale blue gown.
    skin = (170, 195, 230)            # warm peach in RGB → BGR-correct here
    gown = (200, 180, 150)            # pale teal-blue scrubs
    shadow = (130, 130, 130)

    # Torso
    cv2.rectangle(out, (490, 290), (710, 870), gown, thickness=-1)
    cv2.rectangle(out, (490, 290), (710, 870), shadow, thickness=2)

    # Head (centered above torso)
    cv2.circle(out, (600, 215), 80, skin, thickness=-1)
    cv2.circle(out, (600, 215), 80, shadow, thickness=2)
    cv2.circle(out, (600, 215), 80, shadow, thickness=2)

    # Arms — alongside the torso, palms-down
    cv2.rectangle(out, (400, 320), (490, 760), skin, -1)
    cv2.rectangle(out, (400, 320), (490, 760), shadow, 2)
    cv2.rectangle(out, (710, 320), (800, 760), skin, -1)
    cv2.rectangle(out, (710, 320), (800, 760), shadow, 2)

    # Legs (longer to hit anatomically correct knee/ankle landmarks)
    cv2.rectangle(out, (510, 870), (595, 1530), skin, -1)
    cv2.rectangle(out, (510, 870), (595, 1530), shadow, 2)
    cv2.rectangle(out, (605, 870), (690, 1530), skin, -1)
    cv2.rectangle(out, (605, 870), (690, 1530), shadow, 2)

    # Feet
    cv2.rectangle(out, (495, 1530), (600, 1605), skin, -1)
    cv2.rectangle(out, (495, 1530), (600, 1605), shadow, 2)
    cv2.rectangle(out, (600, 1530), (705, 1605), skin, -1)
    cv2.rectangle(out, (600, 1530), (705, 1605), shadow, 2)
    return out


def constructed_landmarks() -> list[Landmark]:
    """Anatomically realistic MediaPipe-Pose-style landmarks for the drawn
    body.  Chosen so that knee-to-ankle pixel distance (~410 px) maps via the
    bed-plane homography to ~48 cm knee height — the target value used in the
    Chumlea regression demo for a 72-year-old woman."""
    return [
        Landmark(name="nose",           x_px=600.0, y_px=200.0, confidence=0.94),
        Landmark(name="left_shoulder",  x_px=510.0, y_px=320.0, confidence=0.92),
        Landmark(name="right_shoulder", x_px=690.0, y_px=320.0, confidence=0.92),
        Landmark(name="left_elbow",     x_px=445.0, y_px=540.0, confidence=0.88),
        Landmark(name="right_elbow",    x_px=755.0, y_px=540.0, confidence=0.88),
        Landmark(name="left_wrist",     x_px=445.0, y_px=750.0, confidence=0.84),
        Landmark(name="right_wrist",    x_px=755.0, y_px=750.0, confidence=0.84),
        Landmark(name="left_hip",       x_px=550.0, y_px=860.0, confidence=0.91),
        Landmark(name="right_hip",      x_px=650.0, y_px=860.0, confidence=0.91),
        Landmark(name="left_knee",      x_px=550.0, y_px=1120.0, confidence=0.93),
        Landmark(name="right_knee",     x_px=650.0, y_px=1120.0, confidence=0.93),
        Landmark(name="left_ankle",     x_px=550.0, y_px=1530.0, confidence=0.90),
        Landmark(name="right_ankle",    x_px=650.0, y_px=1530.0, confidence=0.90),
    ]


def main() -> None:
    print("anthroheight — end-to-end synthetic demo")
    print("-" * 60)

    # 1. Build the input image: ArUco bed mat + supine body overlay.
    print("[1/6] Generating synthetic overhead image (1200×1800 px)...")
    bed_img, _marker_centers = make_image()
    image = draw_synthetic_supine_body(bed_img)

    # 2. Real calibration pass on the real ArUco markers.
    print("[2/6] Running ArUco calibration (homography to bed plane)...")
    cal = calibrate(image, expected_marker_layout=BED_CORNERS_MM)
    print(f"        mm/px = {cal.mm_per_px_central:.3f}, "
          f"reprojection error = {cal.reprojection_error_px:.2f} px")

    # 3. Constructed landmarks (substitute for the deferred pose-model load).
    landmarks = constructed_landmarks()
    print(f"[3/6] Using {len(landmarks)} constructed pose landmarks "
          "(real CV bypassed for demo).")

    # 4. Deformity flags from the actual flags module.
    flags_ = compute_flags(landmarks)
    flags_summary = ", ".join(
        name for name, raised in [
            ("kyphosis", flags_.kyphosis_suspected),
            ("scoliosis", flags_.scoliosis_suspected),
            ("lower-limb contracture", flags_.lower_limb_contracture_suspected),
            ("upper-limb contracture", flags_.upper_limb_contracture_suspected),
        ] if raised
    ) or "none"
    print(f"[4/6] CV-derived deformity flags: {flags_summary}")
    print(f"        Operator-input flags: kyphosis=True (demo patient profile)")

    # 5. Surrogate measurement → formula → validation. Knee height is the
    # primary surrogate for any patient with suspected vertebral height loss.
    print("[5/6] Measuring knee height (bilateral mean) → Chumlea 1985...")
    surrogate = measure_surrogate(
        "knee_height", landmarks, cal, side_preference="both",
    )
    estimate = chumlea.estimate_height(
        knee_mm=surrogate.segment_length_mm,
        age_years=PATIENT.age_years,
        sex=PATIENT.sex,
        ethnicity=PATIENT.ethnicity,
    )
    validation = validate_run(estimate, surrogate, cal, other_estimates=[])
    print(f"        knee height = {surrogate.segment_length_mm:.1f} mm")
    print(f"        estimated stature = {estimate.height_cm:.1f} cm "
          f"(±{estimate.standard_error_cm:.1f} cm SEE, {estimate.formula_id})")
    if validation.warnings:
        print("        validation warnings:")
        for w in validation.warnings:
            print(f"          [{w.source}] {w.message}")
    else:
        print("        no validation warnings.")

    # 6. Build the persistent record and let io_export.export do its job
    # (annotated PNG + JSON + CSV append).
    record = MeasurementRecord(
        patient_metadata=PATIENT,
        capture_timestamp=datetime.now(timezone.utc).isoformat(),
        image_path="",             # filled in by export()
        calibration=cal,
        landmarks=landmarks,
        flags=flags_,
        surrogate=surrogate,
        estimate=estimate,
        validation=validation,
        sum_of_segments_mm=None,
        operator_id="demo_runner",
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    paths = export(record, image, output_root=OUTPUT_ROOT, blur_face=True)
    print(f"[6/6] Saved:")
    print(f"        annotated PNG  : {paths.png_path}")
    print(f"        JSON record    : {paths.json_path}")
    print(f"        CSV log        : {paths.csv_path}")
    print("-" * 60)
    print("Done. Open the PNG to see the demo output.")


if __name__ == "__main__":
    main()
