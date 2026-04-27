"""Streamlit operator UI: capture → flag review → surrogate pick → save."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
import streamlit as st

from anthroheight.capture import capture_from_camera, capture_from_file, run_qc
from anthroheight.records import PatientMetadata
from anthroheight.pipeline import run as pipeline_run
from anthroheight.io_export import export
from tests.fixtures.synthetic_aruco import BED_CORNERS_MM   # placeholder bed layout


DATA_ROOT = Path("data/measurements")


def _patient_form() -> PatientMetadata | None:
    st.header("Patient")
    pid = st.text_input("Patient ID")
    age = st.number_input("Age (years)", min_value=0, max_value=130, value=70)
    sex = st.selectbox("Sex", ["M", "F"])
    eth = st.selectbox("Ethnicity", ["white", "black"])
    flags = {
        "kyphosis": st.checkbox("Kyphosis"),
        "scoliosis": st.checkbox("Scoliosis"),
        "lower_limb_contracture": st.checkbox("Lower-limb contracture"),
        "upper_limb_contracture": st.checkbox("Upper-limb contracture"),
        "amputation": st.checkbox("Amputation"),
    }
    notes = st.text_area("Notes")
    if not pid:
        return None
    return PatientMetadata(
        patient_id=pid, age_years=age, sex=sex, ethnicity=eth,
        operator_input_flags=flags, notes=notes,
    )


def _surrogate_recommendation(flags: dict[str, bool]) -> str:
    if flags.get("lower_limb_contracture") or flags.get("amputation"):
        return "ulna"
    return "knee_height"


def main() -> None:
    st.title("Anthropometric supine height estimator")
    operator = st.sidebar.text_input("Operator ID", value="op1")
    source = st.sidebar.radio("Image source", ["Upload file", "Camera (USB)"])

    patient = _patient_form()
    if patient is None:
        st.info("Enter patient ID to continue.")
        return

    image: np.ndarray | None = None
    if source == "Upload file":
        f = st.file_uploader("Image", type=["png", "jpg", "jpeg"])
        if f is not None:
            arr = np.frombuffer(f.read(), dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        if st.button("Capture frame"):
            image = capture_from_camera()

    if image is None:
        return

    qc = run_qc(image)
    if not qc.passed:
        st.error("Image QC failed:\n- " + "\n- ".join(qc.reasons))
        return

    recommended = _surrogate_recommendation(patient.operator_input_flags)
    surrogate = st.selectbox("Surrogate", ["knee_height", "demispan", "ulna"],
                             index=["knee_height", "demispan", "ulna"].index(recommended))
    side = st.selectbox("Side preference", ["both", "left", "right"], index=0)

    if not st.button("Run measurement"):
        return

    rec = pipeline_run(
        image_bgr=image,
        patient=patient,
        chosen_surrogate=surrogate,
        side_preference=side,
        operator_id=operator,
        expected_marker_layout=BED_CORNERS_MM,
        timestamp_iso=datetime.now(timezone.utc).isoformat(),
    )

    st.subheader("Result")
    st.metric("Estimated standing height",
              f"{rec.estimate.height_cm:.1f} cm",
              delta=f"± {rec.estimate.standard_error_cm:.1f} cm SEE")
    st.write(f"Surrogate: **{rec.surrogate.surrogate_name}**, "
             f"segment {rec.surrogate.segment_length_mm:.1f} mm  ·  "
             f"formula `{rec.estimate.formula_id}`")

    if rec.flags.kyphosis_suspected or rec.flags.scoliosis_suspected:
        st.warning("Deformity flags raised by CV: "
                   + ("kyphosis " if rec.flags.kyphosis_suspected else "")
                   + ("scoliosis" if rec.flags.scoliosis_suspected else ""))

    for w in rec.validation.warnings:
        st.warning(f"[{w.source}] {w.message}")

    if st.button("Save measurement"):
        out = export(rec, image, output_root=DATA_ROOT)
        st.success(f"Saved: {out.png_path}")


if __name__ == "__main__":
    main()
