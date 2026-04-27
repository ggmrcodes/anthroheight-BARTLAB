"""Streamlit operator UI: capture → flag review → surrogate pick → save."""
from __future__ import annotations
from datetime import datetime, timezone
import os
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from anthroheight.capture import run_qc
from anthroheight.records import (
    PatientMetadata, MeasurementRecord, Landmark,
)
from anthroheight.pipeline import run as pipeline_run
from anthroheight.io_export import export
from anthroheight.bed_layouts import BED_CORNERS_MM
from anthroheight.calibration import calibrate
from anthroheight.flags import compute_flags
from anthroheight.formulas import bassey, chumlea, must
from anthroheight.live_cv import LiveCVProcessor
from anthroheight.measure import measure_surrogate
from anthroheight.validate import run as validate_run


DATA_ROOT = Path(os.getenv("ANTHROHEIGHT_DATA_ROOT", "data/measurements")).resolve()

# ---------------------------------------------------------------------------
# Minimal CSS — only tokens and structural corrections.
# Tokens mirror .streamlit/config.toml: bg=#E5E2D5  ink=#1A1A1A  accent=#9B2520
# ---------------------------------------------------------------------------
_CSS = """
<style>
:root {
    --bg:     #E5E2D5;
    --ink:    #1A1A1A;
    --accent: #9B2520;
    --muted:  #555550;
}

.stApp {
    background: var(--bg) !important;
    font-family: 'Inter', 'Helvetica Neue', system-ui, sans-serif !important;
}

.main .block-container {
    max-width: 820px !important;
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
}

[data-testid="stSidebar"] {
    background: #D8D4C6 !important;
    border-right: 1px solid #B0AB9E !important;
}

/* Inputs */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea {
    background: var(--bg) !important;
    border: 1px solid #999692 !important;
    border-radius: 2px !important;
    color: var(--ink) !important;
}
[data-testid="stSelectbox"] > div > div {
    background: var(--bg) !important;
    border: 1px solid #999692 !important;
    border-radius: 2px !important;
}
[data-testid="stFileUploader"] section {
    background: var(--bg) !important;
    border: 1px dashed #999692 !important;
    border-radius: 2px !important;
}

/* Headings */
h1, h2, h3 {
    font-weight: 500 !important;
    color: var(--ink) !important;
    letter-spacing: -0.3px !important;
}

/* Code spans */
.stMarkdown code {
    font-family: 'SF Mono', 'Menlo', 'Consolas', monospace !important;
    font-size: 0.82rem !important;
    background: rgba(0,0,0,0.06) !important;
    padding: 1px 5px !important;
    border-radius: 2px !important;
}

footer { visibility: hidden !important; }
#MainMenu { visibility: hidden !important; }
</style>
"""


def _result_html(
    height_cm: float,
    see_cm: float,
    formula_id: str,
    surrogate_name: str,
    segment_mm: float,
) -> str:
    return f"""
<div style="margin: 1.5rem 0 1rem 1rem;">
  <div style="display:flex; align-items:baseline; gap:0.5rem; margin-bottom:0.5rem;">
    <span style="font-family:'Bodoni Moda','Bodoni 72','Didot','Times New Roman',serif;
                 font-size:4rem; color:var(--ink,#1A1A1A); line-height:1; font-weight:400;">
      {height_cm:.1f}
    </span>
    <span style="font-size:1.1rem; color:#555550; font-weight:300;">cm</span>
    <span style="font-family:'SF Mono','Menlo','Consolas',monospace;
                 font-size:0.82rem; color:#555550; margin-left:0.75rem;">
      &plusmn;&thinsp;{see_cm:.1f}&thinsp;cm SEE
    </span>
  </div>
  <p style="font-family:'SF Mono','Menlo','Consolas',monospace;
            font-size:0.78rem; color:#555550; margin:0; line-height:1.8;">
    formula&nbsp;&nbsp;<strong style="color:#1A1A1A;">{formula_id}</strong>
    &ensp;&middot;&ensp;
    surrogate&nbsp;&nbsp;<strong style="color:#1A1A1A;">{surrogate_name}</strong>
    &ensp;&middot;&ensp;
    segment&nbsp;&nbsp;<strong style="color:#1A1A1A;">{segment_mm:.1f}&thinsp;mm</strong>
  </p>
</div>
"""


def _patient_form() -> PatientMetadata | None:
    st.markdown("### Patient")

    pid = st.text_input("Patient ID")

    col_age, col_sex = st.columns(2)
    with col_age:
        age = st.number_input("Age (years)", min_value=0, max_value=130, value=70)
    with col_sex:
        sex = st.selectbox("Sex", ["M", "F"])

    col_eth, col_notes = st.columns([1, 2])
    with col_eth:
        eth = st.selectbox("Ethnicity", ["white", "black"])
    with col_notes:
        notes = st.text_area("Notes", height=68)

    st.markdown("**Clinical flags**")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        kyphosis = st.checkbox("Kyphosis")
        scoliosis = st.checkbox("Scoliosis")
    with col_f2:
        llc = st.checkbox("Lower-limb contracture")
        ulc = st.checkbox("Upper-limb contracture")
    with col_f3:
        amp = st.checkbox("Amputation")

    flags = {
        "kyphosis": kyphosis,
        "scoliosis": scoliosis,
        "lower_limb_contracture": llc,
        "upper_limb_contracture": ulc,
        "amputation": amp,
    }

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


# ---- demo-mode helpers ---------------------------------------------------
# When the user ticks "Demo mode" in the sidebar, the UI bypasses the real
# pose model and uses a fixed set of anatomically-realistic landmarks
# matched to the synthetic supine body in samples/synthetic_supine.png.
# Everything else (calibration, measure, formula, validate, io_export) runs
# for real.  This lets the meeting show the complete in-UI flow even though
# MediaPipe Pose is out-of-distribution on a stick figure.

_DEMO_LANDMARKS: list[Landmark] = [
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


def _demo_run(image, patient, chosen_surrogate, side_preference,
              operator_id) -> MeasurementRecord:
    """End-to-end pipeline using fixed demo landmarks (bypasses pose model)."""
    cal = calibrate(image, expected_marker_layout=BED_CORNERS_MM)
    landmarks = _DEMO_LANDMARKS
    flags_ = compute_flags(landmarks)
    surrogate = measure_surrogate(
        chosen_surrogate, landmarks, cal, side_preference,
    )
    if chosen_surrogate == "knee_height":
        estimate = chumlea.estimate_height(
            knee_mm=surrogate.segment_length_mm,
            age_years=patient.age_years,
            sex=patient.sex,
            ethnicity=patient.ethnicity,
        )
    elif chosen_surrogate == "demispan":
        estimate = bassey.estimate_height(
            demispan_mm=surrogate.segment_length_mm, sex=patient.sex,
        )
    else:
        estimate = must.estimate_height(
            ulna_mm=surrogate.segment_length_mm,
            age_years=patient.age_years, sex=patient.sex,
        )
    validation = validate_run(estimate, surrogate, cal, other_estimates=[])
    return MeasurementRecord(
        patient_metadata=patient,
        capture_timestamp=datetime.now(timezone.utc).isoformat(),
        image_path="",
        calibration=cal, landmarks=landmarks, flags=flags_,
        surrogate=surrogate, estimate=estimate,
        validation=validation,
        sum_of_segments_mm=None, operator_id=operator_id,
    )


# ---- live-CV capture helper ---------------------------------------------
# Mounts a streamlit-webrtc video stream that runs the LiveCVProcessor on
# every inbound frame.  The processor draws ArUco bounding boxes, the AR
# blueprint wireframe of the 60x180 cm bed plane, the pose skeleton, a QC
# strip and a READY banner directly on the video.  Auto-shutter fires once
# the QC conditions hold for 1.5 s; the operator can also tap "Snap now"
# to capture the most recent clean frame immediately.

_RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def _live_cv_capture() -> np.ndarray | None:
    if "live_capture" not in st.session_state:
        st.session_state["live_capture"] = None

    ctx = webrtc_streamer(
        key="anthroheight-live-cv",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=_RTC_CONFIG,
        video_processor_factory=LiveCVProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
            },
            "audio": False,
        },
        async_processing=True,
    )

    proc = ctx.video_processor
    auto_img = proc.take_captured_image() if proc is not None else None
    if auto_img is not None:
        st.session_state["live_capture"] = auto_img

    col_snap, col_clear = st.columns([1, 1])
    with col_snap:
        snap_clicked = st.button(
            "Snap now",
            disabled=proc is None,
            help="Grab the current frame. The auto-shutter also fires on its "
                 "own when MARKERS, REPROJ, SHARPNESS and EXPOSURE all read "
                 "green for 1.5 seconds.",
            use_container_width=True,
        )
    with col_clear:
        if st.button(
            "Clear capture",
            disabled=st.session_state["live_capture"] is None,
            use_container_width=True,
        ):
            st.session_state["live_capture"] = None
            if proc is not None:
                proc.reset_capture()

    if snap_clicked and proc is not None:
        snap_img = proc.take_current_frame()
        if snap_img is not None:
            st.session_state["live_capture"] = snap_img

    if proc is not None:
        t = proc.telemetry()
        scale_txt = (
            f"{t.mm_per_px:.2f} mm/px" if t.mm_per_px else "scale --"
        )
        reproj_txt = (
            f"{t.reprojection_error_px:.2f} px"
            if t.reprojection_error_px is not None else "--"
        )
        ready_txt = "READY" if t.ready else "aim at bed"
        st.caption(
            f"`{t.n_markers}/4 markers` · `{scale_txt}` · "
            f"`reproj {reproj_txt}` · `blur {t.blur_score:.0f}` · "
            f"`pose {'on' if t.pose_detected else 'off'}` · **{ready_txt}**"
        )
    else:
        st.caption("Click START above to begin streaming from your camera.")

    return st.session_state["live_capture"]


def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("**BART Lab** — Anthropometric Height System")
        st.caption("Supine surrogate estimation for clinical research")
        st.divider()

        operator = st.text_input("Operator ID", value="op1")
        source = st.radio(
            "Image source",
            ["Upload file", "Camera (live CV)"],
        )

        st.divider()
        demo_mode = st.checkbox(
            "Demo mode (use fixed landmarks)",
            value=False,
            help="Bypass MediaPipe Pose and use anatomically realistic "
                 "constructed landmarks. Useful when uploading the synthetic "
                 "stick-figure image for in-meeting demos.",
        )

    # ── Page header ──────────────────────────────────────────────────────────
    st.markdown("## Supine Height Estimator")

    # ── Patient form ─────────────────────────────────────────────────────────
    patient = _patient_form()
    if patient is None:
        st.info("Enter a Patient ID above to continue.")
        return

    # ── Image acquisition ────────────────────────────────────────────────────
    st.markdown("### Image")

    image: np.ndarray | None = None
    if source == "Upload file":
        f = st.file_uploader("Image", type=["png", "jpg", "jpeg"])
        if f is not None:
            arr = np.frombuffer(f.read(), dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        image = _live_cv_capture()

    if image is None:
        return

    # Show the acquired frame so the operator can verify before measuring.
    # Streamlit expects RGB; OpenCV uses BGR — pass channels="BGR" to skip the
    # conversion step.
    st.image(
        image,
        channels="BGR",
        caption=f"Captured frame · {image.shape[1]}×{image.shape[0]} px",
        use_container_width=True,
    )

    qc = run_qc(image)
    if not qc.passed:
        st.error("Image QC failed:\n- " + "\n- ".join(qc.reasons))
        return

    # ── Measurement options ──────────────────────────────────────────────────
    st.markdown("### Measurement")

    recommended = _surrogate_recommendation(patient.operator_input_flags)
    col_surr, col_side = st.columns(2)
    with col_surr:
        surrogate = st.selectbox(
            "Surrogate",
            ["knee_height", "demispan", "ulna"],
            index=["knee_height", "demispan", "ulna"].index(recommended),
        )
    with col_side:
        side = st.selectbox("Side preference", ["both", "left", "right"], index=0)

    run_clicked = st.button("Run Measurement")

    if not run_clicked:
        return

    try:
        if demo_mode:
            rec = _demo_run(
                image=image, patient=patient,
                chosen_surrogate=surrogate, side_preference=side,
                operator_id=operator,
            )
        else:
            rec = pipeline_run(
                image_bgr=image,
                patient=patient,
                chosen_surrogate=surrogate,
                side_preference=side,
                operator_id=operator,
                expected_marker_layout=BED_CORNERS_MM,
                timestamp_iso=datetime.now(timezone.utc).isoformat(),
            )
    except Exception as e:
        st.error(f"Measurement failed: {type(e).__name__}: {e}")
        return

    # ── Result ───────────────────────────────────────────────────────────────
    st.markdown("### Result")

    st.markdown(
        _result_html(
            height_cm=rec.estimate.height_cm,
            see_cm=rec.estimate.standard_error_cm,
            formula_id=rec.estimate.formula_id,
            surrogate_name=rec.surrogate.surrogate_name,
            segment_mm=rec.surrogate.segment_length_mm,
        ),
        unsafe_allow_html=True,
    )

    # CV deformity flags
    if rec.flags.kyphosis_suspected or rec.flags.scoliosis_suspected:
        flag_txt = (
            ("kyphosis " if rec.flags.kyphosis_suspected else "")
            + ("scoliosis" if rec.flags.scoliosis_suspected else "")
        ).strip()
        st.warning(f"Deformity flags raised by CV: {flag_txt}")

    for w in rec.validation.warnings:
        st.warning(f"[{w.source}] {w.message}")

    if st.button("Save Measurement"):
        out = export(rec, image, output_root=DATA_ROOT)
        st.success(f"Saved: {out.png_path}")


if __name__ == "__main__":
    main()
