# Computer-Vision Estimation of Anthropometric Standing Height in Supine Disabled Patients (Single RGB Camera)

**Status:** Design approved, awaiting implementation plan
**Author:** BART Lab
**Date:** 2026-04-27

---

## 1. Overview

A computer-vision system that estimates the standing height of patients who cannot stand (paralysis, contractures, severe disability), using a single overhead RGB camera and validated anthropometric surrogate formulas. The system is designed for use as a research/thesis tool inside the BART Lab and as a methodological foundation for future bedside clinical deployments.

### 1.1 Problem

Standing height is required for BMI, drug dosing, growth tracking, and nutritional screening. Patients who cannot stand are typically measured by tape-measure heel-to-crown supine length, which is invalid in the presence of contractures or spinal deformities such as kyphosis and scoliosis. Clinical practice substitutes manual surrogate measurements (knee height, demispan, ulna length) but these require trained staff and physical contact and introduce inter-operator variability.

This system replaces the manual caliper/tape step with single-RGB computer vision while preserving the clinically validated surrogate-formula framework.

### 1.2 Goals (v1)

- Estimate standing height from a single overhead RGB capture. Two distinct accuracy targets:
  - **System precision (phantom):** measured segment length within ±5 mm of tape-measure ground truth on a rigid mannequin. This isolates CV+calibration error from formula error.
  - **End-to-end accuracy (volunteer):** estimated standing height within the published formula SEE (Chumlea ≈ ±3.5 cm, Bassey ≈ ±4 cm, MUST ≈ ±5 cm) of stadiometer ground truth. The system cannot beat the formula's intrinsic error; the goal is to not add meaningfully to it.
- Support three surrogate measurements: **knee height** (primary), **demispan** (fallback), **ulna length** (fallback).
- Robust to spinal deformities (kyphosis, scoliosis, lordosis) by avoiding the spinal axis entirely in the measurement path.
- Single-shot capture flow with operator review before save; no realtime processing required.
- Audit-quality output: annotated image, machine-readable record, append-only CSV log.
- Validation harness suitable for thesis/paper-grade reporting (Bland-Altman, ICC, limits of agreement).

### 1.3 Non-goals (v1)

- Realtime/video processing.
- Multi-camera or RGB-D capture (single RGB constraint is intentional, drives portability).
- Automatic patient identification, EHR integration, or clinical certification.
- Mobile (phone) deployment — desktop/laptop only in v1; mobile is a v2 candidate.
- Fully autonomous deformity detection — operator-input deformity flags are primary; CV-derived flags are a stretch goal as a second-opinion display.

---

## 2. Users and deployment context

- **Primary user:** the system operator (researcher, BART Lab member). Familiar with anthropometric concepts; will be trained on the protocol.
- **Setting:** controlled lab room with a fixed bed and a ceiling-mounted overhead RGB camera. Lighting can be assumed reasonable (no direct sun on bed). Operator triggers capture from a laptop alongside the bed.
- **Patients (eval phase):** mixed disability cohort recruited under IRB approval; explicitly includes patients with kyphosis, scoliosis, and lower-limb contractures so the system is evaluated on the population it is built for.

---

## 3. System architecture

### 3.1 Pipeline

```
  [overhead RGB image] + [patient metadata: id, age, sex, ethnicity, deformity flags]
         │
         ▼
  capture/      QC: blur (Laplacian variance), exposure (mean luma),
                marker count (≥4 ArUco visible) → reject or pass
         │
         ▼
  calibration/  ArUco detect → homography H + mm/px scale + bed plane
         │
         ▼
  pose/         MediaPipe Pose → 33 landmarks (x, y, confidence)
         │ + hand/   MediaPipe Hands fingertip — only when demispan selected
         ▼
  flags/        Compute deformity indicators from landmarks
                (CV-derived second opinion on operator-input flags)
         │
         ▼
  ┌────────── operator review (UI) ──────────┐
  │ shows detected flags + recommended       │
  │ surrogate; operator confirms / overrides │
  └─────────────────────┬────────────────────┘
                        │ chosen surrogate ∈ {knee_height, demispan, ulna}
                        ▼
  measure/      Compute chosen segment length in mm (apply H)
         │
         ▼
  formulas/     Apply matching regression (Chumlea / Bassey / MUST),
                keyed by sex/age/ethnicity → height_cm + standard_error_cm
         │
         ▼
  validate/     Plausibility (100–220 cm), cross-check (multi-surrogate
                agreement ±5 cm), symmetry (L vs R ±2 cm),
                pose confidence, reprojection error
         │
         ▼
  io/export     Annotated PNG + JSON record + CSV log row
                Diagnostic-only: sum-of-segments value (never reported as height)
```

### 3.2 Module inventory

| Module | Responsibility | Pure? |
|---|---|---|
| `capture/` | Acquire frame, run QC gates, reject bad input | No (I/O) |
| `calibration/` | ArUco → homography H + mm/px scale | Yes (image in, matrix out) |
| `pose/` | Wrapper around pose model (MediaPipe Pose v1) | No (model) |
| `hand/` | Wrapper around hand model (MediaPipe Hands; demispan only) | No (model) |
| `flags/` | Landmarks → deformity indicators | Yes |
| `measure/` | Chosen surrogate + landmarks + H → segment length mm | Yes |
| `formulas/` | Chumlea / Bassey / MUST coefficients & application | Yes |
| `validate/` | Range, cross-check, symmetry, confidence checks | Yes |
| `io/export` | Annotated image, CSV append, JSON record | No (I/O) |
| `pipeline.py` | Orchestrates the above | Stateless |
| `ui_streamlit.py` | Capture review interface | No (UI) |
| `eval/` | Offline validation harness — separate from runtime | No (analysis) |

The pure modules (`flags`, `measure`, `formulas`, `validate`) carry the algorithmic core and are unit-tested with synthetic landmark inputs (no images required). The wrappers (`pose`, `hand`) hide model implementation behind a stable interface so the model can be swapped without disturbing the measurement code.

---

## 4. Component specifications

### 4.1 `capture/`

**Inputs:** trigger event from UI.
**Outputs:** RGB image (numpy array), capture timestamp, QC report.

**QC gates (all must pass):**
- Laplacian variance ≥ threshold (rejects blur). Initial threshold 100; calibrate on phantom.
- Mean luma in [40, 220] (rejects under/over-exposure).
- ArUco marker count ≥ 4 (rejects when patient/blanket occludes calibration).

If any gate fails, return failure with reason; UI re-prompts operator without invoking downstream pipeline.

### 4.2 `calibration/`

**Inputs:** RGB image.
**Outputs:** `CalibrationResult` (Pydantic): `homography_matrix` (3×3), `mm_per_px_central` (float), `marker_corners_px` (list), `reprojection_error_px` (float).

**Algorithm:**
1. Detect ArUco markers (`cv2.aruco.detectMarkers`) using a known dictionary (`DICT_5X5_50`).
2. Require ≥4 markers from a known set placed at known mm coordinates on the bed mat.
3. Compute homography H from detected pixel corners → known mm coords (`cv2.findHomography`, RANSAC).
4. Validate by reprojecting one held-out marker and measuring pixel error; reject if > 2 px.

**Calibration target (physical):** 4× 5 cm ArUco markers at the four corners of a 60×180 cm rectangle on the bed mat. Backup: 4 additional markers offset, so up to 4 can be occluded without losing calibration.

### 4.3 `pose/`

**Inputs:** RGB image (pre-rotated to canonical head-up orientation).
**Outputs:** list of `Landmark` objects: `name`, `x_px`, `y_px`, `confidence`.

**Implementation:** wrapper around MediaPipe Pose (33-landmark model). Names mapped to a canonical schema (`Landmark.name` is one of: `nose`, `left_shoulder`, `right_shoulder`, ..., `left_ankle`, `right_ankle`). Landmarks below confidence 0.5 are surfaced but flagged.

**Critical risk and mitigation:** MediaPipe Pose is trained on standing/walking views; supine overhead is out of distribution. Mitigations:
1. Pre-rotate image to canonical orientation (operator clicks "head end" once during room setup; rotation applied automatically thereafter).
2. Wrapper interface is stable, allowing swap to YOLOv8-pose, MMPose RTMPose, or fine-tuned model without touching `measure/`.
3. Validation phase 1 (phantom) must include explicit pose-accuracy stats; if below threshold, swap or fine-tune before phase 2.

### 4.4 `hand/`

**Inputs:** RGB image, optional bounding-box prior from arm landmarks.
**Outputs:** `Landmark` for `middle_fingertip` (left or right per request), with confidence.

**Implementation:** wrapper around MediaPipe Hands. Only invoked when demispan is the chosen surrogate. If `confidence < 0.5`, the wrapper returns failure; the UI surfaces the failure to the operator and offers ulna as an in-session fallback (operator confirms before re-measure). Fallback decision lives in the UI, not in the wrapper or pipeline — keeps the wrapper pure and lets the operator stay in control.

### 4.5 `flags/`

**Inputs:** landmarks (pixel coords), patient metadata.
**Outputs:** `DeformityFlags` (Pydantic): per-flag boolean + per-flag confidence.

**Flags computed:**
- `kyphosis_suspected`: head landmark lateral offset from shoulder-hip axis exceeds threshold (proxy for raised head in overhead projection).
- `scoliosis_suspected`: deviation of head/shoulder/hip midpoints from a straight axis > threshold.
- `lower_limb_contracture_suspected`: knee joint angle from straight > 30°.
- `upper_limb_contracture_suspected`: elbow joint angle from straight > 30°.

Thresholds are initial values; calibrated empirically during validation phase 1. Flags are advisory — they are displayed alongside operator-input flags but do not override operator decisions.

### 4.6 `measure/`

**Inputs:** chosen surrogate, landmarks, calibration result.
**Outputs:** `SurrogateMeasurement` (Pydantic): `surrogate_name`, `segment_length_mm`, `landmarks_used` (names + confidences), `bilateral_mean_used` (bool).

**Computation per surrogate:**

| Surrogate | Landmarks | Computation |
|---|---|---|
| `knee_height` | knee + ankle (lateral side facing camera; if both visible, use mean) | apply H to both points, Euclidean distance in mm |
| `demispan` | shoulder midpoint (proxy for suprasternal notch — see note) + middle fingertip | apply H, Euclidean distance |
| `ulna` | elbow + wrist (lateral side, or both if visible) | apply H, Euclidean distance |

If both sides are visible for a bilateral surrogate, compute both, return mean, set `bilateral_asymmetry_mm` for downstream symmetry validation.

**Note on demispan landmark proxy.** Bassey's original formula is defined from the suprasternal notch to the middle fingertip. Suprasternal notch is not in MediaPipe Pose's 33-landmark set. v1 substitutes the midpoint of the two shoulder landmarks as a proxy. This introduces a known bias of approximately ±1–2 cm depending on shoulder breadth and is the most likely source of system-induced error in the demispan path. Two follow-ups are required: (a) a small calibration study during phase 2 to characterize the bias on healthy volunteers (compare CV demispan with manual sternal-notch demispan), and (b) consideration of a hand-anatomical-feature model that detects the sternal notch directly (e.g., MMPose WholeBody) before phase 3.

### 4.7 `formulas/`

**Layout:** one file per formula family — `chumlea.py`, `bassey.py`, `must.py`. Each exposes:

```python
def estimate_height(
    segment_mm: float,
    age_years: int | None,
    sex: Literal["M", "F"],
    ethnicity: str | None,
) -> HeightEstimate
```

**`HeightEstimate`** (Pydantic): `height_cm`, `standard_error_cm`, `formula_id`, `formula_citation`, `validity_warnings` (list).

**Coefficients** live as plain Python dicts/JSON inside each formula file with citations in comments. Single source of truth — changing a coefficient touches one file.

**Validity guards:** each formula declares its validation cohort age/sex/ethnicity range. Estimating outside the range returns the result but adds a `validity_warning`. Never silently extrapolates without flagging.

**Citations:**
- Chumlea WC et al. (1985, 1994) — knee-height regressions.
- Bassey EJ (1986) — demispan.
- BAPEN MUST / NICE CG32 — ulna-length tables.

Coefficient values are not reproduced inline in this spec; they are implementation-time citations into the source files.

### 4.8 `validate/`

**Inputs:** `HeightEstimate`, `SurrogateMeasurement`, `CalibrationResult`, optional list of additional surrogate estimates from same session.
**Outputs:** `ValidationReport` (Pydantic): list of `Warning` objects (level: info/warn/error, message, source).

**Checks:**
- **Plausibility:** 100 cm ≤ height_cm ≤ 220 cm.
- **Cross-check:** if ≥2 surrogates measured in same session, pairwise differences ≤ ±5 cm; otherwise warn.
- **Symmetry:** bilateral surrogate L vs R difference ≤ ±2 cm; otherwise warn.
- **Pose confidence:** all landmarks used ≥ 0.7 confidence; otherwise warn.
- **Calibration quality:** reprojection error ≤ 2 px; otherwise warn.

Warnings annotate the output — they do not block save. The operator confirms before save.

### 4.9 `io/export`

**Outputs per measurement:**
- `data/measurements/<patient_id>/<timestamp>.png` — annotated image: landmarks overlaid, surrogate segment line, ArUco corners highlighted, computed height stamped.
- `data/measurements/<patient_id>/<timestamp>.json` — full Pydantic record (image path, calibration result, pose confidences, surrogate measurement, formula result, flags, warnings, operator id, free-text notes).
- `data/measurements/log.csv` — append-only summary row for batch analysis.

A diagnostic field (`sum_of_segments_mm`) is computed and stored in the JSON for QA but is **never** surfaced as the reported height.

**Privacy:** face region is blurred in the saved PNG by default. Operator can opt to retain unblurred (disabled by default).

### 4.10 `ui_streamlit.py`

Single-page Streamlit app. Flow:

1. Operator enters patient metadata (id, age, sex, ethnicity, deformity flags, notes).
2. "Capture" button triggers `capture/` → image preview shown.
3. Pipeline runs through `flags/`. UI shows: flag panel (operator-input + CV-derived) + recommended surrogate.
4. Operator confirms or overrides surrogate choice.
5. Pipeline runs through `measure/` → `formulas/` → `validate/`. UI shows: estimated height with SEE, annotated image, all warnings.
6. Operator confirms ("Save") or discards ("Re-capture").
7. On save: `io/export` writes all artifacts.

---

## 5. Data model

All persisted records use Pydantic v2 schemas in `records.py`:

- `PatientMetadata` — id, age_years, sex, ethnicity, operator_input_flags, notes
- `CalibrationResult` — homography_matrix (serialized), mm_per_px_central, marker_corners_px, reprojection_error_px
- `Landmark` — name, x_px, y_px, confidence
- `DeformityFlags` — per-flag bool + confidence (CV-derived)
- `SurrogateMeasurement` — surrogate_name, segment_length_mm, landmarks_used, bilateral_mean_used, bilateral_asymmetry_mm
- `HeightEstimate` — height_cm, standard_error_cm, formula_id, formula_citation, validity_warnings
- `ValidationReport` — list of Warning(level, message, source)
- `MeasurementRecord` — top-level: patient_metadata + capture_timestamp + image_path + calibration + landmarks + flags + surrogate + estimate + validation + sum_of_segments_mm (diagnostic) + operator_id

---

## 6. Tech stack

| Concern | Choice |
|---|---|
| Language | Python 3.11+ |
| Image / ArUco | `opencv-contrib-python` 4.10+ |
| Pose | `mediapipe` (Pose) |
| Hands | `mediapipe` (Hands), demispan only |
| Records | `pydantic` v2 |
| UI | `streamlit` |
| Tests | `pytest`, `numpy.testing` |
| Eval | `pandas`, `matplotlib`, Jupyter |
| Packaging | `pyproject.toml`, editable install |

No Conda required.

---

## 7. Repo layout

```
BART_LAB/
├── pyproject.toml
├── README.md
├── .gitignore                # ignores data/, .superpowers/, __pycache__/, etc.
├── docs/
│   ├── superpowers/specs/    # design docs (this file + future)
│   └── protocol/             # measurement-protocol PDF for clinicians
├── src/anthroheight/
│   ├── __init__.py
│   ├── capture.py
│   ├── calibration.py
│   ├── pose.py
│   ├── hand.py
│   ├── flags.py
│   ├── measure.py
│   ├── validate.py
│   ├── io_export.py
│   ├── records.py
│   ├── pipeline.py
│   ├── formulas/
│   │   ├── __init__.py
│   │   ├── chumlea.py
│   │   ├── bassey.py
│   │   └── must.py
│   └── ui_streamlit.py
├── tests/                    # mirrors src/
│   ├── test_calibration.py
│   ├── test_pose_wrapper.py
│   ├── test_flags.py
│   ├── test_measure.py
│   ├── test_formulas/
│   │   ├── test_chumlea.py
│   │   ├── test_bassey.py
│   │   └── test_must.py
│   ├── test_validate.py
│   └── fixtures/             # synthetic images, known-answer landmark sets
└── eval/                     # validation harness, separate from runtime
    ├── phantom_eval.py
    ├── volunteer_eval.py
    ├── patient_eval.py
    ├── analysis.ipynb        # Bland-Altman, ICC, LoA
    └── ground_truth.csv
data/                         # gitignored: captures, logs (created at runtime)
└── measurements/
```

---

## 8. Validation plan

Three sequential phases, each gating the next.

### 8.1 Phase 1 — Phantom validation

- **Subject:** rigid mannequin with anthropometric segments measured by tape (ground truth ±2 mm).
- **Protocol:** 20 captures per surrogate per condition. Conditions vary lighting (3 levels) and camera height (2 levels).
- **Outputs:** precision (SD across repeats), accuracy (bias vs ground truth), sensitivity to lighting and camera height.
- **Gate to phase 2:** mean absolute error ≤ ±5 mm per surrogate on phantom; pose-landmark stability acceptable.

### 8.2 Phase 2 — Healthy volunteer validation

- **Subjects:** n≈15 healthy adult volunteers, both sexes, range of heights.
- **Ground truth:** standing height by stadiometer.
- **Protocol:** each volunteer captured supine; one capture per surrogate.
- **Outputs:** Bland-Altman (per surrogate vs stadiometer), mean absolute error, ICC₂,₁, limits of agreement.
- **Gate to phase 3:** results within published surrogate-formula SEE (Chumlea ±~3.5 cm, Bassey ±~4 cm, MUST ±~5 cm) — system-induced error not exceeding the formula's intrinsic SEE.

### 8.3 Phase 3 — Patient validation

- **Subjects:** n≈20–30 disabled patients under IRB. Inclusion explicitly covers kyphosis, scoliosis, lower-limb contractures.
- **Ground truth:** clinical Chumlea knee-height with calipers, performed by trained clinician.
- **Protocol:** each patient measured by both system and clinician; order randomized.
- **Outputs:** Bland-Altman vs clinical surrogate; subgroup analysis by deformity type.
- **Inter-rater study:** subset of patients measured by two operators using the system to characterize inter-operator variability.

---

## 9. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Pose model out-of-distribution on supine overhead views | **High** | pre-rotate input to canonical head-up; swap-able wrapper; fine-tune on supine data if phase-1 accuracy is insufficient |
| ArUco occlusion by bedding, blanket, or patient body | Medium | place 8 markers, require 4; protocol step keeps marker zone clear |
| MediaPipe Hands fails for demispan | Low | ulna fallback; both surrogates can be captured in same session |
| Formula coefficients derived on different population than yours | Medium | report SEE prominently; allow custom coefficients via `formulas/` extension; document as paper limitation |
| Out-of-plane error for raised or contracted limbs | Medium | protocol forces flat-on-mattress posing per surrogate; severe cases default to ulna (always flat) |
| Patient privacy (PHI in stored images) | **High** | face-blur on by default, opt-in retention, IRB review before phase 3 |
| Inter-operator variability | Medium | written protocol document, training video, planned inter-rater agreement study in eval phase |

---

## 10. Out of scope (v2+)

- Realtime/video-based capture
- Mobile (phone) deployment
- Automatic patient identification, EHR integration, clinical certification (CE / FDA)
- Multi-camera or RGB-D integration
- Fully autonomous CV-only deformity classification (v1 treats CV-derived flags as second-opinion only)
- Active operator guidance (e.g., "rotate the camera 5° to your left") — v1 assumes static rig

---

## 11. Open questions for implementation phase

These do not block design approval but should be resolved during the implementation plan:

1. Bed mat with embedded ArUco markers — fabricated locally or printed-and-laminated?
2. Camera model and ceiling mount specifics — does BART Lab already have a usable rig, or does v1 include hardware procurement?
3. Patient metadata source — entered by operator each session, or pulled from a study CSV?
4. Streamlit hosting — local-only on the capture laptop, or LAN-accessible?
5. Sample size for phase 3 — final n depends on IRB and clinical access; the plan assumes 20–30 as a planning figure.
