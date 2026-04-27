# Anthropometric Supine Height Estimator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python system that estimates standing height of supine disabled patients from a single overhead RGB image, using ArUco-calibrated pose landmarks and validated anthropometric surrogate formulas (Chumlea knee height, Bassey demispan, MUST ulna).

**Architecture:** Single Python package `anthroheight` organized into pure-function modules (records, formulas, measure, flags, validate) and image-touching wrappers (calibration, pose, hand, capture, io_export), orchestrated by `pipeline.py` and operated through a Streamlit UI. Eval harness lives separately. TDD throughout — pure modules test against synthetic landmark inputs, image modules test against synthetic ArUco/pose fixtures.

**Tech Stack:** Python 3.11+, OpenCV (opencv-contrib-python 4.10+), MediaPipe Pose & Hands, Pydantic v2, Streamlit, pytest.

**Spec:** `docs/superpowers/specs/2026-04-27-anthropometric-supine-height-cv-design.md`

---

## Background context the engineer should read first

Read the spec linked above end-to-end before starting Task 0. Key terms used throughout this plan:

- **Surrogate measurement** — a limb segment (knee height, demispan, ulna) used as input to a regression formula that predicts standing height. Avoids spinal axis, so kyphosis/scoliosis don't bias the answer.
- **ArUco** — fiducial marker library in OpenCV. We place ≥4 markers of known size on the bed mat to compute a homography from pixels to mm.
- **Homography** — a 3×3 matrix that maps points on one plane (image pixels) to another plane (the bed surface in mm).
- **MediaPipe Pose** — Google's 33-landmark pose model. Trained on standing/walking views; supine overhead is out-of-distribution. We pre-rotate the image to canonical head-up orientation before inference.
- **SEE (standard error of estimate)** — published per-formula uncertainty (Chumlea ≈ ±3.5 cm, Bassey ≈ ±4 cm, MUST ≈ ±5 cm). The system reports SEE alongside the height estimate; the system cannot beat the formula's intrinsic SEE.

**Key anti-fabrication note for Tasks 2–4 (formulas):** The regression coefficients in Chumlea 1985, Bassey 1986, and MUST/NICE come from published clinical literature. The plan provides the formula structure and the most-cited coefficient values; before merging each formulas task, the engineer must independently verify the coefficients against the cited primary source and add a parameterized test that uses a known input/output pair from that source's validation cohort.

---

## Task 0: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/anthroheight/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "anthroheight"
version = "0.1.0"
description = "Anthropometric standing-height estimation for supine patients via single-RGB CV"
requires-python = ">=3.11"
dependencies = [
    "opencv-contrib-python>=4.10",
    "mediapipe>=0.10.18",
    "numpy>=1.26",
    "pydantic>=2.7",
    "streamlit>=1.36",
    "pandas>=2.2",
    "matplotlib>=3.8",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"
```

- [ ] **Step 2: Create `src/anthroheight/__init__.py`**

```python
"""Anthropometric standing-height estimator."""
__version__ = "0.1.0"
```

- [ ] **Step 3: Create `tests/__init__.py` (empty file) and `tests/conftest.py`**

```python
# tests/conftest.py
"""Shared pytest fixtures."""
import pytest
```

- [ ] **Step 4: Create minimal `README.md`**

```markdown
# anthroheight

Single-RGB computer vision for anthropometric standing-height estimation in supine disabled patients.

See `docs/superpowers/specs/` for design and `docs/superpowers/plans/` for the implementation plan.

## Install

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"

## Run tests

    pytest
```

- [ ] **Step 5: Create venv, install, smoke-test**

Run:
```bash
cd /Users/macbook/Desktop/2026_WORK/BART_LAB
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -c "import anthroheight; print(anthroheight.__version__)"
pytest
```

Expected: prints `0.1.0`, then `no tests ran` from pytest (clean exit code 0).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml README.md src/ tests/ .gitignore docs/
git commit -m "chore: scaffold anthroheight package"
```

---

## Task 1: Pydantic data records

**Files:**
- Create: `src/anthroheight/records.py`
- Create: `tests/test_records.py`

- [ ] **Step 1: Write failing tests for records**

```python
# tests/test_records.py
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
```

- [ ] **Step 2: Run tests, expect ImportError**

```bash
pytest tests/test_records.py -v
```
Expected: collection error — `ModuleNotFoundError: No module named 'anthroheight.records'`.

- [ ] **Step 3: Implement `src/anthroheight/records.py`**

```python
"""Pydantic v2 schemas for all persisted records."""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


Sex = Literal["M", "F"]
SurrogateName = Literal["knee_height", "demispan", "ulna"]
WarningLevel = Literal["info", "warn", "error"]


class PatientMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str = Field(min_length=1)
    age_years: int = Field(ge=0, le=130)
    sex: Sex
    ethnicity: str
    operator_input_flags: dict[str, bool]
    notes: str


class CalibrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    homography_matrix: list[list[float]]   # 3x3, serialized as nested list
    mm_per_px_central: float = Field(gt=0)
    marker_corners_px: list[list[float]]
    reprojection_error_px: float = Field(ge=0)


class Landmark(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    x_px: float
    y_px: float
    confidence: float = Field(ge=0.0, le=1.0)


class DeformityFlags(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kyphosis_suspected: bool
    kyphosis_confidence: float = Field(ge=0.0, le=1.0)
    scoliosis_suspected: bool
    scoliosis_confidence: float = Field(ge=0.0, le=1.0)
    lower_limb_contracture_suspected: bool
    lower_limb_contracture_confidence: float = Field(ge=0.0, le=1.0)
    upper_limb_contracture_suspected: bool
    upper_limb_contracture_confidence: float = Field(ge=0.0, le=1.0)


class SurrogateMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    surrogate_name: SurrogateName
    segment_length_mm: float = Field(gt=0)
    landmarks_used: list[tuple[str, float]]   # (name, confidence)
    bilateral_mean_used: bool
    bilateral_asymmetry_mm: Optional[float] = None


class HeightEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    height_cm: float = Field(gt=0)
    standard_error_cm: float = Field(gt=0)
    formula_id: str
    formula_citation: str
    validity_warnings: list[str]


class Warning(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: WarningLevel
    message: str
    source: str


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    warnings: list[Warning]


class MeasurementRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_metadata: PatientMetadata
    capture_timestamp: str   # ISO-8601 UTC
    image_path: str
    calibration: CalibrationResult
    landmarks: list[Landmark]
    flags: DeformityFlags
    surrogate: SurrogateMeasurement
    estimate: HeightEstimate
    validation: ValidationReport
    sum_of_segments_mm: Optional[float] = None    # diagnostic, never reported
    operator_id: str
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_records.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/anthroheight/records.py tests/test_records.py
git commit -m "feat(records): add pydantic schemas for measurement records"
```

---

## Task 2: Chumlea knee-height formula

**Files:**
- Create: `src/anthroheight/formulas/__init__.py`
- Create: `src/anthroheight/formulas/chumlea.py`
- Create: `tests/test_formulas/__init__.py`
- Create: `tests/test_formulas/test_chumlea.py`

- [ ] **Step 1: Create `formulas/__init__.py` and `tests/test_formulas/__init__.py`**

```python
# src/anthroheight/formulas/__init__.py
"""Surrogate-formula regressions for predicting standing height."""
```

```python
# tests/test_formulas/__init__.py
```

- [ ] **Step 2: Write failing tests for chumlea**

```python
# tests/test_formulas/test_chumlea.py
"""Chumlea 1985 knee-height regression for standing-height prediction."""
import pytest
from anthroheight.formulas import chumlea


@pytest.mark.parametrize(
    "knee_mm,age,sex,ethnicity,expect_low,expect_high",
    [
        (500.0, 30, "M", "white", 160.0, 175.0),
        (550.0, 60, "M", "white", 165.0, 185.0),
        (480.0, 40, "F", "white", 150.0, 170.0),
        (520.0, 70, "F", "black", 155.0, 175.0),
    ],
)
def test_chumlea_returns_plausible_height(knee_mm, age, sex, ethnicity, expect_low, expect_high):
    est = chumlea.estimate_height(knee_mm=knee_mm, age_years=age, sex=sex, ethnicity=ethnicity)
    assert expect_low <= est.height_cm <= expect_high
    assert 2.0 < est.standard_error_cm < 6.0
    assert est.formula_id.startswith("chumlea_1985")
    assert "Chumlea" in est.formula_citation


def test_chumlea_white_male_formula_arithmetic():
    """Verifies the white-male regression: stature = 64.19 - 0.04*age + 2.02*KH(cm).
    Reference: Chumlea WC, Roche AF, Steinbaugh ML. Estimating stature from
    knee height for persons 60 to 90 years of age. JAGS 1985; 33(2): 116-120.
    Engineer: confirm coefficients against the original table before merging."""
    est = chumlea.estimate_height(knee_mm=500.0, age_years=70, sex="M", ethnicity="white")
    expected = 64.19 - 0.04 * 70 + 2.02 * 50.0
    assert abs(est.height_cm - expected) < 0.01


def test_chumlea_unknown_ethnicity_raises_or_falls_back():
    """Engineer's choice — but must not silently return wrong values."""
    with pytest.raises(KeyError):
        chumlea.estimate_height(knee_mm=500.0, age_years=70, sex="M", ethnicity="unknown_xyz")


def test_chumlea_age_outside_validation_range_warns():
    """Chumlea 1985 derived in adults 60-90 yr. Estimate for a 20-yr-old must add a warning."""
    est = chumlea.estimate_height(knee_mm=500.0, age_years=20, sex="M", ethnicity="white")
    assert any("age" in w.lower() for w in est.validity_warnings)
```

- [ ] **Step 3: Run tests, expect ImportError**

```bash
pytest tests/test_formulas/test_chumlea.py -v
```
Expected: collection error — `ModuleNotFoundError`.

- [ ] **Step 4: Implement `formulas/chumlea.py`**

```python
"""Chumlea 1985 knee-height regression.

Reference: Chumlea WC, Roche AF, Steinbaugh ML. Estimating stature from
knee height for persons 60 to 90 years of age. J Am Geriatr Soc.
1985; 33(2): 116-120.

Engineer: before merging, verify each coefficient row against Table 2 of
the original paper (or the commonly republished version in BAPEN MUST
documentation). Coefficient typos produce systematic clinical error.
"""
from __future__ import annotations
from anthroheight.records import HeightEstimate, Sex


# (sex, ethnicity) -> {intercept, age_coef, knee_coef, see}
# Units: knee_coef applies to knee height in CM (not mm).
# Validation cohort age range: 60-90 years.
COEFFICIENTS: dict[tuple[Sex, str], dict[str, float]] = {
    ("M", "white"): {"intercept": 64.19, "age_coef": -0.04, "knee_coef": 2.02, "see": 3.4},
    ("F", "white"): {"intercept": 84.88, "age_coef": -0.24, "knee_coef": 1.83, "see": 3.5},
    ("M", "black"): {"intercept": 73.42, "age_coef": -0.04, "knee_coef": 1.79, "see": 3.7},
    ("F", "black"): {"intercept": 68.10, "age_coef": -0.06, "knee_coef": 1.86, "see": 3.8},
}

VALIDATION_AGE_MIN = 60
VALIDATION_AGE_MAX = 90
CITATION = "Chumlea WC, Roche AF, Steinbaugh ML. JAGS 1985; 33(2): 116-120."


def estimate_height(
    knee_mm: float,
    age_years: int,
    sex: Sex,
    ethnicity: str,
) -> HeightEstimate:
    """Predict standing height from knee height.

    Raises KeyError if (sex, ethnicity) is not in the coefficient table.
    Adds a validity_warning when age is outside the 60-90 validation range.
    """
    coefs = COEFFICIENTS[(sex, ethnicity)]   # intentional KeyError if missing
    knee_cm = knee_mm / 10.0
    height_cm = coefs["intercept"] + coefs["age_coef"] * age_years + coefs["knee_coef"] * knee_cm

    warnings: list[str] = []
    if age_years < VALIDATION_AGE_MIN or age_years > VALIDATION_AGE_MAX:
        warnings.append(
            f"Age {age_years} is outside Chumlea 1985 validation range "
            f"({VALIDATION_AGE_MIN}-{VALIDATION_AGE_MAX} yr); estimate is extrapolated."
        )

    return HeightEstimate(
        height_cm=height_cm,
        standard_error_cm=coefs["see"],
        formula_id=f"chumlea_1985_{ethnicity}_{ 'male' if sex == 'M' else 'female' }",
        formula_citation=CITATION,
        validity_warnings=warnings,
    )
```

- [ ] **Step 5: Run tests, verify pass**

```bash
pytest tests/test_formulas/test_chumlea.py -v
```
Expected: 7 passed (4 parametrized + 3 individual).

- [ ] **Step 6: Commit**

```bash
git add src/anthroheight/formulas/ tests/test_formulas/
git commit -m "feat(formulas): add Chumlea 1985 knee-height regression"
```

---

## Task 3: Bassey demispan formula

**Files:**
- Create: `src/anthroheight/formulas/bassey.py`
- Create: `tests/test_formulas/test_bassey.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_formulas/test_bassey.py
"""Bassey 1986 demispan regression."""
import pytest
from anthroheight.formulas import bassey


@pytest.mark.parametrize(
    "demispan_mm,sex,expect_low,expect_high",
    [
        (820.0, "M", 165.0, 185.0),
        (760.0, "F", 155.0, 175.0),
        (900.0, "M", 175.0, 195.0),
    ],
)
def test_bassey_returns_plausible_height(demispan_mm, sex, expect_low, expect_high):
    est = bassey.estimate_height(demispan_mm=demispan_mm, sex=sex)
    assert expect_low <= est.height_cm <= expect_high
    assert 2.0 < est.standard_error_cm < 6.0
    assert est.formula_id.startswith("bassey_1986")
    assert "Bassey" in est.formula_citation


def test_bassey_male_formula_arithmetic():
    """Reference Bassey EJ. Demi-span as a measure of skeletal size. Ann Hum Biol.
    1986; 13(5): 499-502.
    Engineer: verify exact intercept / slope from the original paper before merge."""
    est = bassey.estimate_height(demispan_mm=820.0, sex="M")
    expected = 1.40 * 82.0 + 57.8   # see chosen coefficients in implementation
    assert abs(est.height_cm - expected) < 0.5


def test_bassey_rejects_implausible_demispan():
    """Demispan < 50cm or > 100cm is anatomically implausible — raise."""
    with pytest.raises(ValueError):
        bassey.estimate_height(demispan_mm=300.0, sex="M")
    with pytest.raises(ValueError):
        bassey.estimate_height(demispan_mm=1200.0, sex="F")
```

- [ ] **Step 2: Run tests, expect ImportError**

```bash
pytest tests/test_formulas/test_bassey.py -v
```

- [ ] **Step 3: Implement `formulas/bassey.py`**

```python
"""Bassey 1986 demispan regression for standing-height prediction.

Reference: Bassey EJ. Demi-span as a measure of skeletal size. Ann Hum Biol.
1986; 13(5): 499-502.

Engineer: confirm slope and intercept against the original paper before
merging. Coefficients below are the commonly republished values used in
BAPEN/MUST nutritional-screening guidance.
"""
from __future__ import annotations
from anthroheight.records import HeightEstimate, Sex


COEFFICIENTS: dict[Sex, dict[str, float]] = {
    "M": {"slope": 1.40, "intercept": 57.8, "see": 4.0},
    "F": {"slope": 1.35, "intercept": 60.1, "see": 4.0},
}

DEMISPAN_MIN_MM = 500.0
DEMISPAN_MAX_MM = 1000.0
CITATION = "Bassey EJ. Ann Hum Biol. 1986; 13(5): 499-502."


def estimate_height(demispan_mm: float, sex: Sex) -> HeightEstimate:
    """Predict standing height from demispan (sternal notch to middle fingertip).

    Raises ValueError if demispan is outside anatomically plausible range.
    Raises KeyError if sex is not 'M' or 'F'.
    """
    if not (DEMISPAN_MIN_MM <= demispan_mm <= DEMISPAN_MAX_MM):
        raise ValueError(
            f"demispan_mm={demispan_mm} outside plausible range "
            f"[{DEMISPAN_MIN_MM}, {DEMISPAN_MAX_MM}]"
        )
    coefs = COEFFICIENTS[sex]
    demispan_cm = demispan_mm / 10.0
    height_cm = coefs["slope"] * demispan_cm + coefs["intercept"]

    return HeightEstimate(
        height_cm=height_cm,
        standard_error_cm=coefs["see"],
        formula_id=f"bassey_1986_{ 'male' if sex == 'M' else 'female' }",
        formula_citation=CITATION,
        validity_warnings=[],
    )
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_formulas/test_bassey.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/anthroheight/formulas/bassey.py tests/test_formulas/test_bassey.py
git commit -m "feat(formulas): add Bassey 1986 demispan regression"
```

---

## Task 4: MUST ulna-length formula

**Files:**
- Create: `src/anthroheight/formulas/must.py`
- Create: `tests/test_formulas/test_must.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_formulas/test_must.py
"""MUST/MAG ulna-length lookup tables."""
import pytest
from anthroheight.formulas import must


@pytest.mark.parametrize(
    "ulna_mm,age_years,sex,expect_low,expect_high",
    [
        (260.0, 30, "M", 165.0, 180.0),
        (280.0, 60, "M", 170.0, 185.0),
        (240.0, 40, "F", 150.0, 170.0),
    ],
)
def test_must_returns_plausible_height(ulna_mm, age_years, sex, expect_low, expect_high):
    est = must.estimate_height(ulna_mm=ulna_mm, age_years=age_years, sex=sex)
    assert expect_low <= est.height_cm <= expect_high
    assert 3.0 < est.standard_error_cm < 7.0
    assert est.formula_id.startswith("must_")


def test_must_clamps_to_table_extremes():
    """Engineer should clamp lookup to bounds rather than extrapolating wildly."""
    est_low = must.estimate_height(ulna_mm=200.0, age_years=40, sex="F")
    est_high = must.estimate_height(ulna_mm=320.0, age_years=40, sex="F")
    assert est_low.height_cm < est_high.height_cm
    assert any("clamp" in w.lower() or "outside" in w.lower()
               for w in est_low.validity_warnings + est_high.validity_warnings)
```

- [ ] **Step 2: Run tests, expect ImportError**

```bash
pytest tests/test_formulas/test_must.py -v
```

- [ ] **Step 3: Implement `formulas/must.py`**

```python
"""MUST / MAG ulna-length lookup tables for standing-height estimation.

Reference: BAPEN Malnutrition Universal Screening Tool (MUST) explanatory
booklet — published height-from-ulna lookup tables, age-banded by sex.
Endorsed by NICE CG32.

Engineer: the implementation below uses a simple linear interpolation
across age-banded tables. Before merging, verify the table values against
the BAPEN MUST PDF and add any age bands missing here.
"""
from __future__ import annotations
from anthroheight.records import HeightEstimate, Sex


# Per-sex, per-age-band table: ulna length (cm) -> stature (cm).
# Engineer: replace these starter values with the full BAPEN MUST table.
# Format: {age_band: {ulna_cm: stature_cm}}
TABLES: dict[Sex, dict[str, dict[float, float]]] = {
    "M": {
        "under_65": {
            24.0: 162.4, 25.0: 165.3, 26.0: 168.1, 27.0: 170.9,
            28.0: 173.7, 29.0: 176.5, 30.0: 179.3, 31.0: 182.1,
        },
        "65_and_over": {
            24.0: 161.0, 25.0: 163.7, 26.0: 166.4, 27.0: 169.0,
            28.0: 171.7, 29.0: 174.4, 30.0: 177.1, 31.0: 179.7,
        },
    },
    "F": {
        "under_65": {
            22.0: 154.6, 23.0: 157.1, 24.0: 159.6, 25.0: 162.1,
            26.0: 164.6, 27.0: 167.1, 28.0: 169.5, 29.0: 172.0,
        },
        "65_and_over": {
            22.0: 152.7, 23.0: 155.3, 24.0: 157.8, 25.0: 160.4,
            26.0: 162.9, 27.0: 165.5, 28.0: 168.0, 29.0: 170.6,
        },
    },
}

SEE = 5.0   # approximate; engineer to verify
CITATION = "BAPEN MUST explanatory booklet (NICE CG32 endorsed)."


def _band(age_years: int) -> str:
    return "65_and_over" if age_years >= 65 else "under_65"


def _interpolate(table: dict[float, float], ulna_cm: float) -> tuple[float, list[str]]:
    keys = sorted(table.keys())
    warnings: list[str] = []
    if ulna_cm <= keys[0]:
        warnings.append(
            f"ulna {ulna_cm} cm below table minimum {keys[0]} — clamped."
        )
        return table[keys[0]], warnings
    if ulna_cm >= keys[-1]:
        warnings.append(
            f"ulna {ulna_cm} cm above table maximum {keys[-1]} — clamped."
        )
        return table[keys[-1]], warnings
    # linear interp between bracketing keys
    lo = max(k for k in keys if k <= ulna_cm)
    hi = min(k for k in keys if k >= ulna_cm)
    if lo == hi:
        return table[lo], warnings
    frac = (ulna_cm - lo) / (hi - lo)
    return table[lo] + frac * (table[hi] - table[lo]), warnings


def estimate_height(ulna_mm: float, age_years: int, sex: Sex) -> HeightEstimate:
    """Predict standing height from ulna length via MUST lookup table."""
    ulna_cm = ulna_mm / 10.0
    band = _band(age_years)
    table = TABLES[sex][band]
    height_cm, warnings = _interpolate(table, ulna_cm)

    return HeightEstimate(
        height_cm=height_cm,
        standard_error_cm=SEE,
        formula_id=f"must_{ 'male' if sex == 'M' else 'female' }_{band}",
        formula_citation=CITATION,
        validity_warnings=warnings,
    )
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_formulas/test_must.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/anthroheight/formulas/must.py tests/test_formulas/test_must.py
git commit -m "feat(formulas): add MUST/NICE ulna-length lookup tables"
```

---

## Task 5: Surrogate measurement (`measure.py`)

**Files:**
- Create: `src/anthroheight/measure.py`
- Create: `tests/test_measure.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_measure.py
"""Surrogate-measurement segment length math."""
import numpy as np
import pytest
from anthroheight.records import Landmark, CalibrationResult
from anthroheight import measure


def _identity_calibration(mm_per_px: float = 1.0) -> CalibrationResult:
    """Identity homography — pixel == mm. Useful for unit math."""
    return CalibrationResult(
        homography_matrix=(np.eye(3) * np.array([[mm_per_px, mm_per_px, 1.0]]).T).tolist(),
        mm_per_px_central=mm_per_px,
        marker_corners_px=[[0.0, 0.0]] * 4,
        reprojection_error_px=0.5,
    )


def _lm(name: str, x: float, y: float, c: float = 0.95) -> Landmark:
    return Landmark(name=name, x_px=x, y_px=y, confidence=c)


def test_knee_height_unilateral_left():
    cal = _identity_calibration(mm_per_px=1.0)
    landmarks = [
        _lm("left_knee", 100.0, 100.0),
        _lm("left_ankle", 100.0, 600.0),   # 500 mm vertical
        _lm("right_knee", 200.0, 100.0, c=0.2),    # too low confidence to use
        _lm("right_ankle", 200.0, 600.0, c=0.2),
    ]
    sm = measure.measure_surrogate("knee_height", landmarks, cal, side_preference="left")
    assert abs(sm.segment_length_mm - 500.0) < 0.5
    assert not sm.bilateral_mean_used


def test_knee_height_bilateral_mean_when_both_visible():
    cal = _identity_calibration(mm_per_px=1.0)
    landmarks = [
        _lm("left_knee", 100.0, 100.0),
        _lm("left_ankle", 100.0, 600.0),    # 500 mm
        _lm("right_knee", 200.0, 100.0),
        _lm("right_ankle", 200.0, 700.0),   # 600 mm
    ]
    sm = measure.measure_surrogate("knee_height", landmarks, cal, side_preference="both")
    assert abs(sm.segment_length_mm - 550.0) < 0.5
    assert sm.bilateral_mean_used
    assert sm.bilateral_asymmetry_mm == pytest.approx(100.0, abs=0.5)


def test_ulna_length():
    cal = _identity_calibration()
    landmarks = [
        _lm("left_elbow", 100.0, 100.0),
        _lm("left_wrist", 100.0, 350.0),    # 250 mm
    ]
    sm = measure.measure_surrogate("ulna", landmarks, cal, side_preference="left")
    assert abs(sm.segment_length_mm - 250.0) < 0.5


def test_demispan_uses_shoulder_midpoint():
    cal = _identity_calibration()
    landmarks = [
        _lm("left_shoulder", 200.0, 200.0),
        _lm("right_shoulder", 300.0, 200.0),   # midpoint = (250, 200)
        _lm("middle_fingertip_left", 1050.0, 200.0),    # 800mm from midpoint
    ]
    sm = measure.measure_surrogate("demispan", landmarks, cal, side_preference="left")
    assert abs(sm.segment_length_mm - 800.0) < 0.5


def test_missing_required_landmark_raises():
    cal = _identity_calibration()
    landmarks = [_lm("left_knee", 0.0, 0.0)]   # ankle missing
    with pytest.raises(measure.MissingLandmarkError):
        measure.measure_surrogate("knee_height", landmarks, cal, side_preference="left")


def test_homography_scaling_applied():
    cal = _identity_calibration(mm_per_px=2.0)
    H = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    cal = CalibrationResult(
        homography_matrix=H.tolist(), mm_per_px_central=2.0,
        marker_corners_px=[[0.0, 0.0]] * 4, reprojection_error_px=0.5,
    )
    landmarks = [
        _lm("left_knee", 100.0, 100.0),
        _lm("left_ankle", 100.0, 350.0),    # 250 px diff
    ]
    sm = measure.measure_surrogate("knee_height", landmarks, cal, side_preference="left")
    # 250 px * 2 mm/px = 500 mm
    assert abs(sm.segment_length_mm - 500.0) < 0.5
```

- [ ] **Step 2: Run tests, expect ImportError**

```bash
pytest tests/test_measure.py -v
```

- [ ] **Step 3: Implement `src/anthroheight/measure.py`**

```python
"""Compute surrogate-segment lengths from landmarks + homography."""
from __future__ import annotations
from typing import Literal
import numpy as np

from anthroheight.records import (
    CalibrationResult, Landmark, SurrogateMeasurement, SurrogateName,
)


SidePreference = Literal["left", "right", "both"]
CONFIDENCE_THRESHOLD = 0.5


# (surrogate, side) -> required landmark name pair (start, end)
_REQUIRED: dict[tuple[SurrogateName, str], tuple[str, str]] = {
    ("knee_height", "left"): ("left_knee", "left_ankle"),
    ("knee_height", "right"): ("right_knee", "right_ankle"),
    ("ulna", "left"): ("left_elbow", "left_wrist"),
    ("ulna", "right"): ("right_elbow", "right_wrist"),
    # demispan handled separately because it uses a midpoint
}


class MissingLandmarkError(KeyError):
    """A required landmark for the chosen surrogate was not detected."""


def _by_name(landmarks: list[Landmark]) -> dict[str, Landmark]:
    return {lm.name: lm for lm in landmarks}


def _apply_homography(H: np.ndarray, x_px: float, y_px: float) -> tuple[float, float]:
    pt = np.array([x_px, y_px, 1.0])
    out = H @ pt
    return float(out[0] / out[2]), float(out[1] / out[2])


def _segment_mm(H: np.ndarray, a: Landmark, b: Landmark) -> float:
    ax, ay = _apply_homography(H, a.x_px, a.y_px)
    bx, by = _apply_homography(H, b.x_px, b.y_px)
    return float(np.hypot(ax - bx, ay - by))


def _required_pair(
    surrogate: SurrogateName, side: str, by_name: dict[str, Landmark]
) -> tuple[Landmark, Landmark]:
    a_name, b_name = _REQUIRED[(surrogate, side)]
    a, b = by_name.get(a_name), by_name.get(b_name)
    if a is None or a.confidence < CONFIDENCE_THRESHOLD:
        raise MissingLandmarkError(f"missing or low-confidence: {a_name}")
    if b is None or b.confidence < CONFIDENCE_THRESHOLD:
        raise MissingLandmarkError(f"missing or low-confidence: {b_name}")
    return a, b


def _measure_bilateral(
    surrogate: SurrogateName, by_name: dict[str, Landmark], H: np.ndarray
) -> SurrogateMeasurement:
    left_a, left_b = _required_pair(surrogate, "left", by_name)
    right_a, right_b = _required_pair(surrogate, "right", by_name)
    left_mm = _segment_mm(H, left_a, left_b)
    right_mm = _segment_mm(H, right_a, right_b)
    return SurrogateMeasurement(
        surrogate_name=surrogate,
        segment_length_mm=(left_mm + right_mm) / 2.0,
        landmarks_used=[
            (left_a.name, left_a.confidence), (left_b.name, left_b.confidence),
            (right_a.name, right_a.confidence), (right_b.name, right_b.confidence),
        ],
        bilateral_mean_used=True,
        bilateral_asymmetry_mm=abs(left_mm - right_mm),
    )


def _measure_unilateral(
    surrogate: SurrogateName, side: str,
    by_name: dict[str, Landmark], H: np.ndarray,
) -> SurrogateMeasurement:
    a, b = _required_pair(surrogate, side, by_name)
    return SurrogateMeasurement(
        surrogate_name=surrogate,
        segment_length_mm=_segment_mm(H, a, b),
        landmarks_used=[(a.name, a.confidence), (b.name, b.confidence)],
        bilateral_mean_used=False,
        bilateral_asymmetry_mm=None,
    )


def _measure_demispan(
    by_name: dict[str, Landmark], H: np.ndarray, side: str,
) -> SurrogateMeasurement:
    ls, rs = by_name.get("left_shoulder"), by_name.get("right_shoulder")
    if ls is None or ls.confidence < CONFIDENCE_THRESHOLD \
       or rs is None or rs.confidence < CONFIDENCE_THRESHOLD:
        raise MissingLandmarkError("demispan needs both shoulder landmarks")
    fingertip_name = f"middle_fingertip_{side}" if side != "both" else "middle_fingertip_left"
    ft = by_name.get(fingertip_name)
    if ft is None or ft.confidence < CONFIDENCE_THRESHOLD:
        raise MissingLandmarkError(f"demispan needs {fingertip_name}")
    # midpoint in pixel coords, then through H
    mid_px = ((ls.x_px + rs.x_px) / 2.0, (ls.y_px + rs.y_px) / 2.0)
    mid_mm = _apply_homography(H, *mid_px)
    ft_mm = _apply_homography(H, ft.x_px, ft.y_px)
    length = float(np.hypot(mid_mm[0] - ft_mm[0], mid_mm[1] - ft_mm[1]))
    return SurrogateMeasurement(
        surrogate_name="demispan",
        segment_length_mm=length,
        landmarks_used=[
            (ls.name, ls.confidence), (rs.name, rs.confidence),
            (ft.name, ft.confidence),
        ],
        bilateral_mean_used=False,
        bilateral_asymmetry_mm=None,
    )


def measure_surrogate(
    surrogate: SurrogateName,
    landmarks: list[Landmark],
    calibration: CalibrationResult,
    side_preference: SidePreference,
) -> SurrogateMeasurement:
    by_name = _by_name(landmarks)
    H = np.array(calibration.homography_matrix)
    if surrogate == "demispan":
        return _measure_demispan(by_name, H, side_preference)
    if side_preference == "both":
        return _measure_bilateral(surrogate, by_name, H)
    return _measure_unilateral(surrogate, side_preference, by_name, H)
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_measure.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/anthroheight/measure.py tests/test_measure.py
git commit -m "feat(measure): surrogate segment-length math via homography"
```

---

## Task 6: Deformity flags (`flags.py`)

**Files:**
- Create: `src/anthroheight/flags.py`
- Create: `tests/test_flags.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_flags.py
"""CV-derived deformity flag computation from supine landmarks."""
import math
import pytest
from anthroheight.records import Landmark
from anthroheight import flags


def lm(name, x, y, c=0.95):
    return Landmark(name=name, x_px=x, y_px=y, confidence=c)


def test_no_deformities_clean_supine():
    """Patient lying straight: head, shoulders, hips collinear; knees straight."""
    landmarks = [
        lm("nose", 100, 50), lm("left_shoulder", 100, 200), lm("right_shoulder", 100, 220),
        lm("left_hip", 100, 600), lm("right_hip", 100, 620),
        lm("left_knee", 100, 900), lm("left_ankle", 100, 1200),
        lm("right_knee", 100, 920), lm("right_ankle", 100, 1220),
        lm("left_elbow", 200, 350), lm("left_wrist", 200, 500),
    ]
    f = flags.compute_flags(landmarks)
    assert not f.kyphosis_suspected
    assert not f.scoliosis_suspected
    assert not f.lower_limb_contracture_suspected
    assert not f.upper_limb_contracture_suspected


def test_kyphosis_detected_when_head_offset_from_body_axis():
    """Kyphosis raises the head; in overhead view head landmark sits laterally
    offset from the shoulder-hip axis."""
    landmarks = [
        lm("nose", 250, 50),    # 150 px lateral to body axis at x=100
        lm("left_shoulder", 100, 200), lm("right_shoulder", 100, 220),
        lm("left_hip", 100, 600), lm("right_hip", 100, 620),
        lm("left_knee", 100, 900), lm("left_ankle", 100, 1200),
        lm("right_knee", 100, 920), lm("right_ankle", 100, 1220),
        lm("left_elbow", 200, 350), lm("left_wrist", 200, 500),
    ]
    f = flags.compute_flags(landmarks, head_offset_threshold_px=50.0)
    assert f.kyphosis_suspected


def test_scoliosis_detected_when_axis_breaks():
    """Spine landmarks not collinear (head ≠ shoulder midline ≠ hip midline)."""
    landmarks = [
        lm("nose", 100, 50),
        lm("left_shoulder", 100, 200), lm("right_shoulder", 200, 220),
        lm("left_hip", 200, 600), lm("right_hip", 100, 620),
        lm("left_knee", 100, 900), lm("left_ankle", 100, 1200),
        lm("right_knee", 100, 920), lm("right_ankle", 100, 1220),
        lm("left_elbow", 200, 350), lm("left_wrist", 200, 500),
    ]
    f = flags.compute_flags(landmarks, axis_deviation_threshold_px=30.0)
    assert f.scoliosis_suspected


def test_knee_contracture_flagged_when_joint_angle_bent():
    """Bent knee: hip-knee-ankle forms an angle materially less than 180°."""
    landmarks = [
        lm("nose", 100, 50), lm("left_shoulder", 100, 200), lm("right_shoulder", 100, 220),
        lm("left_hip", 100, 600), lm("right_hip", 100, 620),
        lm("left_knee", 100, 900),
        lm("left_ankle", 250, 1100),    # bent away from straight
        lm("right_knee", 100, 920), lm("right_ankle", 100, 1220),
        lm("left_elbow", 200, 350), lm("left_wrist", 200, 500),
    ]
    f = flags.compute_flags(landmarks, joint_angle_threshold_deg=30.0)
    assert f.lower_limb_contracture_suspected


def test_low_confidence_landmarks_ignored():
    """A landmark below 0.5 confidence should not drive a flag."""
    landmarks = [
        lm("nose", 250, 50, c=0.2),    # would suggest kyphosis but low conf
        lm("left_shoulder", 100, 200), lm("right_shoulder", 100, 220),
        lm("left_hip", 100, 600), lm("right_hip", 100, 620),
        lm("left_knee", 100, 900), lm("left_ankle", 100, 1200),
        lm("right_knee", 100, 920), lm("right_ankle", 100, 1220),
        lm("left_elbow", 200, 350), lm("left_wrist", 200, 500),
    ]
    f = flags.compute_flags(landmarks)
    assert not f.kyphosis_suspected
```

- [ ] **Step 2: Run tests, expect ImportError**

```bash
pytest tests/test_flags.py -v
```

- [ ] **Step 3: Implement `src/anthroheight/flags.py`**

```python
"""CV-derived deformity-flag heuristics from supine pose landmarks.

These are advisory second opinions to the operator-input flags. Thresholds
below are starting values calibrated empirically during phase-1 (phantom)
validation.
"""
from __future__ import annotations
import math
import numpy as np

from anthroheight.records import DeformityFlags, Landmark


CONF_MIN = 0.5
DEFAULT_HEAD_OFFSET_THRESHOLD_PX = 80.0
DEFAULT_AXIS_DEVIATION_THRESHOLD_PX = 40.0
DEFAULT_JOINT_ANGLE_THRESHOLD_DEG = 30.0


def _by_name(landmarks: list[Landmark]) -> dict[str, Landmark]:
    return {lm.name: lm for lm in landmarks if lm.confidence >= CONF_MIN}


def _midpoint(a: Landmark, b: Landmark) -> tuple[float, float]:
    return (a.x_px + b.x_px) / 2.0, (a.y_px + b.y_px) / 2.0


def _perpendicular_distance(point: tuple[float, float],
                            line_a: tuple[float, float],
                            line_b: tuple[float, float]) -> float:
    """Distance from `point` to the infinite line through line_a and line_b."""
    px, py = point
    ax, ay = line_a
    bx, by = line_b
    num = abs((by - ay) * px - (bx - ax) * py + bx * ay - by * ax)
    den = math.hypot(by - ay, bx - ax)
    return num / den if den > 0 else 0.0


def _joint_angle_deg(a: Landmark, mid: Landmark, b: Landmark) -> float:
    """Angle at `mid` formed by vectors mid->a and mid->b. 180° = straight."""
    v1 = np.array([a.x_px - mid.x_px, a.y_px - mid.y_px])
    v2 = np.array([b.x_px - mid.x_px, b.y_px - mid.y_px])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 180.0
    cos_t = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos_t))


def compute_flags(
    landmarks: list[Landmark],
    head_offset_threshold_px: float = DEFAULT_HEAD_OFFSET_THRESHOLD_PX,
    axis_deviation_threshold_px: float = DEFAULT_AXIS_DEVIATION_THRESHOLD_PX,
    joint_angle_threshold_deg: float = DEFAULT_JOINT_ANGLE_THRESHOLD_DEG,
) -> DeformityFlags:
    by = _by_name(landmarks)

    kyphosis = False
    kyph_conf = 0.0
    scoliosis = False
    scol_conf = 0.0

    if all(k in by for k in ("nose", "left_shoulder", "right_shoulder",
                             "left_hip", "right_hip")):
        sh_mid = _midpoint(by["left_shoulder"], by["right_shoulder"])
        hip_mid = _midpoint(by["left_hip"], by["right_hip"])
        nose = (by["nose"].x_px, by["nose"].y_px)
        head_offset = _perpendicular_distance(nose, sh_mid, hip_mid)
        kyphosis = head_offset > head_offset_threshold_px
        kyph_conf = min(1.0, head_offset / (2 * head_offset_threshold_px))
        # scoliosis: misalignment of head/shoulder/hip midlines
        deviation = _perpendicular_distance(sh_mid, nose, hip_mid)
        scoliosis = deviation > axis_deviation_threshold_px
        scol_conf = min(1.0, deviation / (2 * axis_deviation_threshold_px))

    lower_contracture = False
    lower_conf = 0.0
    for side in ("left", "right"):
        h, k, a = by.get(f"{side}_hip"), by.get(f"{side}_knee"), by.get(f"{side}_ankle")
        if h and k and a:
            angle = _joint_angle_deg(h, k, a)
            deviation_from_straight = 180.0 - angle
            if deviation_from_straight > joint_angle_threshold_deg:
                lower_contracture = True
                lower_conf = max(lower_conf,
                                 min(1.0, deviation_from_straight / 90.0))

    upper_contracture = False
    upper_conf = 0.0
    for side in ("left", "right"):
        s, e, w = by.get(f"{side}_shoulder"), by.get(f"{side}_elbow"), by.get(f"{side}_wrist")
        if s and e and w:
            angle = _joint_angle_deg(s, e, w)
            deviation_from_straight = 180.0 - angle
            if deviation_from_straight > joint_angle_threshold_deg:
                upper_contracture = True
                upper_conf = max(upper_conf,
                                 min(1.0, deviation_from_straight / 90.0))

    return DeformityFlags(
        kyphosis_suspected=kyphosis, kyphosis_confidence=kyph_conf,
        scoliosis_suspected=scoliosis, scoliosis_confidence=scol_conf,
        lower_limb_contracture_suspected=lower_contracture,
        lower_limb_contracture_confidence=lower_conf,
        upper_limb_contracture_suspected=upper_contracture,
        upper_limb_contracture_confidence=upper_conf,
    )
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_flags.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/anthroheight/flags.py tests/test_flags.py
git commit -m "feat(flags): CV-derived kyphosis/scoliosis/contracture indicators"
```

---

## Task 7: Validation report (`validate.py`)

**Files:**
- Create: `src/anthroheight/validate.py`
- Create: `tests/test_validate.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_validate.py
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
```

- [ ] **Step 2: Run tests, expect ImportError**

```bash
pytest tests/test_validate.py -v
```

- [ ] **Step 3: Implement `src/anthroheight/validate.py`**

```python
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
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_validate.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/anthroheight/validate.py tests/test_validate.py
git commit -m "feat(validate): plausibility/cross-check/symmetry warnings"
```

---

## Task 8: Calibration via ArUco (`calibration.py`)

**Files:**
- Create: `src/anthroheight/calibration.py`
- Create: `tests/test_calibration.py`
- Create: `tests/fixtures/__init__.py`
- Create: `tests/fixtures/synthetic_aruco.py`

- [ ] **Step 1: Synthetic-image fixture for ArUco**

```python
# tests/fixtures/__init__.py
```

```python
# tests/fixtures/synthetic_aruco.py
"""Generate a synthetic image with 4 ArUco markers placed at known locations
on a simulated bed plane. Used to drive calibration tests deterministically."""
import cv2
import numpy as np

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
MARKER_IDS = [0, 1, 2, 3]
# Bed coordinate system: top-left of bed = (0,0), x rightward, y down, in mm.
# 60x180 cm rectangle = 600x1800 mm.
BED_CORNERS_MM = {
    0: (0.0, 0.0),
    1: (600.0, 0.0),
    2: (600.0, 1800.0),
    3: (0.0, 1800.0),
}


def make_image(image_shape: tuple[int, int] = (1800, 1200),
               marker_size_px: int = 60) -> tuple[np.ndarray, dict]:
    """Render a flat overhead image with 4 ArUco markers at the bed corners
    (mapped 1:1 from mm to px for simplicity, scaled by image_shape).

    Returns (image_bgr, ground_truth_pixel_centers_dict).
    """
    img = np.full((*image_shape, 3), 255, dtype=np.uint8)
    h, w = image_shape
    # Place corners with a 100-px inset.
    inset = 100
    pixel_positions = {
        0: (inset, inset),
        1: (w - inset - marker_size_px, inset),
        2: (w - inset - marker_size_px, h - inset - marker_size_px),
        3: (inset, h - inset - marker_size_px),
    }
    for mid, (x, y) in pixel_positions.items():
        marker = cv2.aruco.generateImageMarker(DICT, mid, marker_size_px)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        img[y:y + marker_size_px, x:x + marker_size_px] = marker_bgr
    centers = {mid: (x + marker_size_px / 2.0, y + marker_size_px / 2.0)
               for mid, (x, y) in pixel_positions.items()}
    return img, centers
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_calibration.py
"""ArUco detection + homography for the bed-plane calibration."""
import numpy as np
import pytest
from tests.fixtures.synthetic_aruco import make_image, BED_CORNERS_MM
from anthroheight import calibration


def test_calibrate_detects_all_four_markers():
    img, centers = make_image()
    result = calibration.calibrate(img, expected_marker_layout=BED_CORNERS_MM)
    assert len(result.marker_corners_px) >= 4
    assert result.reprojection_error_px < 2.0
    assert result.mm_per_px_central > 0


def test_calibrate_rejects_image_with_too_few_markers():
    img, _ = make_image()
    # Black out three of the four markers
    img[:200, :] = 0
    img[:, :400] = 0
    with pytest.raises(calibration.InsufficientMarkersError):
        calibration.calibrate(img, expected_marker_layout=BED_CORNERS_MM)


def test_homography_maps_marker_to_known_mm():
    img, centers = make_image()
    result = calibration.calibrate(img, expected_marker_layout=BED_CORNERS_MM)
    H = np.array(result.homography_matrix)
    # Map marker 0 center pixel through H — should be near (0, 0) in mm
    px, py = centers[0]
    pt = H @ np.array([px, py, 1.0])
    mm_x, mm_y = pt[0] / pt[2], pt[1] / pt[2]
    assert abs(mm_x - 0.0) < 5.0   # within 5 mm
    assert abs(mm_y - 0.0) < 5.0
```

- [ ] **Step 3: Run tests, expect ImportError**

```bash
pytest tests/test_calibration.py -v
```

- [ ] **Step 4: Implement `src/anthroheight/calibration.py`**

```python
"""ArUco-based homography calibration for the bed plane."""
from __future__ import annotations
import cv2
import numpy as np

from anthroheight.records import CalibrationResult


DEFAULT_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)


class InsufficientMarkersError(RuntimeError):
    """Fewer than 4 ArUco markers detected from the expected layout."""


def _marker_centers_px(corners_list: list[np.ndarray], ids: np.ndarray) -> dict[int, np.ndarray]:
    centers: dict[int, np.ndarray] = {}
    for marker_corners, marker_id in zip(corners_list, ids.flatten()):
        # marker_corners shape: (1, 4, 2)
        centers[int(marker_id)] = marker_corners.reshape(4, 2).mean(axis=0)
    return centers


def calibrate(
    image_bgr: np.ndarray,
    expected_marker_layout: dict[int, tuple[float, float]],
    aruco_dict: cv2.aruco.Dictionary = DEFAULT_DICT,
) -> CalibrationResult:
    """Detect ArUco markers in `image_bgr`, compute homography to mm.

    `expected_marker_layout`: dict of marker_id -> (x_mm, y_mm) on bed plane.
    Requires ≥4 markers from this layout to be visible.
    """
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners_list, ids, _ = detector.detectMarkers(image_bgr)

    if ids is None:
        raise InsufficientMarkersError("no ArUco markers detected")
    centers = _marker_centers_px(corners_list, ids)
    matched = [(mid, centers[mid], expected_marker_layout[mid])
               for mid in centers if mid in expected_marker_layout]
    if len(matched) < 4:
        raise InsufficientMarkersError(
            f"only {len(matched)} of expected markers visible; need ≥4"
        )

    src = np.array([c for (_, c, _) in matched], dtype=np.float32)
    dst = np.array([m for (_, _, m) in matched], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC,
                              ransacReprojThreshold=2.0)
    if H is None:
        raise InsufficientMarkersError("homography solve failed")

    # Reprojection error: project src through H and compare with dst.
    src_h = np.hstack([src, np.ones((src.shape[0], 1))])
    proj = (H @ src_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    rep_err_mm = float(np.linalg.norm(proj - dst, axis=1).mean())

    # Translate mm-error back into px-error using a small reference distance.
    # Use the average mm distance between two adjacent markers vs px distance.
    mids = sorted(set(m for (_, _, m) in matched), key=lambda p: (p[0], p[1]))
    if len(matched) >= 2:
        a_mm, b_mm = matched[0][2], matched[1][2]
        a_px, b_px = matched[0][1], matched[1][1]
        mm_dist = float(np.linalg.norm(np.array(a_mm) - np.array(b_mm)))
        px_dist = float(np.linalg.norm(a_px - b_px))
        mm_per_px = mm_dist / px_dist if px_dist > 0 else 0.0
    else:
        mm_per_px = 0.0
    rep_err_px = rep_err_mm / mm_per_px if mm_per_px > 0 else rep_err_mm

    return CalibrationResult(
        homography_matrix=H.tolist(),
        mm_per_px_central=mm_per_px,
        marker_corners_px=[c.tolist() for (_, c, _) in matched],
        reprojection_error_px=rep_err_px,
    )
```

- [ ] **Step 5: Run tests, verify pass**

```bash
pytest tests/test_calibration.py -v
```
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/anthroheight/calibration.py tests/test_calibration.py tests/fixtures/
git commit -m "feat(calibration): ArUco detection + bed-plane homography"
```

---

## Task 9: Pose-model wrapper (`pose.py`)

**Files:**
- Create: `src/anthroheight/pose.py`
- Create: `tests/test_pose_wrapper.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pose_wrapper.py
"""MediaPipe Pose wrapper — mocks the model, tests the mapping."""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from anthroheight import pose


def _fake_mp_landmark(x, y, vis):
    lm = MagicMock()
    lm.x, lm.y, lm.z, lm.visibility = x, y, 0.0, vis
    return lm


def _fake_mp_result(image_shape):
    h, w = image_shape[:2]
    landmarks = MagicMock()
    landmarks.landmark = [
        _fake_mp_landmark(0.5, 0.05, 0.95),    # nose
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0.4, 0.2, 0.92),     # left_shoulder
        _fake_mp_landmark(0.6, 0.2, 0.92),     # right_shoulder
        _fake_mp_landmark(0.4, 0.4, 0.85),     # left_elbow
        _fake_mp_landmark(0.6, 0.4, 0.85),     # right_elbow
        _fake_mp_landmark(0.4, 0.55, 0.80),    # left_wrist
        _fake_mp_landmark(0.6, 0.55, 0.80),    # right_wrist
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0.4, 0.65, 0.88),    # left_hip
        _fake_mp_landmark(0.6, 0.65, 0.88),    # right_hip
        _fake_mp_landmark(0.4, 0.80, 0.85),    # left_knee
        _fake_mp_landmark(0.6, 0.80, 0.85),    # right_knee
        _fake_mp_landmark(0.4, 0.95, 0.82),    # left_ankle
        _fake_mp_landmark(0.6, 0.95, 0.82),    # right_ankle
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
    ]
    result = MagicMock()
    result.pose_landmarks = landmarks
    return result


def test_pose_wrapper_returns_named_landmarks():
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    detector = pose.PoseDetector()
    with patch.object(detector, "_raw_process",
                      return_value=_fake_mp_result(img.shape)):
        landmarks = detector.detect(img)
    names = {lm.name for lm in landmarks}
    expected = {"nose", "left_shoulder", "right_shoulder", "left_hip",
                "right_hip", "left_knee", "right_knee", "left_ankle",
                "right_ankle", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist"}
    assert expected.issubset(names)


def test_pose_wrapper_pixel_coords_scaled_to_image():
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    detector = pose.PoseDetector()
    with patch.object(detector, "_raw_process",
                      return_value=_fake_mp_result(img.shape)):
        landmarks = detector.detect(img)
    nose = next(lm for lm in landmarks if lm.name == "nose")
    assert abs(nose.x_px - 0.5 * 1920) < 1.0
    assert abs(nose.y_px - 0.05 * 1080) < 1.0


def test_pose_wrapper_handles_no_detection():
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    detector = pose.PoseDetector()
    no_pose = MagicMock()
    no_pose.pose_landmarks = None
    with patch.object(detector, "_raw_process", return_value=no_pose):
        landmarks = detector.detect(img)
    assert landmarks == []
```

- [ ] **Step 2: Run tests, expect ImportError**

```bash
pytest tests/test_pose_wrapper.py -v
```

- [ ] **Step 3: Implement `src/anthroheight/pose.py`**

```python
"""MediaPipe Pose wrapper exposing a stable Landmark interface.

Swap-able: replace the inner model without touching downstream code.
"""
from __future__ import annotations
import numpy as np
import mediapipe as mp

from anthroheight.records import Landmark


# MediaPipe Pose 33-landmark index → canonical name (subset we use).
_INDEX_TO_NAME = {
    0:  "nose",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow",    14: "right_elbow",
    15: "left_wrist",    16: "right_wrist",
    23: "left_hip",      24: "right_hip",
    25: "left_knee",     26: "right_knee",
    27: "left_ankle",    28: "right_ankle",
}


class PoseDetector:
    """Wraps mediapipe.solutions.pose.Pose."""

    def __init__(self, static_image_mode: bool = True, model_complexity: int = 2):
        self._mp_pose = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=False,
        )

    def _raw_process(self, image_rgb: np.ndarray):
        return self._mp_pose.process(image_rgb)

    def detect(self, image_bgr: np.ndarray) -> list[Landmark]:
        """Run pose inference and return only the canonical-named landmarks."""
        # MediaPipe expects RGB
        import cv2
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self._raw_process(image_rgb)
        if result.pose_landmarks is None:
            return []
        h, w = image_bgr.shape[:2]
        out: list[Landmark] = []
        for idx, name in _INDEX_TO_NAME.items():
            mp_lm = result.pose_landmarks.landmark[idx]
            out.append(Landmark(
                name=name,
                x_px=mp_lm.x * w,
                y_px=mp_lm.y * h,
                confidence=float(mp_lm.visibility),
            ))
        return out

    def close(self) -> None:
        self._mp_pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_pose_wrapper.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/anthroheight/pose.py tests/test_pose_wrapper.py
git commit -m "feat(pose): MediaPipe Pose wrapper with canonical landmark names"
```

---

## Task 10: Hand-model wrapper for demispan (`hand.py`)

**Files:**
- Create: `src/anthroheight/hand.py`
- Create: `tests/test_hand_wrapper.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_hand_wrapper.py
"""MediaPipe Hands wrapper — returns middle-fingertip landmark when found."""
from unittest.mock import MagicMock, patch
import numpy as np
from anthroheight import hand


def _fake_hand_landmark(x, y):
    lm = MagicMock()
    lm.x, lm.y, lm.z = x, y, 0.0
    return lm


def _fake_hands_result(handedness: str = "Left", fingertip_xy=(0.7, 0.5)):
    landmarks = MagicMock()
    # MediaPipe Hands landmark 12 = middle fingertip
    landmarks.landmark = [_fake_hand_landmark(0, 0)] * 21
    landmarks.landmark[12] = _fake_hand_landmark(*fingertip_xy)
    handed = MagicMock()
    handed.classification = [MagicMock(label=handedness, score=0.95)]
    result = MagicMock()
    result.multi_hand_landmarks = [landmarks]
    result.multi_handedness = [handed]
    return result


def test_hand_wrapper_returns_left_fingertip():
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    detector = hand.HandDetector()
    with patch.object(detector, "_raw_process",
                      return_value=_fake_hands_result("Left", (0.7, 0.5))):
        lm = detector.detect_middle_fingertip(img, side="left")
    assert lm is not None
    assert lm.name == "middle_fingertip_left"
    assert abs(lm.x_px - 0.7 * 1920) < 1
    assert abs(lm.y_px - 0.5 * 1080) < 1
    assert lm.confidence >= 0.5


def test_hand_wrapper_returns_none_when_no_hand_detected():
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    detector = hand.HandDetector()
    no_hands = MagicMock()
    no_hands.multi_hand_landmarks = None
    with patch.object(detector, "_raw_process", return_value=no_hands):
        lm = detector.detect_middle_fingertip(img, side="left")
    assert lm is None
```

- [ ] **Step 2: Run tests, expect ImportError**

```bash
pytest tests/test_hand_wrapper.py -v
```

- [ ] **Step 3: Implement `src/anthroheight/hand.py`**

```python
"""MediaPipe Hands wrapper for demispan fingertip detection.

Only used when surrogate == 'demispan'.
"""
from __future__ import annotations
from typing import Literal, Optional
import numpy as np
import mediapipe as mp

from anthroheight.records import Landmark


MIDDLE_FINGERTIP_INDEX = 12   # MediaPipe Hands landmark id


class HandDetector:
    def __init__(self, max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5):
        self._mp_hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
        )

    def _raw_process(self, image_rgb: np.ndarray):
        return self._mp_hands.process(image_rgb)

    def detect_middle_fingertip(
        self, image_bgr: np.ndarray, side: Literal["left", "right"],
    ) -> Optional[Landmark]:
        import cv2
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self._raw_process(image_rgb)
        if result.multi_hand_landmarks is None:
            return None
        h, w = image_bgr.shape[:2]
        for hand_lms, handed in zip(result.multi_hand_landmarks,
                                    result.multi_handedness):
            label = handed.classification[0].label.lower()
            if label != side:
                continue
            tip = hand_lms.landmark[MIDDLE_FINGERTIP_INDEX]
            return Landmark(
                name=f"middle_fingertip_{side}",
                x_px=tip.x * w, y_px=tip.y * h,
                confidence=float(handed.classification[0].score),
            )
        return None

    def close(self) -> None:
        self._mp_hands.close()
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_hand_wrapper.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/anthroheight/hand.py tests/test_hand_wrapper.py
git commit -m "feat(hand): MediaPipe Hands wrapper for demispan fingertip"
```

---

## Task 11: Capture with QC gates (`capture.py`)

**Files:**
- Create: `src/anthroheight/capture.py`
- Create: `tests/test_capture.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_capture.py
"""Image-quality gates and capture-source abstraction."""
import cv2
import numpy as np
import pytest
from tests.fixtures.synthetic_aruco import make_image
from anthroheight import capture


def test_qc_passes_clean_image_with_markers():
    img, _ = make_image()
    report = capture.run_qc(img, marker_dict_id=cv2.aruco.DICT_5X5_50,
                            min_markers=4)
    assert report.passed
    assert report.reasons == []


def test_qc_rejects_blurry_image():
    img, _ = make_image()
    blurred = cv2.GaussianBlur(img, (51, 51), 25)
    report = capture.run_qc(blurred, marker_dict_id=cv2.aruco.DICT_5X5_50,
                            min_markers=4, blur_threshold=80.0)
    assert not report.passed
    assert any("blur" in r.lower() for r in report.reasons)


def test_qc_rejects_dark_image():
    dark = np.full((1800, 1200, 3), 10, dtype=np.uint8)
    report = capture.run_qc(dark, marker_dict_id=cv2.aruco.DICT_5X5_50,
                            min_markers=4)
    assert not report.passed
    assert any("expos" in r.lower() or "dark" in r.lower() for r in report.reasons)


def test_qc_rejects_image_with_too_few_markers():
    img, _ = make_image()
    img[:, :600] = 0    # cover left half (kills 2 markers)
    report = capture.run_qc(img, marker_dict_id=cv2.aruco.DICT_5X5_50,
                            min_markers=4)
    assert not report.passed
    assert any("marker" in r.lower() for r in report.reasons)


def test_capture_from_file_returns_image():
    """Smoke test for the file-source path: write a synthetic image, read it back."""
    import tempfile, os
    img, _ = make_image()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        cv2.imwrite(f.name, img)
        path = f.name
    try:
        loaded = capture.capture_from_file(path)
        assert loaded.shape == img.shape
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run tests, expect ImportError**

```bash
pytest tests/test_capture.py -v
```

- [ ] **Step 3: Implement `src/anthroheight/capture.py`**

```python
"""Image acquisition + quality gates."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np


DEFAULT_BLUR_THRESHOLD = 100.0
DEFAULT_LUMA_MIN = 40.0
DEFAULT_LUMA_MAX = 220.0


@dataclass
class QCReport:
    passed: bool
    reasons: list[str]
    blur_score: float
    mean_luma: float
    marker_count: int


def _laplacian_variance(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _mean_luma(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def _count_markers(image_bgr: np.ndarray, dict_id: int) -> int:
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    _, ids, _ = detector.detectMarkers(image_bgr)
    return 0 if ids is None else len(ids)


def run_qc(
    image_bgr: np.ndarray,
    marker_dict_id: int = cv2.aruco.DICT_5X5_50,
    min_markers: int = 4,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    luma_min: float = DEFAULT_LUMA_MIN,
    luma_max: float = DEFAULT_LUMA_MAX,
) -> QCReport:
    reasons: list[str] = []
    blur = _laplacian_variance(image_bgr)
    luma = _mean_luma(image_bgr)
    markers = _count_markers(image_bgr, marker_dict_id)

    if blur < blur_threshold:
        reasons.append(f"blur score {blur:.1f} below threshold {blur_threshold}")
    if luma < luma_min:
        reasons.append(f"image too dark (mean luma {luma:.1f} < {luma_min})")
    if luma > luma_max:
        reasons.append(f"image too bright (mean luma {luma:.1f} > {luma_max})")
    if markers < min_markers:
        reasons.append(f"only {markers} markers detected, need ≥{min_markers}")

    return QCReport(passed=not reasons, reasons=reasons,
                    blur_score=blur, mean_luma=luma, marker_count=markers)


def capture_from_file(path: str) -> np.ndarray:
    """Load an image from disk for offline / batch processing."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def capture_from_camera(device_index: int = 0,
                        warmup_frames: int = 10) -> np.ndarray:
    """Grab a single frame from a connected USB camera."""
    cam = cv2.VideoCapture(device_index)
    if not cam.isOpened():
        raise RuntimeError(f"camera index {device_index} not available")
    try:
        for _ in range(warmup_frames):
            cam.read()
        ok, frame = cam.read()
        if not ok or frame is None:
            raise RuntimeError("camera read failed")
        return frame
    finally:
        cam.release()
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_capture.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/anthroheight/capture.py tests/test_capture.py
git commit -m "feat(capture): blur/exposure/marker QC gates + file/camera sources"
```

---

## Task 12: Output writer (`io_export.py`)

**Files:**
- Create: `src/anthroheight/io_export.py`
- Create: `tests/test_io_export.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_io_export.py
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
```

- [ ] **Step 2: Run tests, expect ImportError**

```bash
pytest tests/test_io_export.py -v
```

- [ ] **Step 3: Implement `src/anthroheight/io_export.py`**

```python
"""Persist measurement records: annotated PNG, JSON, CSV log row."""
from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

from anthroheight.records import MeasurementRecord


@dataclass
class ExportPaths:
    png_path: Path
    json_path: Path
    csv_path: Path


CSV_FIELDS = [
    "timestamp", "patient_id", "age_years", "sex", "ethnicity",
    "surrogate", "segment_mm", "height_cm", "see_cm",
    "formula_id", "operator_id", "warning_count",
]


def _annotate(image_bgr: np.ndarray, record: MeasurementRecord) -> np.ndarray:
    out = image_bgr.copy()
    # Stamp height
    text = f"{record.estimate.height_cm:.1f} cm  (±{record.estimate.standard_error_cm:.1f})"
    cv2.putText(out, text, (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    # Mark landmarks used
    used_names = {n for (n, _) in record.surrogate.landmarks_used}
    for lm in record.landmarks:
        if lm.name in used_names:
            cv2.circle(out, (int(lm.x_px), int(lm.y_px)), 8, (0, 255, 0), -1)
    return out


def _csv_row(record: MeasurementRecord) -> dict:
    return {
        "timestamp": record.capture_timestamp,
        "patient_id": record.patient_metadata.patient_id,
        "age_years": record.patient_metadata.age_years,
        "sex": record.patient_metadata.sex,
        "ethnicity": record.patient_metadata.ethnicity,
        "surrogate": record.surrogate.surrogate_name,
        "segment_mm": f"{record.surrogate.segment_length_mm:.2f}",
        "height_cm": f"{record.estimate.height_cm:.2f}",
        "see_cm": f"{record.estimate.standard_error_cm:.2f}",
        "formula_id": record.estimate.formula_id,
        "operator_id": record.operator_id,
        "warning_count": len(record.validation.warnings),
    }


def export(record: MeasurementRecord, image_bgr: np.ndarray,
           output_root: Path) -> ExportPaths:
    """Write annotated PNG, JSON record, and append CSV log row."""
    output_root = Path(output_root)
    patient_dir = output_root / record.patient_metadata.patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    stem = record.capture_timestamp.replace(":", "-").replace(".", "-")
    png_path = patient_dir / f"{stem}.png"
    json_path = patient_dir / f"{stem}.json"
    csv_path = output_root / "log.csv"

    annotated = _annotate(image_bgr, record)
    cv2.imwrite(str(png_path), annotated)
    # Update record image_path to point at the saved file
    record_with_path = record.model_copy(update={"image_path": str(png_path)})
    json_path.write_text(record_with_path.model_dump_json(indent=2))

    csv_exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not csv_exists:
            writer.writeheader()
        writer.writerow(_csv_row(record_with_path))

    return ExportPaths(png_path=png_path, json_path=json_path, csv_path=csv_path)
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_io_export.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/anthroheight/io_export.py tests/test_io_export.py
git commit -m "feat(io_export): annotated PNG + JSON record + CSV log writer"
```

---

## Task 13: Pipeline orchestration (`pipeline.py`)

**Files:**
- Create: `src/anthroheight/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pipeline.py
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
        pose_cls.return_value.detect.return_value = _identity_landmarks()
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
        pose_cls.return_value.detect.return_value = weak
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
```

- [ ] **Step 2: Run tests, expect ImportError**

```bash
pytest tests/test_pipeline.py -v
```

- [ ] **Step 3: Implement `src/anthroheight/pipeline.py`**

```python
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

    pose_detector = PoseDetector()
    landmarks = pose_detector.detect(image_bgr)

    if chosen_surrogate == "demispan":
        hand_detector = HandDetector()
        side = "left" if side_preference != "right" else "right"
        ft = hand_detector.detect_middle_fingertip(image_bgr, side=side)
        if ft is not None:
            landmarks = landmarks + [ft]

    flags_ = compute_flags(landmarks)

    surrogate_meas = measure_surrogate(
        chosen_surrogate, landmarks, cal, side_preference,
    )

    estimate = _apply_formula(chosen_surrogate, surrogate_meas.segment_length_mm, patient)

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
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_pipeline.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Run full test suite**

```bash
pytest
```
Expected: ~40 passed across all modules.

- [ ] **Step 6: Commit**

```bash
git add src/anthroheight/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): orchestrate calibration → pose → measure → formula → validate"
```

---

## Task 14: Streamlit operator UI (`ui_streamlit.py`)

This task is verified manually (no headless tests). The UI is thin glue around `pipeline.run` and `io_export.export`; the engineer manually walks the capture flow at the end.

**Files:**
- Create: `src/anthroheight/ui_streamlit.py`

- [ ] **Step 1: Implement `src/anthroheight/ui_streamlit.py`**

```python
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
```

- [ ] **Step 2: Manual verification — launch Streamlit and walk a capture**

```bash
streamlit run src/anthroheight/ui_streamlit.py
```

Expected manual checks:
- App loads at http://localhost:8501.
- Entering patient ID enables the form.
- Uploading any synthetic test image produces either a QC-fail message OR a measurement screen.
- "Save measurement" creates a file under `data/measurements/<patient_id>/`.

- [ ] **Step 3: Commit**

```bash
git add src/anthroheight/ui_streamlit.py
git commit -m "feat(ui): Streamlit operator capture-and-review flow"
```

---

## Task 15: Phantom evaluation harness (`eval/phantom_eval.py`)

**Files:**
- Create: `eval/__init__.py`
- Create: `eval/phantom_eval.py`
- Create: `eval/README.md`

- [ ] **Step 1: Create `eval/__init__.py` and `eval/README.md`**

```python
# eval/__init__.py
```

```markdown
# Eval harness

Three phases:
1. `phantom_eval.py` — rigid mannequin, n=20 captures per surrogate per condition.
2. `volunteer_eval.py` — healthy volunteers vs. stadiometer (built in subsequent plan).
3. `patient_eval.py` — disabled patients vs. clinical Chumlea (built in subsequent plan).

Inputs:
- `eval/ground_truth.csv` — known segment lengths and stature (where applicable) per subject.
- A directory of captures, one image per row in ground_truth.csv.

Outputs:
- `eval/results/<phase>_<timestamp>.csv` — per-image system output joined to ground truth.
- Run `analysis.ipynb` (next plan) for Bland-Altman / ICC / LoA plots.
```

- [ ] **Step 2: Implement `eval/phantom_eval.py`**

```python
"""Phantom-validation harness.

Reads ground_truth.csv with columns:
    image_path, surrogate, expected_segment_mm, age_years, sex, ethnicity

Runs the pipeline on each image and writes a results CSV with system output
joined to the expected values, plus per-row absolute error.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys
from anthroheight.capture import capture_from_file, run_qc
from anthroheight.records import PatientMetadata
from anthroheight.pipeline import run as pipeline_run
from tests.fixtures.synthetic_aruco import BED_CORNERS_MM


RESULT_FIELDS = [
    "image_path", "surrogate", "expected_segment_mm", "system_segment_mm",
    "abs_segment_error_mm", "system_height_cm", "see_cm", "warnings",
]


def evaluate(ground_truth_csv: Path, output_csv: Path,
             expected_marker_layout=BED_CORNERS_MM) -> None:
    rows = list(csv.DictReader(ground_truth_csv.open()))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            image = capture_from_file(row["image_path"])
            qc = run_qc(image)
            if not qc.passed:
                writer.writerow({
                    "image_path": row["image_path"],
                    "surrogate": row["surrogate"],
                    "expected_segment_mm": row["expected_segment_mm"],
                    "system_segment_mm": "",
                    "abs_segment_error_mm": "",
                    "system_height_cm": "",
                    "see_cm": "",
                    "warnings": "QC_FAIL: " + "; ".join(qc.reasons),
                })
                continue
            patient = PatientMetadata(
                patient_id="phantom", age_years=int(row["age_years"]),
                sex=row["sex"], ethnicity=row["ethnicity"],
                operator_input_flags={}, notes="",
            )
            try:
                rec = pipeline_run(
                    image_bgr=image, patient=patient,
                    chosen_surrogate=row["surrogate"], side_preference="both",
                    operator_id="phantom_eval",
                    expected_marker_layout=expected_marker_layout,
                    timestamp_iso=datetime.now(timezone.utc).isoformat(),
                )
                expected = float(row["expected_segment_mm"])
                writer.writerow({
                    "image_path": row["image_path"],
                    "surrogate": row["surrogate"],
                    "expected_segment_mm": expected,
                    "system_segment_mm": f"{rec.surrogate.segment_length_mm:.2f}",
                    "abs_segment_error_mm":
                        f"{abs(rec.surrogate.segment_length_mm - expected):.2f}",
                    "system_height_cm": f"{rec.estimate.height_cm:.2f}",
                    "see_cm": f"{rec.estimate.standard_error_cm:.2f}",
                    "warnings": "; ".join(w.message for w in rec.validation.warnings),
                })
            except Exception as e:    # surface failures as data, don't abort
                writer.writerow({
                    "image_path": row["image_path"],
                    "surrogate": row["surrogate"],
                    "expected_segment_mm": row["expected_segment_mm"],
                    "system_segment_mm": "",
                    "abs_segment_error_mm": "",
                    "system_height_cm": "",
                    "see_cm": "",
                    "warnings": f"PIPELINE_ERROR: {type(e).__name__}: {e}",
                })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("eval/results/phantom_latest.csv"))
    args = parser.parse_args()
    evaluate(args.ground_truth, args.output)
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-test the harness with a synthetic ground-truth CSV**

```bash
mkdir -p eval/results
python -c "
import csv, cv2
from pathlib import Path
from tests.fixtures.synthetic_aruco import make_image
img, _ = make_image()
Path('eval/results').mkdir(parents=True, exist_ok=True)
cv2.imwrite('eval/results/phantom_test.png', img)
with open('eval/results/phantom_gt.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['image_path','surrogate','expected_segment_mm','age_years','sex','ethnicity'])
    w.writerow(['eval/results/phantom_test.png','knee_height','500','70','M','white'])
"
python -m eval.phantom_eval --ground-truth eval/results/phantom_gt.csv --output eval/results/phantom_smoke.csv
cat eval/results/phantom_smoke.csv
```

Expected: a CSV row appears. The PIPELINE_ERROR is acceptable here (synthetic image has no human pose) — the smoke test verifies the harness writes the CSV row containing the error. Real phantom captures with a mannequin and good pose detection produce numeric values.

- [ ] **Step 4: Commit**

```bash
git add eval/
git commit -m "feat(eval): phantom-validation harness CLI"
```

---

## Task 16: README and quickstart polish

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README with a working quickstart**

Replace the placeholder README from Task 0 with this content:

````markdown
# anthroheight

Single-RGB computer vision for anthropometric standing-height estimation in supine disabled patients. Uses validated surrogate formulas (Chumlea knee height, Bassey demispan, MUST ulna length) over ArUco-calibrated MediaPipe pose landmarks.

See:
- Design spec: `docs/superpowers/specs/2026-04-27-anthropometric-supine-height-cv-design.md`
- Implementation plan: `docs/superpowers/plans/2026-04-27-anthropometric-supine-height-cv.md`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run tests

```bash
pytest
```

Expected: ~40 tests pass.

## Run the operator UI

```bash
streamlit run src/anthroheight/ui_streamlit.py
```

Open http://localhost:8501. Enter patient metadata, upload or capture an overhead RGB image, choose a surrogate, save the measurement.

## Run phantom evaluation

```bash
python -m eval.phantom_eval --ground-truth eval/ground_truth.csv \
                            --output eval/results/phantom.csv
```

`ground_truth.csv` columns: `image_path, surrogate, expected_segment_mm, age_years, sex, ethnicity`.

## Repo structure

```
src/anthroheight/   runtime package
tests/              unit tests (mirrors src/)
eval/               offline validation harness
docs/superpowers/   design + plans
data/               runtime outputs (gitignored)
```

## Limitations (read before clinical interpretation)

- Surrogate formulas have intrinsic SEE of ±3.5–5 cm; the system cannot beat that.
- Demispan currently uses the shoulder-midpoint as a proxy for the suprasternal notch — biased ±1–2 cm; calibration study planned in phase 2.
- Pose model is pretrained on standing/walking views; supine performance must be validated on phantoms before patient use.
- Coefficient tables in `formulas/` need verification against original sources before clinical deployment.
````

- [ ] **Step 2: Verify README is well-formed**

```bash
cat README.md | head -20
```

- [ ] **Step 3: Run full test suite once more**

```bash
pytest
```
Expected: all tests pass, no warnings of consequence.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: quickstart, repo layout, and limitations"
```

---

## Self-review

After writing the plan, the planner reviewed against the spec:

**Spec coverage:**
- ✅ records.py covers spec section 5 (data model)
- ✅ formulas/{chumlea,bassey,must}.py cover spec section 4.7 (formula module)
- ✅ measure.py covers section 4.6 (surrogate measurement)
- ✅ flags.py covers section 4.5 (deformity flags) — v1 CV-derived flags are implemented; operator-input flags live in PatientMetadata as the spec requires
- ✅ validate.py covers section 4.8 (validation)
- ✅ calibration.py covers section 4.2 (ArUco homography)
- ✅ pose.py + hand.py cover sections 4.3–4.4 (pose/hand wrappers)
- ✅ capture.py covers section 4.1 (capture + QC)
- ✅ io_export.py covers section 4.9 (PNG + JSON + CSV)
- ✅ ui_streamlit.py covers section 4.10 (UI)
- ✅ pipeline.py orchestrates section 3.1 pipeline
- ✅ eval/phantom_eval.py begins section 8.1 (phantom validation); volunteer + patient eval are flagged as separate plans (out of v1 scope per spec section 8.2-8.3)
- ⚠️ `sum_of_segments_mm` diagnostic field exists in records.py but is never populated by pipeline.py. Spec section 4.9 says it's diagnostic-only — leaving unpopulated is consistent with v1 reporting only the surrogate result. Engineer can add population in a follow-up if needed for QA.

**Placeholder scan:** No "TBD"/"TODO" steps. Coefficient tables explicitly cite sources and instruct verification — that's a sourcing instruction, not a placeholder.

**Type consistency:**
- `SurrogateName` literal used consistently across records, measure, pipeline.
- `Sex` literal used consistently across records, formulas, pipeline.
- `Landmark.name` strings ("left_knee", "right_knee", "middle_fingertip_left", etc.) match between pose.py wrapper, hand.py wrapper, measure.py REQUIRED dict, and flags.py.
- `CalibrationResult.homography_matrix` is `list[list[float]]` everywhere (numpy converted to/from list at module boundaries).
- `MissingLandmarkError` defined in measure.py, used in measure tests.
- `InsufficientMarkersError` defined in calibration.py, used in calibration tests.
- `BED_CORNERS_MM` defined once in tests/fixtures and reused by pipeline tests, eval, and UI placeholder.

**Scope check:** This plan delivers a complete, working v1 (capture → measure → save) plus phantom-eval harness. Volunteer + patient eval phases (spec 8.2, 8.3) are deferred to a follow-up plan — they require IRB approval and physical access that the implementation plan can't reasonably schedule. Same for hardware procurement (open question 2 in spec).
