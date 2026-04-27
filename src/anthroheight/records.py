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
