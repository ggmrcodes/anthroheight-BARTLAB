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
