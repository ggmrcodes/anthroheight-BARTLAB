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

DEFAULT_FACE_BLUR_RADIUS_PX = 120
FACE_BLUR_KERNEL = (51, 51)


def _blur_face(image_bgr: np.ndarray, record: MeasurementRecord,
               radius_px: int = DEFAULT_FACE_BLUR_RADIUS_PX) -> np.ndarray:
    """Heuristic privacy blur: Gaussian-blur a circle around the nose landmark.
    Radius scales with shoulder-width if both shoulder landmarks are detected;
    otherwise uses radius_px default. No-op if no nose landmark."""
    nose = next((lm for lm in record.landmarks if lm.name == "nose"), None)
    if nose is None:
        return image_bgr
    out = image_bgr.copy()

    # Scale radius by shoulder width if available (head ~= 1/3 shoulder width)
    ls = next((lm for lm in record.landmarks if lm.name == "left_shoulder"), None)
    rs = next((lm for lm in record.landmarks if lm.name == "right_shoulder"), None)
    if ls is not None and rs is not None:
        shoulder_w = float(np.hypot(ls.x_px - rs.x_px, ls.y_px - rs.y_px))
        radius = max(int(shoulder_w * 0.6), 40)
    else:
        radius = radius_px

    h, w = out.shape[:2]
    cx, cy = int(nose.x_px), int(nose.y_px)
    # Bounding box of the blur circle, clipped to image
    x0, y0 = max(cx - radius, 0), max(cy - radius, 0)
    x1, y1 = min(cx + radius, w), min(cy + radius, h)
    if x1 <= x0 or y1 <= y0:
        return out

    roi = out[y0:y1, x0:x1].copy()
    blurred_roi = cv2.GaussianBlur(roi, FACE_BLUR_KERNEL, 0)
    # Apply blur only inside the circle (mask)
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx - x0, cy - y0), radius, 255, -1)
    mask_3 = cv2.merge([mask, mask, mask]) // 255
    out[y0:y1, x0:x1] = roi * (1 - mask_3) + blurred_roi * mask_3
    return out


def _annotate(image_bgr: np.ndarray, record: MeasurementRecord) -> np.ndarray:
    out = image_bgr.copy()
    # cv2.putText uses Hershey fonts which are ASCII-only; "±" → "??".
    text = f"{record.estimate.height_cm:.1f} cm  (+/- {record.estimate.standard_error_cm:.1f})"
    cv2.putText(out, text, (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
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
           output_root: Path, blur_face: bool = True) -> ExportPaths:
    """Write annotated PNG, JSON record, and append CSV log row.

    `blur_face=True` (default) blurs a circular region around the nose
    landmark before annotation, for patient privacy. Set False to retain
    the unblurred face — operator must opt in.
    """
    output_root = Path(output_root)
    patient_dir = output_root / record.patient_metadata.patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    stem = record.capture_timestamp.replace(":", "-").replace(".", "-")
    png_path = patient_dir / f"{stem}.png"
    json_path = patient_dir / f"{stem}.json"
    csv_path = output_root / "log.csv"

    pre = _blur_face(image_bgr, record) if blur_face else image_bgr
    annotated = _annotate(pre, record)
    if not cv2.imwrite(str(png_path), annotated):
        raise IOError(f"cv2.imwrite failed for {png_path}")

    record_with_path = record.model_copy(update={"image_path": str(png_path)})
    json_path.write_text(record_with_path.model_dump_json(indent=2))

    csv_exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not csv_exists:
            writer.writeheader()
        writer.writerow(_csv_row(record_with_path))

    return ExportPaths(png_path=png_path, json_path=json_path, csv_path=csv_path)
