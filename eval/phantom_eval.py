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
