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
