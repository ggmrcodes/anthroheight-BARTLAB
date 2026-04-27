"""Image acquisition + quality gates."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np


DEFAULT_BLUR_THRESHOLD = 100.0
DEFAULT_LUMA_MIN = 40.0
DEFAULT_LUMA_MAX = 254.0


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
