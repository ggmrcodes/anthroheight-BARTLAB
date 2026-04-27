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
    Requires >=4 markers from this layout to be visible.
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
            f"only {len(matched)} of expected markers visible; need >=4"
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
