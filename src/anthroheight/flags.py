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
DEFAULT_JOINT_ANGLE_THRESHOLD_DEG = 40.0


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
