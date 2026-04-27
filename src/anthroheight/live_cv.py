"""Live CV pipeline for the Streamlit operator UI.

Runs ArUco detect, MediaPipe Pose (every Nth frame), and QC checks on every
inbound webcam frame.  Annotates the frame with bounding boxes, an AR
blueprint wireframe of the 60x180 cm bed plane, pose skeleton, QC strip and
READY banner.  Auto-captures a clean snapshot once the QC conditions hold for
~1.5 seconds.

Designed to be wired into ``streamlit_webrtc.webrtc_streamer`` as the
``video_processor_factory``: the processor instance persists across frames
and is queried from the main Streamlit thread for the latest snapshot and
telemetry.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import threading
import time
from pathlib import Path
from typing import Optional

import av
import cv2
import numpy as np

from anthroheight.bed_layouts import BED_CORNERS_MM


# ---------------------------------------------------------------------------
# Color palette (BGR, matching .streamlit/config.toml tokens)
# ---------------------------------------------------------------------------
_ACCENT = (32, 37, 155)        # #9B2520 maroon
_INK = (26, 26, 26)            # #1A1A1A
_PAPER = (213, 226, 229)       # #E5E2D5 cream
_MUTED = (80, 85, 85)
_OK = (90, 170, 80)            # success green
_WARN = (50, 165, 220)         # amber

# ---------------------------------------------------------------------------
# QC thresholds for the READY indicator
# ---------------------------------------------------------------------------
_BLUR_OK_THRESHOLD = 60.0      # Laplacian variance on quarter-res frame
_BRIGHT_MIN = 50.0
_BRIGHT_MAX = 220.0
_REPROJ_OK_PX = 4.0
_READY_HOLD_SECONDS = 1.5

# Pose runs every Nth frame to keep latency low.  ~10 fps inference at 30 fps
# input is plenty for a live preview while keeping the recv loop responsive.
_POSE_EVERY_N_FRAMES = 3
_POSE_VIS_THRESHOLD = 0.3      # below this, joint is hidden (likely occluded)

# MediaPipe Pose 33-landmark connections (subset that reads as a skeleton).
_POSE_BONES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),    # shoulders / arms
    (11, 23), (12, 24), (23, 24),                          # torso
    (23, 25), (25, 27), (24, 26), (26, 28),                # legs
    (27, 31), (28, 32),                                    # feet
]


@dataclass
class FrameTelemetry:
    """Snapshot of what the live processor saw on the most recent frame.

    Read from the main Streamlit thread via ``LiveCVProcessor.telemetry()``.
    """
    frame_idx: int = 0
    n_markers: int = 0
    marker_ids: tuple[int, ...] = ()
    has_homography: bool = False
    mm_per_px: Optional[float] = None
    reprojection_error_px: Optional[float] = None
    blur_score: float = 0.0
    brightness: float = 0.0
    pose_detected: bool = False
    ready: bool = False
    ready_hold_seconds: float = 0.0


class LiveCVProcessor:
    """streamlit-webrtc video processor running real CV on every frame."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame_idx = 0

        self._aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_5X5_50
        )
        self._aruco = cv2.aruco.ArucoDetector(
            self._aruco_dict, cv2.aruco.DetectorParameters()
        )

        self._pose = None
        self._pose_load_failed = False
        self._last_pose_landmarks = None  # list[NormalizedLandmark] from Tasks API

        self._ready_hold_start: Optional[float] = None
        self._captured_image_bgr: Optional[np.ndarray] = None
        self._last_clean_frame: Optional[np.ndarray] = None

        self._telemetry = FrameTelemetry()

    # ------------------------------------------------------------------
    # Public read-only accessors (called from main Streamlit thread)
    # ------------------------------------------------------------------

    def telemetry(self) -> FrameTelemetry:
        with self._lock:
            return self._telemetry

    def take_captured_image(self) -> Optional[np.ndarray]:
        """Pop the auto-captured clean frame, if any.  Caller takes ownership."""
        with self._lock:
            img, self._captured_image_bgr = self._captured_image_bgr, None
            return img

    def take_current_frame(self) -> Optional[np.ndarray]:
        """Pop the most recent clean frame (manual snap fallback)."""
        with self._lock:
            img, self._last_clean_frame = self._last_clean_frame, None
            return img

    def reset_capture(self) -> None:
        with self._lock:
            self._captured_image_bgr = None
            self._ready_hold_start = None

    # ------------------------------------------------------------------
    # Lazy pose loader.  Tries env var, then bundled models/ dir.
    # Failure is cached so we don't retry the heavy load every 3rd frame.
    # ------------------------------------------------------------------

    def _ensure_pose(self) -> bool:
        if self._pose is not None:
            return True
        if self._pose_load_failed:
            return False
        model_path = os.environ.get("ANTHROHEIGHT_POSE_MODEL")
        if not model_path:
            cand = (
                Path(__file__).resolve().parent.parent.parent
                / "models" / "pose_landmarker_lite.task"
            )
            if cand.exists():
                model_path = str(cand)
        if not model_path or not Path(model_path).exists():
            self._pose_load_failed = True
            return False
        try:
            import mediapipe as mp
            opts = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=model_path
                ),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_poses=1,
                output_segmentation_masks=False,
            )
            self._pose = mp.tasks.vision.PoseLandmarker.create_from_options(
                opts
            )
            return True
        except Exception:
            self._pose_load_failed = True
            return False

    # ------------------------------------------------------------------
    # Main per-frame entry point (called by streamlit-webrtc on its thread)
    # ------------------------------------------------------------------

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        clean = img_bgr.copy()           # snapshot saved on auto-capture
        out = img_bgr                    # we draw on this in-place

        h, w = out.shape[:2]
        self._frame_idx += 1

        # 1. ArUco detection
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = self._aruco.detectMarkers(gray)
        marker_ids: tuple[int, ...] = ()
        H_mm_to_px: Optional[np.ndarray] = None
        mm_per_px: Optional[float] = None
        reproj_px: Optional[float] = None

        if ids is not None and len(ids) > 0:
            marker_ids = tuple(int(i) for i in ids.flatten())
            self._draw_marker_boxes(out, corners_list, ids)

            expected = set(BED_CORNERS_MM.keys())
            if expected.issubset(set(marker_ids)):
                centers_px, mm_pts = [], []
                for marker_corners, mid in zip(corners_list, ids.flatten()):
                    mid_i = int(mid)
                    if mid_i in BED_CORNERS_MM:
                        centers_px.append(
                            marker_corners.reshape(4, 2).mean(axis=0)
                        )
                        mm_pts.append(BED_CORNERS_MM[mid_i])
                src = np.array(centers_px, dtype=np.float32)
                dst = np.array(mm_pts, dtype=np.float32)
                H_px_to_mm, _ = cv2.findHomography(
                    src, dst, cv2.RANSAC, 2.0
                )
                if H_px_to_mm is not None:
                    H_mm_to_px = np.linalg.inv(H_px_to_mm)
                    mm_dist = float(np.linalg.norm(
                        np.array(mm_pts[0]) - np.array(mm_pts[1])
                    ))
                    px_dist = float(np.linalg.norm(src[0] - src[1]))
                    if px_dist > 0:
                        mm_per_px = mm_dist / px_dist
                    src_h = np.hstack(
                        [src, np.ones((src.shape[0], 1))]
                    )
                    proj = (H_px_to_mm @ src_h.T).T
                    proj = proj[:, :2] / proj[:, 2:3]
                    rep_err_mm = float(
                        np.linalg.norm(proj - dst, axis=1).mean()
                    )
                    if mm_per_px:
                        reproj_px = rep_err_mm / mm_per_px

        if H_mm_to_px is not None:
            self._draw_blueprint_wireframe(out, H_mm_to_px, w, h)

        # 2. QC: blur (Laplacian variance) + brightness on quarter-res copy
        small = cv2.resize(gray, (max(1, gray.shape[1] // 4),
                                  max(1, gray.shape[0] // 4)))
        blur_score = float(cv2.Laplacian(small, cv2.CV_64F).var())
        brightness = float(small.mean())

        # 3. Pose every Nth frame; cache the result for the in-between frames
        pose_detected = False
        if (self._frame_idx % _POSE_EVERY_N_FRAMES == 0
                and self._ensure_pose()):
            try:
                import mediapipe as mp
                rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self._pose.detect(mp_image)
                if result.pose_landmarks:
                    self._last_pose_landmarks = result.pose_landmarks[0]
                else:
                    self._last_pose_landmarks = None
            except Exception:
                self._last_pose_landmarks = None
        if self._last_pose_landmarks is not None:
            pose_detected = True
            self._draw_pose_skeleton(out, self._last_pose_landmarks, w, h)

        # 4. READY: cosmetic "all green" gating.  Pose is intentionally NOT
        # required because supine bodies are out-of-distribution for the
        # standing-pose model — gating on it would block the shutter forever.
        ready_now = (
            len(marker_ids) >= 4
            and H_mm_to_px is not None
            and reproj_px is not None and reproj_px < _REPROJ_OK_PX
            and blur_score > _BLUR_OK_THRESHOLD
            and _BRIGHT_MIN < brightness < _BRIGHT_MAX
        )
        now = time.monotonic()
        hold_secs = 0.0
        captured_this_frame = False
        with self._lock:
            self._last_clean_frame = clean
            if ready_now:
                if self._ready_hold_start is None:
                    self._ready_hold_start = now
                hold_secs = now - self._ready_hold_start
                if (hold_secs >= _READY_HOLD_SECONDS
                        and self._captured_image_bgr is None):
                    self._captured_image_bgr = clean
                    captured_this_frame = True
            else:
                self._ready_hold_start = None

        # 5. UI overlays: QC strip + READY banner
        self._draw_qc_strip(
            out, blur_score, brightness, len(marker_ids), reproj_px
        )
        self._draw_ready_banner(out, ready_now, hold_secs, captured_this_frame)

        # 6. Telemetry snapshot for the main thread
        with self._lock:
            self._telemetry = FrameTelemetry(
                frame_idx=self._frame_idx,
                n_markers=len(marker_ids),
                marker_ids=marker_ids,
                has_homography=H_mm_to_px is not None,
                mm_per_px=mm_per_px,
                reprojection_error_px=reproj_px,
                blur_score=blur_score,
                brightness=brightness,
                pose_detected=pose_detected,
                ready=ready_now,
                ready_hold_seconds=hold_secs,
            )

        return av.VideoFrame.from_ndarray(out, format="bgr24")

    # ------------------------------------------------------------------
    # Drawing helpers (in-place on BGR frame)
    # ------------------------------------------------------------------

    def _draw_marker_boxes(self, img, corners_list, ids):
        for marker_corners, mid in zip(corners_list, ids.flatten()):
            pts = marker_corners.reshape(4, 2).astype(np.int32)
            cv2.polylines(img, [pts], True, _ACCENT, 2, cv2.LINE_AA)
            tag_pt = tuple(pts[0])
            label = f"ID {int(mid)}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
            )
            cv2.rectangle(
                img,
                (tag_pt[0], tag_pt[1] - th - 7),
                (tag_pt[0] + tw + 6, tag_pt[1] - 1),
                _ACCENT, -1,
            )
            cv2.putText(
                img, label, (tag_pt[0] + 3, tag_pt[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, _PAPER, 1, cv2.LINE_AA,
            )

    def _draw_blueprint_wireframe(self, img, H_mm_to_px, w, h):
        """Project the 60x180 cm bed-plane rectangle + grid onto the image.

        Renders a blueprint-style overlay: outer rectangle, L-corner ticks,
        light grid lines every 200 mm down the long axis, and a faint
        centre-line.  All four anchors come from the live homography, so the
        wireframe sticks to the bed as the camera moves.
        """
        bed_w_mm, bed_l_mm = 600.0, 1800.0
        corners_mm = np.array([
            [0, 0], [bed_w_mm, 0],
            [bed_w_mm, bed_l_mm], [0, bed_l_mm],
        ], dtype=np.float32)
        outer = self._project(corners_mm, H_mm_to_px)
        cv2.polylines(
            img, [outer.astype(np.int32)], True, _ACCENT, 2, cv2.LINE_AA
        )

        # Cross-axis grid, every 200 mm
        for y_mm in np.arange(200.0, bed_l_mm, 200.0):
            line_mm = np.array(
                [[0, y_mm], [bed_w_mm, y_mm]], dtype=np.float32
            )
            lp = self._project(line_mm, H_mm_to_px)
            p0 = tuple(lp[0].astype(int))
            p1 = tuple(lp[1].astype(int))
            cv2.line(img, p0, p1, _ACCENT, 1, cv2.LINE_AA)

        # Centre-line down the long axis (subtle, for spine alignment)
        center_mm = np.array(
            [[bed_w_mm / 2, 0], [bed_w_mm / 2, bed_l_mm]], dtype=np.float32
        )
        cp = self._project(center_mm, H_mm_to_px)
        cv2.line(
            img, tuple(cp[0].astype(int)), tuple(cp[1].astype(int)),
            _MUTED, 1, cv2.LINE_AA,
        )

        # Heavy L-corner ticks for a "blueprint" feel
        tick_len = 80.0
        for cx, cy in [(0, 0), (bed_w_mm, 0),
                       (bed_w_mm, bed_l_mm), (0, bed_l_mm)]:
            sx = -1 if cx == 0 else 1
            sy = -1 if cy == 0 else 1
            tick_pts = np.array([
                [cx + sx * tick_len, cy],
                [cx, cy],
                [cx, cy + sy * tick_len],
            ], dtype=np.float32)
            tp = self._project(tick_pts, H_mm_to_px).astype(np.int32)
            cv2.polylines(img, [tp], False, _ACCENT, 3, cv2.LINE_AA)

    @staticmethod
    def _project(pts_mm: np.ndarray, H_mm_to_px: np.ndarray) -> np.ndarray:
        ph = np.hstack(
            [pts_mm, np.ones((pts_mm.shape[0], 1), dtype=np.float32)]
        )
        proj = (H_mm_to_px @ ph.T).T
        return proj[:, :2] / proj[:, 2:3]

    def _draw_pose_skeleton(self, img, lms, w, h):
        for a, b in _POSE_BONES:
            la, lb = lms[a], lms[b]
            if (la.visibility > _POSE_VIS_THRESHOLD
                    and lb.visibility > _POSE_VIS_THRESHOLD):
                pa = (int(la.x * w), int(la.y * h))
                pb = (int(lb.x * w), int(lb.y * h))
                cv2.line(img, pa, pb, _OK, 2, cv2.LINE_AA)
        for i in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            lm = lms[i]
            if lm.visibility > _POSE_VIS_THRESHOLD:
                p = (int(lm.x * w), int(lm.y * h))
                cv2.circle(img, p, 4, _PAPER, -1, cv2.LINE_AA)
                cv2.circle(img, p, 4, _OK, 1, cv2.LINE_AA)

    def _draw_qc_strip(self, img, blur_score, brightness,
                       n_markers, reproj_px):
        h, w = img.shape[:2]
        strip_h = 52
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h - strip_h), (w, h), _INK, -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

        # Layout four pills evenly across the bottom strip
        pills = [
            ("MARKERS", f"{n_markers}/4", n_markers >= 4),
            (
                "REPROJ",
                f"{reproj_px:.2f} px" if reproj_px is not None else "--",
                reproj_px is not None and reproj_px < _REPROJ_OK_PX,
            ),
            (
                "SHARPNESS", f"{blur_score:.0f}",
                blur_score > _BLUR_OK_THRESHOLD,
            ),
            (
                "EXPOSURE", f"{brightness:.0f}",
                _BRIGHT_MIN < brightness < _BRIGHT_MAX,
            ),
        ]
        col_w = w // len(pills)
        for i, (label, value, ok) in enumerate(pills):
            x = i * col_w + 14
            color = _OK if ok else _WARN
            cv2.putText(
                img, label, (x, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, _PAPER, 1, cv2.LINE_AA,
            )
            cv2.putText(
                img, value, (x, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA,
            )

    def _draw_ready_banner(self, img, ready_now, hold_secs, captured):
        h, w = img.shape[:2]
        banner_h = 36
        bx0, bx1 = w // 2 - 130, w // 2 + 130
        bg = _OK if (ready_now or captured) else _INK
        cv2.rectangle(img, (bx0, 8), (bx1, 8 + banner_h), bg, -1)
        cv2.rectangle(img, (bx0, 8), (bx1, 8 + banner_h), _PAPER, 1)

        if captured:
            label = "CAPTURED"
        elif ready_now:
            pct = min(1.0, hold_secs / _READY_HOLD_SECONDS)
            label = f"READY  {int(pct * 100)}%"
            prog_w = int((bx1 - bx0 - 12) * pct)
            cv2.rectangle(
                img,
                (bx0 + 6, 8 + banner_h - 6),
                (bx0 + 6 + prog_w, 8 + banner_h - 3),
                _PAPER, -1,
            )
        else:
            label = "AIM AT BED"

        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        tx = (bx0 + bx1) // 2 - tw // 2
        ty = 8 + banner_h // 2 + th // 2 - 2
        cv2.putText(
            img, label, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, _PAPER, 1, cv2.LINE_AA,
        )
