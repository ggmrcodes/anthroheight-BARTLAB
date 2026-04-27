"""MediaPipe Pose wrapper exposing a stable Landmark interface.

Swap-able: replace the inner model without touching downstream code.

Notes
-----
MediaPipe >=0.10.x ships only the Tasks API; the legacy ``solutions.pose``
module no longer exists.  Internally we use ``PoseLandmarker`` (Tasks API) but
adapt its result into the same duck-typed shape that the original
``solutions.pose`` API produced::

    result.pose_landmarks          # truthy / None
    result.pose_landmarks.landmark # indexable list of landmark objects
    landmark.x, .y, .visibility   # normalised coords + confidence

This means unit tests written against the old shape (mocking ``_raw_process``)
continue to pass unchanged, while real inference goes through the Tasks API.
Construction of the heavy model is deferred to the first ``detect`` call so
that ``PoseDetector()`` never fails in environments without a ``.task`` file
(e.g. CI / unit tests where ``_raw_process`` is fully mocked).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Any

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


# ---------------------------------------------------------------------------
# Adapter: wrap the Tasks API result to look like the legacy solutions result.
# ---------------------------------------------------------------------------

class _LandmarkList:
    """Minimal adapter that exposes ``.landmark[i]`` over a flat list."""

    def __init__(self, landmarks: list) -> None:
        self.landmark = landmarks


class _LegacyStyleResult:
    """Wraps a Tasks ``PoseLandmarkerResult`` with a ``.pose_landmarks``
    attribute that has ``.landmark[i].x / .y / .visibility``."""

    def __init__(self, tasks_result: Any) -> None:
        pose_lms = tasks_result.pose_landmarks
        if not pose_lms:
            self.pose_landmarks = None
        else:
            # Tasks API returns list[list[NormalizedLandmark]]; take pose 0.
            self.pose_landmarks = _LandmarkList(pose_lms[0])


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PoseDetector:
    """Wraps mediapipe.tasks.vision.PoseLandmarker.

    Parameters
    ----------
    model_path:
        Path to a ``pose_landmarker_*.task`` model bundle.  May be ``None``
        during unit tests when ``_raw_process`` is fully mocked.
    model_complexity:
        Kept for API parity with plans written against the legacy API.
        Ignored here — select the appropriate ``.task`` file via
        ``model_path`` instead.
    static_image_mode:
        Kept for API parity; always uses ``RunningMode.IMAGE``.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        model_complexity: int = 2,
        static_image_mode: bool = True,
    ) -> None:
        # If no explicit model_path, fall back to ANTHROHEIGHT_POSE_MODEL env var.
        import os
        if model_path is None:
            env = os.environ.get("ANTHROHEIGHT_POSE_MODEL")
            if env:
                model_path = env
        self._model_path: Optional[Path] = (
            Path(model_path) if model_path is not None else None
        )
        self._model_complexity = model_complexity
        self._landmarker: Optional[Any] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_landmarker(self) -> Any:
        """Instantiate the Tasks API PoseLandmarker (deferred)."""
        if self._model_path is None:
            raise FileNotFoundError(
                "No model_path provided to PoseDetector.  Download a "
                "pose_landmarker_*.task file from "
                "https://developers.google.com/mediapipe/solutions/vision/"
                "pose_landmarker and pass it via model_path=."
            )
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        options = PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self._model_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            output_segmentation_masks=False,
        )
        return PoseLandmarker.create_from_options(options)

    def _raw_process(self, image_rgb: np.ndarray) -> Any:
        """Run the Tasks API model and return a *legacy-shaped* result object.

        The returned object exposes::

            result.pose_landmarks          # None  OR  object with .landmark list
            result.pose_landmarks.landmark[i].x / .y / .visibility

        This is the same shape the old ``solutions.pose.Pose`` API produced, so
        ``detect`` (and unit-test mocks) work against a single stable contract.

        Patch this method in tests to inject fake results without loading model
        weights::

            with patch.object(detector, '_raw_process', return_value=fake):
                ...
        """
        if self._landmarker is None:
            self._landmarker = self._build_landmarker()
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb,
        )
        tasks_result = self._landmarker.detect(mp_image)
        return _LegacyStyleResult(tasks_result)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image_bgr: np.ndarray) -> list[Landmark]:
        """Run pose inference and return only the canonical-named landmarks.

        Parameters
        ----------
        image_bgr:
            BGR uint8 image as returned by ``cv2.imread``.

        Returns
        -------
        list[Landmark]
            Empty list when no pose is detected.
        """
        import cv2

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self._raw_process(image_rgb)

        if result.pose_landmarks is None:
            return []

        h, w = image_bgr.shape[:2]
        out: list[Landmark] = []
        for idx, name in _INDEX_TO_NAME.items():
            mp_lm = result.pose_landmarks.landmark[idx]
            out.append(
                Landmark(
                    name=name,
                    x_px=mp_lm.x * w,
                    y_px=mp_lm.y * h,
                    confidence=float(mp_lm.visibility),
                )
            )
        return out

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
