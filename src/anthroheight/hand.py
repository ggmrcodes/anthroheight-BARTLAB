"""MediaPipe Hands wrapper exposing a stable Landmark interface for demispan.

Swap-able: replace the inner model without touching downstream code.

Notes
-----
MediaPipe >=0.10.x ships only the Tasks API; the legacy ``solutions.hands``
module no longer exists.  Internally we use ``HandLandmarker`` (Tasks API) but
adapt its result into the same duck-typed shape that the original
``solutions.hands`` API produced::

    result.multi_hand_landmarks          # None  OR  list of hand-landmark groups
    result.multi_hand_landmarks[i].landmark   # indexable list of landmark objects
    landmark.x, .y, .z                        # normalised coords
    result.multi_handedness[i].classification[0].label  # "Left" / "Right"
    result.multi_handedness[i].classification[0].score  # confidence

This means unit tests written against the old shape (mocking ``_raw_process``)
continue to pass unchanged, while real inference goes through the Tasks API.
Construction of the heavy model is deferred to the first
``detect_middle_fingertip`` call so that ``HandDetector()`` never fails in
environments without a ``.task`` file (e.g. CI / unit tests where
``_raw_process`` is fully mocked).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Any

import numpy as np

import mediapipe as mp

from anthroheight.records import Landmark
from anthroheight.pose import _LandmarkList  # reusable adapter


# MediaPipe Hands landmark index for middle fingertip.
MIDDLE_FINGERTIP_INDEX = 12


# ---------------------------------------------------------------------------
# Adapters: wrap the Tasks API result to look like the legacy solutions result.
# ---------------------------------------------------------------------------

class _Handedness:
    """Minimal adapter exposing ``.classification[0].label`` / ``.score``."""

    class _Classification:
        def __init__(self, label: str, score: float) -> None:
            self.label = label
            self.score = score

    def __init__(self, label: str, score: float) -> None:
        self.classification = [self._Classification(label, score)]


class _LegacyHandsResult:
    """Wraps a Tasks ``HandLandmarkerResult`` with legacy-shaped attributes.

    Produces::

        result.multi_hand_landmarks   # None OR list[_LandmarkList]
        result.multi_handedness       # None OR list[_Handedness]
    """

    def __init__(self, tasks_result: Any) -> None:
        hand_lms = tasks_result.hand_landmarks
        handedness = tasks_result.handedness

        if not hand_lms:
            self.multi_hand_landmarks = None
            self.multi_handedness = None
        else:
            # Tasks API returns list[list[NormalizedLandmark]] per detected hand.
            self.multi_hand_landmarks = [
                _LandmarkList(lm_list) for lm_list in hand_lms
            ]
            # Tasks API returns list[list[Category]] for handedness per hand.
            self.multi_handedness = [
                _Handedness(
                    label=h[0].category_name,
                    score=h[0].score,
                )
                for h in handedness
            ]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HandDetector:
    """Wraps mediapipe.tasks.vision.HandLandmarker.

    Parameters
    ----------
    model_path:
        Path to a ``hand_landmarker.task`` model bundle.  May be ``None``
        during unit tests when ``_raw_process`` is fully mocked.
    max_num_hands:
        Maximum number of hands to detect (default 2).
    min_detection_confidence:
        Kept for API parity with plans written against the legacy API.
        Use ``model_path`` to select a more/less capable model instead.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
    ) -> None:
        self._model_path: Optional[Path] = (
            Path(model_path) if model_path is not None else None
        )
        self._max_num_hands = max_num_hands
        self._min_detection_confidence = min_detection_confidence
        self._landmarker: Optional[Any] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_landmarker(self) -> Any:
        """Instantiate the Tasks API HandLandmarker (deferred)."""
        if self._model_path is None:
            raise FileNotFoundError(
                "No model_path provided to HandDetector.  Download a "
                "hand_landmarker.task file from "
                "https://developers.google.com/mediapipe/solutions/vision/"
                "hand_landmarker and pass it via model_path=."
            )
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        options = HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self._model_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=self._max_num_hands,
            min_hand_detection_confidence=self._min_detection_confidence,
        )
        return HandLandmarker.create_from_options(options)

    def _raw_process(self, image_rgb: np.ndarray) -> Any:
        """Run the Tasks API model and return a *legacy-shaped* result object.

        The returned object exposes::

            result.multi_hand_landmarks          # None OR list of landmark groups
            result.multi_hand_landmarks[i].landmark[j].x / .y / .z
            result.multi_handedness[i].classification[0].label  # "Left"/"Right"
            result.multi_handedness[i].classification[0].score  # float 0-1

        This is the same shape the old ``solutions.hands.Hands`` API produced,
        so ``detect_middle_fingertip`` (and unit-test mocks) work against a
        single stable contract.

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
        return _LegacyHandsResult(tasks_result)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_middle_fingertip(
        self,
        image_bgr: np.ndarray,
        side: str,
    ) -> Optional[Landmark]:
        """Detect the middle fingertip on the requested hand side.

        Parameters
        ----------
        image_bgr:
            BGR uint8 image as returned by ``cv2.imread``.
        side:
            ``"left"`` or ``"right"`` (case-insensitive).

        Returns
        -------
        Landmark or None
            ``Landmark(name=f"middle_fingertip_{side}", ...)`` when the
            requested hand is detected; ``None`` otherwise.
        """
        import cv2

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self._raw_process(image_rgb)

        if result.multi_hand_landmarks is None:
            return None

        h, w = image_bgr.shape[:2]
        target_label = side.capitalize()  # "left" → "Left", "right" → "Right"

        for hand_lms, handedness in zip(
            result.multi_hand_landmarks, result.multi_handedness
        ):
            label = handedness.classification[0].label
            score = float(handedness.classification[0].score)
            if label != target_label:
                continue

            mp_lm = hand_lms.landmark[MIDDLE_FINGERTIP_INDEX]
            # NOTE: MediaPipe Hands Tasks API does not expose per-landmark visibility/
            # confidence. We use the handedness classification score as a proxy — it
            # reflects detector confidence in the hand as a whole, not the fingertip
            # specifically. Downstream validate.py treats this the same as pose
            # landmark confidence; this is a known approximation, not a bug.
            return Landmark(
                name=f"middle_fingertip_{side.lower()}",
                x_px=mp_lm.x * w,
                y_px=mp_lm.y * h,
                confidence=score,
            )

        return None

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
