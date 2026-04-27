"""MediaPipe Hands wrapper — returns middle-fingertip landmark when found."""
from unittest.mock import MagicMock, patch
import numpy as np
from anthroheight import hand


def _fake_hand_landmark(x, y):
    lm = MagicMock()
    lm.x, lm.y, lm.z = x, y, 0.0
    return lm


def _fake_hands_result(handedness: str = "Left", fingertip_xy=(0.7, 0.5)):
    landmarks = MagicMock()
    # MediaPipe Hands landmark 12 = middle fingertip
    landmarks.landmark = [_fake_hand_landmark(0, 0)] * 21
    landmarks.landmark[12] = _fake_hand_landmark(*fingertip_xy)
    handed = MagicMock()
    handed.classification = [MagicMock(label=handedness, score=0.95)]
    result = MagicMock()
    result.multi_hand_landmarks = [landmarks]
    result.multi_handedness = [handed]
    return result


def test_hand_wrapper_returns_left_fingertip():
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    detector = hand.HandDetector()
    with patch.object(detector, "_raw_process",
                      return_value=_fake_hands_result("Left", (0.7, 0.5))):
        lm = detector.detect_middle_fingertip(img, side="left")
    assert lm is not None
    assert lm.name == "middle_fingertip_left"
    assert abs(lm.x_px - 0.7 * 1920) < 1
    assert abs(lm.y_px - 0.5 * 1080) < 1
    assert lm.confidence >= 0.5


def test_hand_wrapper_returns_none_when_no_hand_detected():
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    detector = hand.HandDetector()
    no_hands = MagicMock()
    no_hands.multi_hand_landmarks = None
    with patch.object(detector, "_raw_process", return_value=no_hands):
        lm = detector.detect_middle_fingertip(img, side="left")
    assert lm is None
