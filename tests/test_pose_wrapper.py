"""MediaPipe Pose wrapper — mocks the model, tests the mapping."""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from anthroheight import pose


def _fake_mp_landmark(x, y, vis):
    lm = MagicMock()
    lm.x, lm.y, lm.z, lm.visibility = x, y, 0.0, vis
    return lm


def _fake_mp_result(image_shape):
    h, w = image_shape[:2]
    landmarks = MagicMock()
    landmarks.landmark = [
        _fake_mp_landmark(0.5, 0.05, 0.95),    # nose
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0.4, 0.2, 0.92),     # left_shoulder
        _fake_mp_landmark(0.6, 0.2, 0.92),     # right_shoulder
        _fake_mp_landmark(0.4, 0.4, 0.85),     # left_elbow
        _fake_mp_landmark(0.6, 0.4, 0.85),     # right_elbow
        _fake_mp_landmark(0.4, 0.55, 0.80),    # left_wrist
        _fake_mp_landmark(0.6, 0.55, 0.80),    # right_wrist
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0.4, 0.65, 0.88),    # left_hip
        _fake_mp_landmark(0.6, 0.65, 0.88),    # right_hip
        _fake_mp_landmark(0.4, 0.80, 0.85),    # left_knee
        _fake_mp_landmark(0.6, 0.80, 0.85),    # right_knee
        _fake_mp_landmark(0.4, 0.95, 0.82),    # left_ankle
        _fake_mp_landmark(0.6, 0.95, 0.82),    # right_ankle
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
        _fake_mp_landmark(0, 0, 0),
    ]
    result = MagicMock()
    result.pose_landmarks = landmarks
    return result


def test_pose_wrapper_returns_named_landmarks():
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    detector = pose.PoseDetector()
    with patch.object(detector, "_raw_process",
                      return_value=_fake_mp_result(img.shape)):
        landmarks = detector.detect(img)
    names = {lm.name for lm in landmarks}
    expected = {"nose", "left_shoulder", "right_shoulder", "left_hip",
                "right_hip", "left_knee", "right_knee", "left_ankle",
                "right_ankle", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist"}
    assert expected.issubset(names)


def test_pose_wrapper_pixel_coords_scaled_to_image():
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    detector = pose.PoseDetector()
    with patch.object(detector, "_raw_process",
                      return_value=_fake_mp_result(img.shape)):
        landmarks = detector.detect(img)
    nose = next(lm for lm in landmarks if lm.name == "nose")
    assert abs(nose.x_px - 0.5 * 1920) < 1.0
    assert abs(nose.y_px - 0.05 * 1080) < 1.0


def test_pose_wrapper_handles_no_detection():
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    detector = pose.PoseDetector()
    no_pose = MagicMock()
    no_pose.pose_landmarks = None
    with patch.object(detector, "_raw_process", return_value=no_pose):
        landmarks = detector.detect(img)
    assert landmarks == []
