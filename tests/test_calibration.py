"""ArUco detection + homography for the bed-plane calibration."""
import numpy as np
import pytest
from unittest.mock import patch
from tests.fixtures.synthetic_aruco import make_image, BED_CORNERS_MM
from anthroheight import calibration


def test_calibrate_detects_all_four_markers():
    img, centers = make_image()
    result = calibration.calibrate(img, expected_marker_layout=BED_CORNERS_MM)
    assert len(result.marker_corners_px) >= 4
    assert result.reprojection_error_px < 2.0
    assert result.mm_per_px_central > 0


def test_calibrate_rejects_image_with_too_few_markers():
    img, _ = make_image()
    # Black out three of the four markers
    img[:200, :] = 0
    img[:, :400] = 0
    with pytest.raises(calibration.InsufficientMarkersError):
        calibration.calibrate(img, expected_marker_layout=BED_CORNERS_MM)


def test_homography_maps_marker_to_known_mm():
    img, centers = make_image()
    result = calibration.calibrate(img, expected_marker_layout=BED_CORNERS_MM)
    H = np.array(result.homography_matrix)
    # Map marker 0 center pixel through H — should be near (0, 0) in mm
    px, py = centers[0]
    pt = H @ np.array([px, py, 1.0])
    mm_x, mm_y = pt[0] / pt[2], pt[1] / pt[2]
    assert abs(mm_x - 0.0) < 5.0   # within 5 mm
    assert abs(mm_y - 0.0) < 5.0


def test_calibrate_rejects_degenerate_marker_layout():
    """Two markers detected at the same pixel coords must raise, not silently
    return a bogus reprojection error in mm units."""
    img, _ = make_image()
    fake_corners = [
        np.array([[[100.0, 100.0], [110.0, 100.0], [110.0, 110.0], [100.0, 110.0]]]),
        np.array([[[100.0, 100.0], [110.0, 100.0], [110.0, 110.0], [100.0, 110.0]]]),  # duplicate
        np.array([[[200.0, 100.0], [210.0, 100.0], [210.0, 110.0], [200.0, 110.0]]]),
        np.array([[[200.0, 200.0], [210.0, 200.0], [210.0, 210.0], [200.0, 210.0]]]),
    ]
    fake_ids = np.array([[0], [1], [2], [3]])
    with patch("anthroheight.calibration.cv2.aruco.ArucoDetector") as detector_cls:
        detector_cls.return_value.detectMarkers.return_value = (fake_corners, fake_ids, None)
        with pytest.raises(calibration.InsufficientMarkersError):
            calibration.calibrate(img, expected_marker_layout=BED_CORNERS_MM)
