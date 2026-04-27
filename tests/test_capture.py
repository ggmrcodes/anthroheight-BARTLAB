"""Image-quality gates and capture-source abstraction."""
import cv2
import numpy as np
import pytest
from tests.fixtures.synthetic_aruco import make_image
from anthroheight import capture


def test_qc_passes_clean_image_with_markers():
    img, _ = make_image()
    report = capture.run_qc(img, marker_dict_id=cv2.aruco.DICT_5X5_50,
                            min_markers=4)
    assert report.passed
    assert report.reasons == []


def test_qc_rejects_blurry_image():
    img, _ = make_image()
    blurred = cv2.GaussianBlur(img, (51, 51), 25)
    report = capture.run_qc(blurred, marker_dict_id=cv2.aruco.DICT_5X5_50,
                            min_markers=4, blur_threshold=80.0)
    assert not report.passed
    assert any("blur" in r.lower() for r in report.reasons)


def test_qc_rejects_dark_image():
    dark = np.full((1800, 1200, 3), 10, dtype=np.uint8)
    report = capture.run_qc(dark, marker_dict_id=cv2.aruco.DICT_5X5_50,
                            min_markers=4)
    assert not report.passed
    assert any("expos" in r.lower() or "dark" in r.lower() for r in report.reasons)


def test_qc_rejects_image_with_too_few_markers():
    img, _ = make_image()
    img[:, :600] = 0    # cover left half (kills 2 markers)
    report = capture.run_qc(img, marker_dict_id=cv2.aruco.DICT_5X5_50,
                            min_markers=4)
    assert not report.passed
    assert any("marker" in r.lower() for r in report.reasons)


def test_capture_from_file_returns_image():
    """Smoke test for the file-source path: write a synthetic image, read it back."""
    import tempfile, os
    img, _ = make_image()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        cv2.imwrite(f.name, img)
        path = f.name
    try:
        loaded = capture.capture_from_file(path)
        assert loaded.shape == img.shape
    finally:
        os.unlink(path)
