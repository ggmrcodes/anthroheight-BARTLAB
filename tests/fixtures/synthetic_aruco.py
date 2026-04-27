"""Generate a synthetic image with 4 ArUco markers placed at known locations
on a simulated bed plane. Used to drive calibration tests deterministically."""
import cv2
import numpy as np

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
MARKER_IDS = [0, 1, 2, 3]
# Bed coordinate system: top-left of bed = (0,0), x rightward, y down, in mm.
# 60x180 cm rectangle = 600x1800 mm.
BED_CORNERS_MM = {
    0: (0.0, 0.0),
    1: (600.0, 0.0),
    2: (600.0, 1800.0),
    3: (0.0, 1800.0),
}


def make_image(image_shape: tuple[int, int] = (1800, 1200),
               marker_size_px: int = 60) -> tuple[np.ndarray, dict]:
    """Render a flat overhead image with 4 ArUco markers at the bed corners
    (mapped 1:1 from mm to px for simplicity, scaled by image_shape).

    Returns (image_bgr, ground_truth_pixel_centers_dict).
    """
    img = np.full((*image_shape, 3), 255, dtype=np.uint8)
    h, w = image_shape
    # Place corners with a 100-px inset.
    inset = 100
    pixel_positions = {
        0: (inset, inset),
        1: (w - inset - marker_size_px, inset),
        2: (w - inset - marker_size_px, h - inset - marker_size_px),
        3: (inset, h - inset - marker_size_px),
    }
    for mid, (x, y) in pixel_positions.items():
        marker = cv2.aruco.generateImageMarker(DICT, mid, marker_size_px)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        img[y:y + marker_size_px, x:x + marker_size_px] = marker_bgr
    centers = {mid: (x + marker_size_px / 2.0, y + marker_size_px / 2.0)
               for mid, (x, y) in pixel_positions.items()}
    return img, centers
