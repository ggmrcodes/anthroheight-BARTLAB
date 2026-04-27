"""Canonical bed-mat marker layouts (mm coordinates).

Each layout maps an ArUco marker ID -> (x_mm, y_mm) on the bed surface,
with origin (0,0) at the head-end / left corner, x rightward, y down toward
the foot-end.

Used by:
    - calibration.calibrate(image, expected_marker_layout=...)
    - pipeline.run(..., expected_marker_layout=...)
    - the Streamlit UI and eval harness as their default layout

Defining the layouts here (in the runtime package) rather than in tests/
ensures the values are available to deployed code that has no access to the
test tree.  The synthetic-image test fixture imports from this module to
stay in sync.
"""
from __future__ import annotations


# Standard 60 cm x 180 cm bed-mat layout (4 corner markers, 5x5_50 dict).
BED_60x180_CM: dict[int, tuple[float, float]] = {
    0: (0.0, 0.0),         # head-left corner
    1: (600.0, 0.0),       # head-right corner
    2: (600.0, 1800.0),    # foot-right corner
    3: (0.0, 1800.0),      # foot-left corner
}

# Backwards-compatible alias (tests/fixtures/synthetic_aruco.py used this name).
BED_CORNERS_MM = BED_60x180_CM
