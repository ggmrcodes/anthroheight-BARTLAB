"""MUST/MAG ulna-length lookup tables."""
import pytest
from anthroheight.formulas import must


@pytest.mark.parametrize(
    "ulna_mm,age_years,sex,expect_low,expect_high",
    [
        (260.0, 30, "M", 165.0, 180.0),
        (280.0, 60, "M", 170.0, 185.0),
        (240.0, 40, "F", 150.0, 170.0),
    ],
)
def test_must_returns_plausible_height(ulna_mm, age_years, sex, expect_low, expect_high):
    est = must.estimate_height(ulna_mm=ulna_mm, age_years=age_years, sex=sex)
    assert expect_low <= est.height_cm <= expect_high
    assert 3.0 < est.standard_error_cm < 7.0
    assert est.formula_id.startswith("must_")


def test_must_clamps_to_table_extremes():
    """Engineer should clamp lookup to bounds rather than extrapolating wildly."""
    est_low = must.estimate_height(ulna_mm=200.0, age_years=40, sex="F")
    est_high = must.estimate_height(ulna_mm=320.0, age_years=40, sex="F")
    assert est_low.height_cm < est_high.height_cm
    assert any("clamp" in w.lower() or "outside" in w.lower()
               for w in est_low.validity_warnings + est_high.validity_warnings)
