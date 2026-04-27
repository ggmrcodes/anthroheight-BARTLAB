"""Chumlea 1985 knee-height regression for standing-height prediction."""
import pytest
from anthroheight.formulas import chumlea


@pytest.mark.parametrize(
    "knee_mm,age,sex,ethnicity,expect_low,expect_high",
    [
        (500.0, 30, "M", "white", 160.0, 175.0),
        (550.0, 60, "M", "white", 165.0, 185.0),
        (480.0, 40, "F", "white", 150.0, 170.0),
        (520.0, 70, "F", "black", 155.0, 175.0),
    ],
)
def test_chumlea_returns_plausible_height(knee_mm, age, sex, ethnicity, expect_low, expect_high):
    est = chumlea.estimate_height(knee_mm=knee_mm, age_years=age, sex=sex, ethnicity=ethnicity)
    assert expect_low <= est.height_cm <= expect_high
    assert 2.0 < est.standard_error_cm < 6.0
    assert est.formula_id.startswith("chumlea_1985")
    assert "Chumlea" in est.formula_citation


def test_chumlea_white_male_formula_arithmetic():
    """Verifies the white-male regression: stature = 64.19 - 0.04*age + 2.02*KH(cm).
    Reference: Chumlea WC, Roche AF, Steinbaugh ML. Estimating stature from
    knee height for persons 60 to 90 years of age. JAGS 1985; 33(2): 116-120.
    Engineer: confirm coefficients against the original table before merging."""
    est = chumlea.estimate_height(knee_mm=500.0, age_years=70, sex="M", ethnicity="white")
    expected = 64.19 - 0.04 * 70 + 2.02 * 50.0
    assert abs(est.height_cm - expected) < 0.01


def test_chumlea_unknown_ethnicity_raises_or_falls_back():
    """Engineer's choice — but must not silently return wrong values."""
    with pytest.raises(KeyError):
        chumlea.estimate_height(knee_mm=500.0, age_years=70, sex="M", ethnicity="unknown_xyz")


def test_chumlea_age_outside_validation_range_warns():
    """Chumlea 1985 derived in adults 60-90 yr. Estimate for a 20-yr-old must add a warning."""
    est = chumlea.estimate_height(knee_mm=500.0, age_years=20, sex="M", ethnicity="white")
    assert any("age" in w.lower() for w in est.validity_warnings)
