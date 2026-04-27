"""Bassey 1986 demispan regression."""
import pytest
from anthroheight.formulas import bassey


@pytest.mark.parametrize(
    "demispan_mm,sex,expect_low,expect_high",
    [
        (820.0, "M", 165.0, 185.0),
        (760.0, "F", 155.0, 175.0),
        (900.0, "M", 175.0, 195.0),
    ],
)
def test_bassey_returns_plausible_height(demispan_mm, sex, expect_low, expect_high):
    est = bassey.estimate_height(demispan_mm=demispan_mm, sex=sex)
    assert expect_low <= est.height_cm <= expect_high
    assert 2.0 < est.standard_error_cm < 6.0
    assert est.formula_id.startswith("bassey_1986")
    assert "Bassey" in est.formula_citation


def test_bassey_male_formula_arithmetic():
    """Reference Bassey EJ. Demi-span as a measure of skeletal size. Ann Hum Biol.
    1986; 13(5): 499-502.
    Engineer: verify exact intercept / slope from the original paper before merge."""
    est = bassey.estimate_height(demispan_mm=820.0, sex="M")
    expected = 1.40 * 82.0 + 57.8   # see chosen coefficients in implementation
    assert abs(est.height_cm - expected) < 0.5


def test_bassey_rejects_implausible_demispan():
    """Demispan < 50cm or > 100cm is anatomically implausible — raise."""
    with pytest.raises(ValueError):
        bassey.estimate_height(demispan_mm=300.0, sex="M")
    with pytest.raises(ValueError):
        bassey.estimate_height(demispan_mm=1200.0, sex="F")
