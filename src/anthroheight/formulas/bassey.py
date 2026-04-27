"""Bassey 1986 demispan regression for standing-height prediction.

Reference: Bassey EJ. Demi-span as a measure of skeletal size. Ann Hum Biol.
1986; 13(5): 499-502.

Engineer: confirm slope and intercept against the original paper before
merging. Coefficients below are the commonly republished values used in
BAPEN/MUST nutritional-screening guidance.
"""
from __future__ import annotations
from anthroheight.records import HeightEstimate, Sex


COEFFICIENTS: dict[Sex, dict[str, float]] = {
    "M": {"slope": 1.40, "intercept": 57.8, "see": 4.0},
    "F": {"slope": 1.35, "intercept": 60.1, "see": 4.0},
}

DEMISPAN_MIN_MM = 500.0
DEMISPAN_MAX_MM = 1000.0
CITATION = "Bassey EJ. Ann Hum Biol. 1986; 13(5): 499-502."


def estimate_height(demispan_mm: float, sex: Sex) -> HeightEstimate:
    """Predict standing height from demispan (sternal notch to middle fingertip).

    Raises ValueError if demispan is outside anatomically plausible range.
    Raises KeyError if sex is not 'M' or 'F'.
    """
    if not (DEMISPAN_MIN_MM <= demispan_mm <= DEMISPAN_MAX_MM):
        raise ValueError(
            f"demispan_mm={demispan_mm} outside plausible range "
            f"[{DEMISPAN_MIN_MM}, {DEMISPAN_MAX_MM}]"
        )
    coefs = COEFFICIENTS[sex]
    demispan_cm = demispan_mm / 10.0
    height_cm = coefs["slope"] * demispan_cm + coefs["intercept"]

    return HeightEstimate(
        height_cm=height_cm,
        standard_error_cm=coefs["see"],
        formula_id=f"bassey_1986_{ 'male' if sex == 'M' else 'female' }",
        formula_citation=CITATION,
        validity_warnings=[],
    )
