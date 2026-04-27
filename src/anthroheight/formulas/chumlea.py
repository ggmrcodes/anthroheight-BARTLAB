"""Chumlea 1985 knee-height regression.

Reference: Chumlea WC, Roche AF, Steinbaugh ML. Estimating stature from
knee height for persons 60 to 90 years of age. J Am Geriatr Soc.
1985; 33(2): 116-120.

Engineer: before merging, verify each coefficient row against Table 2 of
the original paper (or the commonly republished version in BAPEN MUST
documentation). Coefficient typos produce systematic clinical error.
"""
from __future__ import annotations
from anthroheight.records import HeightEstimate, Sex


# (sex, ethnicity) -> {intercept, age_coef, knee_coef, see}
# Units: knee_coef applies to knee height in CM (not mm).
# Validation cohort age range: 60-90 years.
COEFFICIENTS: dict[tuple[Sex, str], dict[str, float]] = {
    ("M", "white"): {"intercept": 64.19, "age_coef": -0.04, "knee_coef": 2.02, "see": 3.4},
    ("F", "white"): {"intercept": 84.88, "age_coef": -0.24, "knee_coef": 1.83, "see": 3.5},
    ("M", "black"): {"intercept": 73.42, "age_coef": -0.04, "knee_coef": 1.79, "see": 3.7},
    ("F", "black"): {"intercept": 68.10, "age_coef": -0.06, "knee_coef": 1.86, "see": 3.8},
}

VALIDATION_AGE_MIN = 60
VALIDATION_AGE_MAX = 90
CITATION = "Chumlea WC, Roche AF, Steinbaugh ML. JAGS 1985; 33(2): 116-120."


def estimate_height(
    knee_mm: float,
    age_years: int,
    sex: Sex,
    ethnicity: str,
) -> HeightEstimate:
    """Predict standing height from knee height.

    Raises KeyError if (sex, ethnicity) is not in the coefficient table.
    Adds a validity_warning when age is outside the 60-90 validation range.
    """
    coefs = COEFFICIENTS[(sex, ethnicity)]   # intentional KeyError if missing
    knee_cm = knee_mm / 10.0
    height_cm = coefs["intercept"] + coefs["age_coef"] * age_years + coefs["knee_coef"] * knee_cm

    warnings: list[str] = []
    if age_years < VALIDATION_AGE_MIN or age_years > VALIDATION_AGE_MAX:
        warnings.append(
            f"Age {age_years} is outside Chumlea 1985 validation range "
            f"({VALIDATION_AGE_MIN}-{VALIDATION_AGE_MAX} yr); estimate is extrapolated."
        )

    return HeightEstimate(
        height_cm=height_cm,
        standard_error_cm=coefs["see"],
        formula_id=f"chumlea_1985_{ethnicity}_{ 'male' if sex == 'M' else 'female' }",
        formula_citation=CITATION,
        validity_warnings=warnings,
    )
