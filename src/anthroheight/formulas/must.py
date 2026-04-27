"""MUST / MAG ulna-length lookup tables for standing-height estimation.

Reference: BAPEN Malnutrition Universal Screening Tool (MUST) explanatory
booklet — published height-from-ulna lookup tables, age-banded by sex.
Endorsed by NICE CG32.

Engineer: the implementation below uses a simple linear interpolation
across age-banded tables. Before merging, verify the table values against
the BAPEN MUST PDF and add any age bands missing here.
"""
from __future__ import annotations
from anthroheight.records import HeightEstimate, Sex


# Per-sex, per-age-band table: ulna length (cm) -> stature (cm).
# Engineer: replace these starter values with the full BAPEN MUST table.
# Format: {age_band: {ulna_cm: stature_cm}}
TABLES: dict[Sex, dict[str, dict[float, float]]] = {
    "M": {
        "under_65": {
            24.0: 162.4, 25.0: 165.3, 26.0: 168.1, 27.0: 170.9,
            28.0: 173.7, 29.0: 176.5, 30.0: 179.3, 31.0: 182.1,
        },
        "65_and_over": {
            24.0: 161.0, 25.0: 163.7, 26.0: 166.4, 27.0: 169.0,
            28.0: 171.7, 29.0: 174.4, 30.0: 177.1, 31.0: 179.7,
        },
    },
    "F": {
        "under_65": {
            22.0: 154.6, 23.0: 157.1, 24.0: 159.6, 25.0: 162.1,
            26.0: 164.6, 27.0: 167.1, 28.0: 169.5, 29.0: 172.0,
        },
        "65_and_over": {
            22.0: 152.7, 23.0: 155.3, 24.0: 157.8, 25.0: 160.4,
            26.0: 162.9, 27.0: 165.5, 28.0: 168.0, 29.0: 170.6,
        },
    },
}

SEE = 5.0   # approximate; engineer to verify
CITATION = "BAPEN MUST explanatory booklet (NICE CG32 endorsed)."


def _band(age_years: int) -> str:
    return "65_and_over" if age_years >= 65 else "under_65"


def _interpolate(table: dict[float, float], ulna_cm: float) -> tuple[float, list[str]]:
    keys = sorted(table.keys())
    warnings: list[str] = []
    if ulna_cm < keys[0]:
        warnings.append(
            f"ulna {ulna_cm} cm below table minimum {keys[0]} — clamped."
        )
        return table[keys[0]], warnings
    if ulna_cm > keys[-1]:
        warnings.append(
            f"ulna {ulna_cm} cm above table maximum {keys[-1]} — clamped."
        )
        return table[keys[-1]], warnings
    # linear interp between bracketing keys
    lo = max(k for k in keys if k <= ulna_cm)
    hi = min(k for k in keys if k >= ulna_cm)
    if lo == hi:
        return table[lo], warnings
    frac = (ulna_cm - lo) / (hi - lo)
    return table[lo] + frac * (table[hi] - table[lo]), warnings


def estimate_height(ulna_mm: float, age_years: int, sex: Sex) -> HeightEstimate:
    """Predict standing height from ulna length via MUST lookup table."""
    ulna_cm = ulna_mm / 10.0
    band = _band(age_years)
    table = TABLES[sex][band]
    height_cm, warnings = _interpolate(table, ulna_cm)

    return HeightEstimate(
        height_cm=height_cm,
        standard_error_cm=SEE,
        formula_id=f"must_{ 'male' if sex == 'M' else 'female' }_{band}",
        formula_citation=CITATION,
        validity_warnings=warnings,
    )
