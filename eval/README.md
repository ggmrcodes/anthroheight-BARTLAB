# Eval harness

Three phases:
1. `phantom_eval.py` — rigid mannequin, n=20 captures per surrogate per condition.
2. `volunteer_eval.py` — healthy volunteers vs. stadiometer (built in subsequent plan).
3. `patient_eval.py` — disabled patients vs. clinical Chumlea (built in subsequent plan).

Inputs:
- `eval/ground_truth.csv` — known segment lengths and stature (where applicable) per subject.
- A directory of captures, one image per row in ground_truth.csv.

Outputs:
- `eval/results/<phase>_<timestamp>.csv` — per-image system output joined to ground truth.
- Run `analysis.ipynb` (next plan) for Bland-Altman / ICC / LoA plots.
