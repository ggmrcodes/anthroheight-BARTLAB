# anthroheight

Single-RGB computer vision for anthropometric standing-height estimation in supine disabled patients. Uses validated surrogate formulas (Chumlea knee height, Bassey demispan, MUST ulna length) over ArUco-calibrated MediaPipe pose landmarks.

See:
- Design spec: `docs/superpowers/specs/2026-04-27-anthropometric-supine-height-cv-design.md`
- Implementation plan: `docs/superpowers/plans/2026-04-27-anthropometric-supine-height-cv.md`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run tests

```bash
pytest
```

Expected: ~66 tests pass.

## Run the operator UI

```bash
streamlit run src/anthroheight/ui_streamlit.py
```

Open http://localhost:8501. Enter patient metadata, upload or capture an overhead RGB image, choose a surrogate, save the measurement.

## Run phantom evaluation

```bash
python -m eval.phantom_eval --ground-truth eval/ground_truth.csv \
                            --output eval/results/phantom.csv
```

`ground_truth.csv` columns: `image_path, surrogate, expected_segment_mm, age_years, sex, ethnicity`.

## Repo structure

```
src/anthroheight/   runtime package
tests/              unit tests (mirrors src/)
eval/               offline validation harness
docs/superpowers/   design + plans
data/               runtime outputs (gitignored)
```

## Limitations (read before clinical interpretation)

- Surrogate formulas have intrinsic SEE of ±3.5–5 cm; the system cannot beat that.
- Demispan currently uses the shoulder-midpoint as a proxy for the suprasternal notch — biased ±1–2 cm; calibration study planned in phase 2.
- Pose model is pretrained on standing/walking views; supine performance must be validated on phantoms before patient use.
- Coefficient tables in `formulas/` need verification against original sources before clinical deployment.
- MediaPipe `solutions` API was dropped in 0.10.33; pose/hand wrappers use the Tasks API and require a downloaded `.task` model file (passed via `model_path=`) for real inference. Tests use mocks and don't require the model.
