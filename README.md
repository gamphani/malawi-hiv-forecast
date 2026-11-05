# Malawi HIV new-diagnosis forecasting (prototype v0.1)

Explainable machine-learning forecasts of **district-level monthly HIV new diagnoses**, with calibrated
uncertainty, for programme planners. Capstone project, Africa CDC AES-PHI fellowship (Dave Phiri).

> **SYNTHETIC DATA ONLY.** Every number here comes from invented data (`src/generate_data.py`).
> The results show the workflow works. They say nothing about Malawi.
> Diagnoses are a proxy for infections and depend on how much testing is done.

## What is in it
| Part | File |
|---|---|
| Synthetic data generator and data card | `src/generate_data.py`, `docs/data-card-synthetic.md` |
| Baselines (naive, seasonal naive, exponential smoothing) | `src/baselines.py` |
| Features (past only, no leakage) | `src/features.py` |
| Gradient boosting, quantile intervals, conformal calibration | `src/gbm.py`, `src/intervals.py`, `src/conformal.py` |
| Explanations (SHAP) | `src/explain.py` |
| Forecast service and planner dashboard | `src/forecast_service.py`, `app.py` |
| Results, model card, write-up | `docs/results.md`, `docs/model-card.md`, `docs/writeup.md` |

## Headline results (synthetic)
Overall MASE (lower is better; 1.0 is about as good as repeating last month): exponential smoothing 0.855, gradient boosting 0.755.
Calibrated 80% / 95% intervals cover 0.799 / 0.943 of outcomes. Testing campaigns are not predicted and remain the biggest source of error.
Details and limits: `docs/results.md`, `docs/model-card.md`.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/generate_data.py      # creates data/synthetic/district_months.csv
pytest -q                        # all tests should pass
python src/report.py             # writes docs/results.md (about 1 minute)
streamlit run app.py             # planner dashboard prototype
```

## Status and next steps
v0.1 is a working prototype on synthetic data. Next: research clearance and a data-sharing agreement,
re-run on real routine series, user-test the dashboard with planners, agree an owner and update plan.

## Licence
To be set at handover.
