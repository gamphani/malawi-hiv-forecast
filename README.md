# Malawi HIV Diagnoses Forecasting

Explainable machine-learning forecasts of district-level new HIV diagnoses in Malawi, with calibrated uncertainty, for programme planners.

> **Status: development prototype. All results use SYNTHETIC data. They show that the workflow works and say nothing about Malawi.** Real Ministry of Health data are not part of this repository and must never be committed.

## Why diagnoses, not infections
Infections cannot be observed. Diagnoses are recorded, but they depend on how much testing is done (a testing campaign raises diagnoses even if infections do not change). This project forecasts diagnoses as a proxy, models testing, and says so on every output.

## What it does
1. Generates a documented synthetic district-month dataset (`src/generate_data.py`, `docs/data-card-synthetic.md`)
2. Baseline forecasts: naive, seasonal naive, exponential smoothing (`src/baselines.py`)
3. Gradient boosting with lag features, using only information known at forecast time (`src/features.py`, `src/gbm.py`)
4. Honest testing: rolling-origin evaluation, MASE, and an ablation test (`src/evaluate.py`, `src/metrics.py`)
5. Prediction intervals, coverage and weighted interval score (`src/intervals.py`)
6. Conformal calibration to repair overconfident intervals (`src/conformal.py`)
7. SHAP explanations in plain language (`src/explain.py`)
8. One results document with equity breakdowns (`src/report.py` writes `docs/results.md`)

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/generate_data.py      # creates data/synthetic/district_months.csv
python -m pytest -q tests        # automatic checks
python src/report.py             # about 1 minute; writes docs/results.md
```

## Headline findings (synthetic data)
- Gradient boosting was about 12% more accurate than the best baseline (MASE 0.755 vs 0.855). Hiding the testing inputs changed nothing, so the gain does not come from testing information.
- Rural and small districts are harder to forecast. Testing campaigns break every method.
- Uncalibrated 80% intervals held about 75% of outcomes. Conformal calibration brought overall coverage to target, but campaign months stayed under-covered.
- SHAP shows what the model uses, not what it needs, and does not prove cause.

## Limitations
Synthetic data; diagnoses are a proxy; settings were not tuned; forecasts from one district are correlated; results are not about Malawi.

## Roadmap
Ministry of Health data access and ethics route; planner-facing dashboard; user testing with programme planners; handover to a named owner.

## Standards followed
Reporting follows TRIPOD+AI headings; risk of bias self-assessed with PROBAST.

Author: Dave Phiri, Africa CDC AES-PHI fellow, Digital Health Division.
