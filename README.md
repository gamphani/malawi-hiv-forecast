# Malawi HIV Diagnoses Forecasting

Explainable machine-learning forecasts of district-level new HIV diagnoses in Malawi, with uncertainty, for programme planners.

**Status:** early development. Uses SYNTHETIC data only until Ministry of Health access is granted.

## Why diagnoses, not infections
Infections cannot be observed. Diagnoses are recorded but depend on how much testing is done, so this project forecasts diagnoses as a proxy and models testing explicitly.

## Plan
1. Synthetic data generator with a data card
2. Baseline forecasts (seasonal naive, exponential smoothing, ARIMA)
3. Gradient boosting with lag features
4. Prediction intervals, coverage, weighted interval score
5. Equity checks (urban/rural, district size) and SHAP explanations

Author: Dave Phiri, Africa CDC AES-PHI fellow.
