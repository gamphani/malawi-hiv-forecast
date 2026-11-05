"""Prediction intervals from quantile gradient boosting, and the honesty test.

A quantile model predicts a chosen point of the range of outcomes, e.g. the 10th
percentile ("only 1 time in 10 should be lower than this"). Using the 10th and 90th
gives an 80% interval; the 2.5th and 97.5th give a 95% interval.

COVERAGE = share of real outcomes that fall inside the interval. A well-calibrated 80%
interval should contain about 80% of outcomes. If it contains 60%, it is overconfident.

Run:  python src/intervals.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from evaluate import FIRST_ORIGIN, HORIZON, STEP, fill_gaps
from features import build_row
from gbm import prepare, training_set
from metrics import wis

QUANTILES = [0.025, 0.10, 0.50, 0.90, 0.975]


def make_quantile_model(q):
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=q, max_iter=200, learning_rate=0.05,
        max_depth=3, min_samples_leaf=20, random_state=0,
    )


def interval_rolling_origin(df, first_origin=FIRST_ORIGIN, horizon=HORIZON, step=STEP):
    series = prepare(df)
    n = len(next(iter(series.values()))["y_raw"])
    rows = []
    for origin in range(first_origin, n - horizon + 1, step):
        filled = {d: (fill_gaps(s["y_raw"][:origin]), fill_gaps(s["tests_raw"][:origin]))
                  for d, s in series.items()}
        X, y = training_set(series, filled, origin, horizon)
        models = {q: make_quantile_model(q).fit(X, y) for q in QUANTILES}
        for d, s in series.items():
            yf, tf = filled[d]
            scale = np.mean(np.abs(np.diff(yf)))
            for k in range(horizon):
                actual = s["y_raw"][origin + k]
                if np.isnan(actual):
                    continue
                row, lvl = build_row(yf, tf, origin, k, s["moy"], s["rel"], s["urban"])
                preds = np.array([models[q].predict([row])[0] for q in QUANTILES])
                preds = np.sort(np.clip(preds, 0, None)) * lvl      # sorted: no crossing quantiles
                rows.append(dict(
                    district=d, origin=origin, horizon=k + 1, actual=actual, scale=scale,
                    q025=preds[0], q10=preds[1], q50=preds[2], q90=preds[3], q975=preds[4],
                    campaign=int(s["camp"][origin + k]), urban_rural=s["ur"], size_class=s["sc"],
                ))
    out = pd.DataFrame(rows)
    out["in80"] = ((out.actual >= out.q10) & (out.actual <= out.q90)).astype(int)
    out["in95"] = ((out.actual >= out.q025) & (out.actual <= out.q975)).astype(int)
    out["width80_scaled"] = (out.q90 - out.q10) / out.scale
    out["wis_scaled"] = wis(out.actual, out.q50, [out.q10, out.q025], [out.q90, out.q975], [0.2, 0.05]) / out.scale
    return out


def main():
    df = pd.read_csv("data/synthetic/district_months.csv")
    r = interval_rolling_origin(df)
    print(f"Forecasts scored: {len(r):,}")
    print(f"Overall coverage  80% interval: {r.in80.mean():.3f} (target 0.80)   95% interval: {r.in95.mean():.3f} (target 0.95)")
    print(f"Overall WIS (scaled, lower is better): {r.wis_scaled.mean():.3f}\n")
    for col in ["horizon", "urban_rural", "size_class", "campaign"]:
        t = r.groupby(col).agg(cov80=("in80", "mean"), cov95=("in95", "mean"),
                               wis=("wis_scaled", "mean"), n=("in80", "size")).round(3)
        print(f"By {col}:\n{t.to_string()}\n")


if __name__ == "__main__":
    main()
