"""Repair overconfident intervals with conformal calibration (CQR).

Problem: our quantile intervals under-cover (an "80%" interval held ~75% of outcomes).
Idea (Conformalized Quantile Regression, Romano et al. 2019):
  1. Fit the quantile models on OLDER data.
  2. On the most RECENT data whose outcomes are known (the calibration set), measure how
     far real outcomes fell outside the interval.
  3. Widen every future interval by that typical miss (or shrink it if intervals were too wide).

Time order is respected: calibration data are always the latest known months.
Caveat: the guarantee assumes the future looks like the calibration period. A campaign
nobody saw coming breaks that assumption, so campaign months stay under-covered.

Run:  python src/conformal.py
"""
import numpy as np
import pandas as pd

from evaluate import FIRST_ORIGIN, HORIZON, STEP, fill_gaps
from features import MIN_HISTORY, build_row
from gbm import prepare
from intervals import QUANTILES, make_quantile_model
from metrics import wis

CALIB_ORIGINS = 12   # the latest 12 forecast origins are held out to calibrate


def examples(series, filled, origin, horizon):
    """Like gbm.training_set, but also returns the origin of each example."""
    X, y, o_arr = [], [], []
    for d, s in series.items():
        yf, tf = filled[d]
        for o in range(MIN_HISTORY, origin):
            for k in range(horizon):
                if o + k >= origin or np.isnan(s["y_raw"][o + k]):
                    continue
                row, lvl = build_row(yf, tf, o, k, s["moy"], s["rel"], s["urban"])
                X.append(row)
                y.append(s["y_raw"][o + k] / lvl)
                o_arr.append(o)
    return np.array(X), np.array(y), np.array(o_arr)


def correction(scores, alpha):
    """Quantile of the calibration misses, with the small-sample (n+1) adjustment."""
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level))


def conformal_rolling_origin(df, first_origin=FIRST_ORIGIN, horizon=HORIZON, step=STEP):
    series = prepare(df)
    n = len(next(iter(series.values()))["y_raw"])
    rows = []
    for origin in range(first_origin, n - horizon + 1, step):
        filled = {d: (fill_gaps(s["y_raw"][:origin]), fill_gaps(s["tests_raw"][:origin]))
                  for d, s in series.items()}
        X, y, o_arr = examples(series, filled, origin, horizon)
        fit, cal = o_arr < origin - CALIB_ORIGINS, o_arr >= origin - CALIB_ORIGINS
        models = {q: make_quantile_model(q).fit(X[fit], y[fit]) for q in QUANTILES}
        P = {q: models[q].predict(X[cal]) for q in QUANTILES}
        # conformity score = how far the real value fell outside [lower, upper] (negative if inside)
        c80 = correction(np.maximum(P[0.10] - y[cal], y[cal] - P[0.90]), 0.20)
        c95 = correction(np.maximum(P[0.025] - y[cal], y[cal] - P[0.975]), 0.05)
        for d, s in series.items():
            yf, tf = filled[d]
            scale = np.mean(np.abs(np.diff(yf)))
            for k in range(horizon):
                actual = s["y_raw"][origin + k]
                if np.isnan(actual):
                    continue
                row, lvl = build_row(yf, tf, origin, k, s["moy"], s["rel"], s["urban"])
                p = np.sort([models[q].predict([row])[0] for q in QUANTILES])
                raw = np.clip(p, 0, None) * lvl
                cqr = np.clip([p[1] - c80, p[3] + c80, p[0] - c95, p[4] + c95], 0, None) * lvl
                rows.append(dict(
                    district=d, origin=origin, horizon=k + 1, actual=actual, scale=scale,
                    raw_lo80=raw[1], raw_hi80=raw[3], raw_lo95=raw[0], raw_hi95=raw[4], q50=raw[2],
                    cqr_lo80=cqr[0], cqr_hi80=cqr[1], cqr_lo95=cqr[2], cqr_hi95=cqr[3],
                    campaign=int(s["camp"][origin + k]), urban_rural=s["ur"], size_class=s["sc"],
                ))
    out = pd.DataFrame(rows)
    for tag in ("raw", "cqr"):
        out[f"{tag}_in80"] = ((out.actual >= out[f"{tag}_lo80"]) & (out.actual <= out[f"{tag}_hi80"])).astype(int)
        out[f"{tag}_in95"] = ((out.actual >= out[f"{tag}_lo95"]) & (out.actual <= out[f"{tag}_hi95"])).astype(int)
        out[f"{tag}_wis"] = wis(out.actual, out.q50, [out[f"{tag}_lo80"], out[f"{tag}_lo95"]],
                                [out[f"{tag}_hi80"], out[f"{tag}_hi95"]], [0.2, 0.05]) / out.scale
    return out


def summarise(r, by=None):
    g = r.groupby(by) if by else r.assign(all="all").groupby("all")
    return g.agg(raw_cov80=("raw_in80", "mean"), cqr_cov80=("cqr_in80", "mean"),
                 raw_cov95=("raw_in95", "mean"), cqr_cov95=("cqr_in95", "mean"),
                 raw_wis=("raw_wis", "mean"), cqr_wis=("cqr_wis", "mean"), n=("raw_in80", "size")).round(3)


def main():
    df = pd.read_csv("data/synthetic/district_months.csv")
    r = conformal_rolling_origin(df)
    print("Coverage targets: 80% -> 0.80, 95% -> 0.95. raw = uncalibrated, cqr = conformal.\n")
    print("Overall:\n" + summarise(r).to_string() + "\n")
    for col in ["campaign", "urban_rural", "size_class"]:
        print(f"By {col}:\n" + summarise(r, col).to_string() + "\n")


if __name__ == "__main__":
    main()
