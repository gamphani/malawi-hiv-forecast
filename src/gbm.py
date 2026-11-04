"""Gradient boosting forecaster, evaluated with the SAME rolling-origin test as the baselines.

One pooled model learns from all districts and all horizons together (horizon is an input).
At each origin it is retrained using only data that was known at that time.

Run:  python src/gbm.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from evaluate import FIRST_ORIGIN, HORIZON, STEP, fill_gaps, rolling_origin
from features import FEATURE_NAMES, MIN_HISTORY, build_row


def prepare(df):
    """Per-district arrays: raw diagnoses (with blanks), tests, calendar, attributes."""
    series = {}
    for district, g in df.groupby("district"):
        g = g.sort_values("month_index")
        series[district] = dict(
            y_raw=g["new_diagnoses"].to_numpy(float),
            tests_raw=g["tests"].to_numpy(float),
            moy=pd.PeriodIndex(g["month"], freq="M").month.to_numpy(),
            rel=float(g["relative_size"].iloc[0]),
            urban=int(g["urban_rural"].iloc[0] == "urban"),
            camp=g["campaign_month"].to_numpy(),
            ur=g["urban_rural"].iloc[0], sc=g["size_class"].iloc[0],
        )
    return series


def make_model():
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=20, random_state=0,
    )


def training_set(series, filled, origin, horizon):
    """Every (district, earlier origin, horizon) example whose answer was already known."""
    X, y = [], []
    for d, s in series.items():
        yf, tf = filled[d]
        for o in range(MIN_HISTORY, origin):
            for k in range(horizon):
                if o + k >= origin or np.isnan(s["y_raw"][o + k]):
                    continue
                row, lvl = build_row(yf, tf, o, k, s["moy"], s["rel"], s["urban"])
                X.append(row)
                y.append(s["y_raw"][o + k] / lvl)
    return np.array(X), np.array(y)


TESTING_FEATURES = ("tests_last", "tests_mean3", "tests_trend", "positivity_3m")


def gbm_rolling_origin(df, first_origin=FIRST_ORIGIN, horizon=HORIZON, step=STEP,
                       exclude=(), name="gbm"):
    """`exclude` lists feature names to hide from the model (used for the ablation test)."""
    keep = [i for i, f in enumerate(FEATURE_NAMES) if f not in exclude]
    series = prepare(df)
    n = len(next(iter(series.values()))["y_raw"])
    rows = []
    for origin in range(first_origin, n - horizon + 1, step):
        filled = {d: (fill_gaps(s["y_raw"][:origin]), fill_gaps(s["tests_raw"][:origin]))
                  for d, s in series.items()}
        X, y = training_set(series, filled, origin, horizon)
        model = make_model().fit(X[:, keep], y)
        for d, s in series.items():
            yf, tf = filled[d]
            scale = np.mean(np.abs(np.diff(yf)))
            for k in range(horizon):
                actual = s["y_raw"][origin + k]
                if np.isnan(actual):
                    continue
                row, lvl = build_row(yf, tf, origin, k, s["moy"], s["rel"], s["urban"])
                pred = max(float(model.predict([[row[i] for i in keep]])[0]), 0.0) * lvl
                rows.append(dict(
                    district=d, model=name, origin=origin, horizon=k + 1,
                    abs_error=abs(actual - pred), scale=scale, campaign=int(s["camp"][origin + k]),
                    urban_rural=s["ur"], size_class=s["sc"],
                ))
    out = pd.DataFrame(rows)
    out["scaled_error"] = out["abs_error"] / out["scale"]
    return out


def main():
    df = pd.read_csv("data/synthetic/district_months.csv")
    res = pd.concat([
        rolling_origin(df),
        gbm_rolling_origin(df),
        gbm_rolling_origin(df, exclude=TESTING_FEATURES, name="gbm_no_testing"),
    ], ignore_index=True)
    show = lambda col: res.pivot_table(index="model", columns=col, values="scaled_error").round(2).to_string()
    print("Overall MASE (lower is better):")
    print(res.groupby("model")["scaled_error"].mean().round(3).to_string(), "\n")
    print("By horizon:\n" + show("horizon"), "\n")
    print("By urban/rural:\n" + show("urban_rural"), "\n")
    print("By district size:\n" + show("size_class"), "\n")
    print("Campaign (1) vs normal (0) months:\n" + show("campaign"))


if __name__ == "__main__":
    main()
