"""Turn the tested models into the table a dashboard shows.

make_forecast() fits the quantile models on all data up to the LAST month, calibrates the
intervals (conformal, see conformal.py) and forecasts the next 6 months for every district.
All the logic lives here, where it is tested; the dashboard (app.py) only displays it.
"""
import numpy as np
import pandas as pd

from conformal import CALIB_ORIGINS, correction, examples
from evaluate import HORIZON, fill_gaps
from features import build_row
from gbm import prepare
from intervals import QUANTILES, make_quantile_model


def make_forecast(df, horizon=HORIZON):
    series = prepare(df)
    n = len(next(iter(series.values()))["y_raw"])
    months = pd.period_range(df["month"].min(), periods=n + horizon, freq="M")
    for s in series.values():
        s["moy"] = months.month.to_numpy()          # calendar extended into the forecast months
    filled = {d: (fill_gaps(s["y_raw"]), fill_gaps(s["tests_raw"])) for d, s in series.items()}

    X, y, o_arr = examples(series, filled, n, horizon)
    fit, cal = o_arr < n - CALIB_ORIGINS, o_arr >= n - CALIB_ORIGINS
    models = {q: make_quantile_model(q).fit(X[fit], y[fit]) for q in QUANTILES}
    P = {q: models[q].predict(X[cal]) for q in QUANTILES}
    c80 = correction(np.maximum(P[0.10] - y[cal], y[cal] - P[0.90]), 0.20)
    c95 = correction(np.maximum(P[0.025] - y[cal], y[cal] - P[0.975]), 0.05)

    missing = (df.sort_values("month_index").groupby("district")["reported"]
               .apply(lambda r: 1 - r.iloc[-12:].mean()))
    rows = []
    for d, s in series.items():
        yf, tf = filled[d]
        for k in range(horizon):
            row, lvl = build_row(yf, tf, n, k, s["moy"], s["rel"], s["urban"])
            p = np.sort([models[q].predict([row])[0] for q in QUANTILES])
            lo80, hi80 = p[1] - c80, p[3] + c80
            lo95, hi95 = p[0] - c95, p[4] + c95
            vals = np.clip([lo95, lo80, p[2], hi80, hi95], 0, None) * lvl
            rows.append(dict(
                district=d, month=str(months[n + k]), horizon=k + 1,
                lo95=vals[0], lo80=vals[1], forecast=vals[2], hi80=vals[3], hi95=vals[4],
                size_class=s["sc"], urban_rural=s["ur"], missing_share_12m=float(missing[d]),
            ))
    table = pd.DataFrame(rows)
    artifacts = dict(median_model=models[0.5], X_train=X, series=series, filled=filled, origin=n)
    return table, artifacts


def warnings_for(row):
    """Plain-language cautions for one district (row of the forecast table, any horizon)."""
    out = []
    if row["size_class"] == "small":
        out.append("Small district: counts are low and noisy, so forecasts are less reliable here.")
    if row["missing_share_12m"] > 0.10:
        out.append(f"{row['missing_share_12m']:.0%} of monthly reports were missing in the last 12 months.")
    out.append("Testing campaigns are not predicted. If one is planned or under way, expect more diagnoses than shown.")
    return out


def history(df, district, last=24):
    g = df[df["district"] == district].sort_values("month_index").tail(last)
    return g[["month", "new_diagnoses"]].rename(columns={"new_diagnoses": "actual"})


def explanations(artifacts):
    """Plain-language drivers of next month's forecast, per district. Empty if shap is missing."""
    try:
        from explain import explain_values, rows_for, sentence
    except ImportError:
        return {}
    try:
        X_now, levels, names = rows_for(artifacts["series"], artifacts["filled"], artifacts["origin"], k=0)
        X_train = artifacts["X_train"]
        background = X_train[np.random.default_rng(0).choice(len(X_train), min(100, len(X_train)), replace=False)]
        expl = explain_values(artifacts["median_model"], background, X_now)
    except ImportError:
        return {}
    return {d: sentence(expl.values[i], levels[i]) for i, d in enumerate(names)}
