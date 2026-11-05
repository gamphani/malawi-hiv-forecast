"""Explain forecasts with SHAP.

SHAP (SHapley Additive exPlanations) answers: "for THIS forecast, how much did each input
push the number up or down compared with a typical case?" The contributions add up:

    typical forecast  +  sum of all contributions  =  this forecast

Important: SHAP explains the MODEL, not the real world. It does not prove cause.

Our model predicts a ratio (diagnoses / district level), so we multiply each contribution
by the district's level to express it in cases.

Run:  python src/explain.py
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evaluate import fill_gaps  # noqa: E402
from features import FEATURE_NAMES, build_row  # noqa: E402
from gbm import make_model, prepare, training_set  # noqa: E402

ORIGIN = 60        # explain forecasts made with data up to month 59
HORIZON = 6
LABELS = {
    "y_last": "last month's diagnoses",
    "y_last2": "diagnoses 2 months ago",
    "y_last3": "diagnoses 3 months ago",
    "y_mean3": "the average of the last 3 months",
    "y_same_month_last_year": "the same month last year",
    "tests_last": "last month's testing volume",
    "tests_mean3": "average testing over the last 3 months",
    "tests_trend": "whether testing is rising or falling",
    "positivity_3m": "the share of tests that were positive recently",
    "month_of_year": "the time of year",
    "horizon": "how far ahead we are forecasting",
    "log_level": "the district's usual level",
    "relative_size": "the size of the district",
    "is_urban": "whether the district is urban or rural",
}

# Planners think in themes, not 14 separate inputs. Several inputs only encode WHICH district
# it is (size, usual level, urban/rural), so we report them together as one theme.
GROUPS = {
    "district characteristics (size, usual level, urban or rural)": ["log_level", "relative_size", "is_urban"],
    "recent diagnoses": ["y_last", "y_last2", "y_last3", "y_mean3"],
    "the same month last year": ["y_same_month_last_year"],
    "recent testing and positivity": ["tests_last", "tests_mean3", "tests_trend", "positivity_3m"],
    "the time of year": ["month_of_year"],
    "how far ahead we forecast": ["horizon"],
}


def fit_for_origin(df, origin=ORIGIN, horizon=HORIZON):
    series = prepare(df)
    filled = {d: (fill_gaps(s["y_raw"][:origin]), fill_gaps(s["tests_raw"][:origin]))
              for d, s in series.items()}
    X, y = training_set(series, filled, origin, horizon)
    return make_model().fit(X, y), X, series, filled


def rows_for(series, filled, origin, k):
    """One feature row per district for the forecast k months past the origin."""
    X, levels, names = [], [], []
    for d, s in series.items():
        yf, tf = filled[d]
        row, lvl = build_row(yf, tf, origin, k, s["moy"], s["rel"], s["urban"])
        X.append(row)
        levels.append(lvl)
        names.append(d)
    return np.array(X), np.array(levels), names


def explain_values(model, X_background, X_explain):
    import shap   # imported here so the rest of the file works without it
    explainer = shap.Explainer(model.predict, X_background, feature_names=FEATURE_NAMES)
    return explainer(X_explain)


def sentence(values, level, top=3):
    """Turn contributions into plain language, summed by theme."""
    themed = {g: sum(values[FEATURE_NAMES.index(f)] for f in feats) * level for g, feats in GROUPS.items()}
    order = sorted(themed, key=lambda g: -abs(themed[g]))[:top]
    return "; ".join(
        f"{g} {'raises' if themed[g] > 0 else 'lowers'} it by about {abs(themed[g]):.1f} cases" for g in order)


def main():
    df = pd.read_csv("data/synthetic/district_months.csv")
    model, X_train, series, filled = fit_for_origin(df)
    background = X_train[np.random.default_rng(0).choice(len(X_train), 100, replace=False)]

    X_now, levels, names = rows_for(series, filled, ORIGIN, k=0)   # 1 month ahead, all districts
    expl = explain_values(model, background, X_now)
    shap_cases = expl.values * levels[:, None]                      # contributions in cases
    base_cases = expl.base_values * levels                          # "typical" forecast in cases
    pred_cases = np.clip(model.predict(X_now), 0, None) * levels

    # 1) additivity check: typical + contributions = forecast
    gap = np.abs(base_cases + shap_cases.sum(axis=1) - model.predict(X_now) * levels).max()
    print(f"Additivity check, largest gap in cases (should be close to 0): {gap:.4f}\n")

    # 2) global importance: which inputs matter most overall
    imp = pd.Series(np.abs(expl.values).mean(axis=0), index=FEATURE_NAMES).sort_values()
    print("Mean absolute contribution (scaled units), all districts, 1 month ahead:")
    print(imp.sort_values(ascending=False).round(3).to_string(), "\n")
    Path("docs").mkdir(exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    plt.barh([LABELS[f] for f in imp.index], imp.values, color="#17A398")
    plt.xlabel("Average size of contribution to the forecast (scaled)")
    plt.title("What drives the forecasts? (synthetic data)")
    plt.tight_layout()
    plt.savefig("docs/shap_importance.png", dpi=150)
    print("Saved docs/shap_importance.png\n")

    # 3) plain-language explanation for three districts
    for d in ["Lilongwe", "Karonga", "Likoma"]:
        i = names.index(d)
        print(f"{d}: forecast {pred_cases[i]:.1f} cases next month "
              f"(the model's typical case at this district's level: {base_cases[i]:.1f}).")
        print("  Main drivers: " + sentence(expl.values[i], levels[i]) + ".\n")


if __name__ == "__main__":
    main()
