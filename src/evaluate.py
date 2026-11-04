"""Rolling-origin evaluation of baseline forecasters.

For each cut-off month ("origin") we train ONLY on months before it, forecast the next
H months, and compare with what really happened. Then the origin moves forward and we
repeat. This copies real use, where you never see the future.

Run:  python src/evaluate.py
"""
import numpy as np
import pandas as pd

from baselines import BASELINES
from metrics import mase

FIRST_ORIGIN = 36   # train on at least 3 years before the first forecast
HORIZON = 6         # forecast 1 to 6 months ahead
STEP = 3            # move the origin 3 months each time


def fill_gaps(train):
    """Fill blank months using ONLY the past (linear between known points, then carry
    forward). Anything fitted on the whole series would leak the future."""
    s = pd.Series(train).interpolate(limit_direction="forward").ffill().bfill()
    return s.to_numpy(float)


def rolling_origin(df, models=BASELINES, first_origin=FIRST_ORIGIN, horizon=HORIZON, step=STEP):
    rows = []
    for district, g in df.groupby("district"):
        g = g.sort_values("month_index")
        y = g["new_diagnoses"].to_numpy(float)
        camp = g["campaign_month"].to_numpy()
        for origin in range(first_origin, len(y) - horizon + 1, step):
            train = fill_gaps(y[:origin])
            actual = y[origin:origin + horizon]
            for name, fn in models.items():
                pred = fn(train, horizon)
                for k in range(horizon):
                    if np.isnan(actual[k]):
                        continue                   # no report that month: cannot score
                    rows.append(dict(
                        district=district, model=name, origin=origin, horizon=k + 1,
                        abs_error=abs(actual[k] - pred[k]), scale=np.mean(np.abs(np.diff(train))),
                        campaign=int(camp[origin + k]),
                        urban_rural=g["urban_rural"].iloc[0], size_class=g["size_class"].iloc[0],
                    ))
    out = pd.DataFrame(rows)
    out["scaled_error"] = out["abs_error"] / out["scale"]   # MASE contribution (m=1 scale)
    return out


def main():
    df = pd.read_csv("data/synthetic/district_months.csv")
    res = rolling_origin(df)
    print("Overall MASE by model (lower is better; <1 beats 'same as last month'):")
    print(res.groupby("model")["scaled_error"].mean().round(3).to_string(), "\n")
    print("MASE by horizon:")
    print(res.pivot_table(index="model", columns="horizon", values="scaled_error").round(2).to_string(), "\n")
    print("MASE by urban/rural:")
    print(res.pivot_table(index="model", columns="urban_rural", values="scaled_error").round(2).to_string(), "\n")
    print("MASE in campaign vs normal months:")
    print(res.pivot_table(index="model", columns="campaign", values="scaled_error").round(2).to_string())


if __name__ == "__main__":
    main()
