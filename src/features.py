"""Turn a district's history into a row of INPUTS (features) for a forecasting model.

The golden rule: a feature may only use information that would be KNOWN at the moment
of forecasting. For a forecast made at `origin` (the first unknown month), we know
months 0..origin-1 and nothing after. So we use LAGGED tests, never this month's tests.

To let one model serve tiny and huge districts, counts are divided by the district's
recent average level ("scaling"). The model predicts a ratio; we multiply it back.
"""
import numpy as np

MIN_HISTORY = 12   # need a year of history to build the "same month last year" input
FEATURE_NAMES = [
    "y_last", "y_last2", "y_last3", "y_mean3", "y_same_month_last_year",
    "tests_last", "tests_mean3", "tests_trend", "positivity_3m",
    "month_of_year", "horizon", "log_level", "relative_size", "is_urban",
]


def level_of(y, origin):
    return max(float(np.mean(y[origin - 12:origin])), 0.5)


def build_row(y, tests, origin, k, month_of_year, relative_size, is_urban):
    """Features for predicting month `origin + k` (horizon k+1) using data BEFORE origin."""
    lvl = level_of(y, origin)
    t_lvl = max(float(np.mean(tests[origin - 12:origin])), 1.0)
    last3_y = y[origin - 3:origin]
    last3_t = tests[origin - 3:origin]
    t_old = max(float(np.mean(tests[origin - 12:origin - 9])), 1.0)
    return [
        y[origin - 1] / lvl, y[origin - 2] / lvl, y[origin - 3] / lvl, np.mean(last3_y) / lvl,
        y[origin + k - 12] / lvl,                      # same calendar month, a year earlier
        tests[origin - 1] / t_lvl, np.mean(last3_t) / t_lvl, np.mean(last3_t) / t_old,
        last3_y.sum() / max(last3_t.sum(), 1.0),
        month_of_year[origin + k], k + 1, np.log(lvl), relative_size, is_urban,
    ], lvl
