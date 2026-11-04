"""Simple baseline forecasters. Each takes the PAST values `train` (a 1-D array) and
the number of months to forecast `h`, and returns an array of length h.

Baselines are the yardstick: a machine-learning model is only worthwhile if it beats these.
"""
import warnings

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def naive(train, h):
    """Next months equal the last observed month."""
    return np.repeat(train[-1], h)


def seasonal_naive(train, h, m=12):
    """Each future month equals the same month one year earlier."""
    return np.array([train[len(train) - m + (k % m)] for k in range(h)], dtype=float)


def exp_smoothing(train, h, m=12):
    """Holt-Winters exponential smoothing: a weighted average that favours recent months,
    with a damped trend and a yearly seasonal pattern."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(
            train, trend="add", damped_trend=True, seasonal="add", seasonal_periods=m,
        ).fit()
        fc = np.asarray(fit.forecast(h), float)
    return np.clip(fc, 0, None)   # counts cannot be negative


BASELINES = {"naive": naive, "seasonal_naive": seasonal_naive, "exp_smoothing": exp_smoothing}
