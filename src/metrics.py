"""Forecast error metrics.

MASE = (average size of your errors) / (average size of a naive benchmark's errors on
the TRAINING data). Below 1 = better than the benchmark. Because it is a ratio, a
district with 5 cases a month and one with 500 can be compared fairly.
"""
import numpy as np


def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return np.nanmean(np.abs(y_true - y_pred))


def mase(y_true, y_pred, y_train, m=1):
    """m=1: benchmark is 'same as last month'. m=12: 'same as the same month last year'."""
    y_train = np.asarray(y_train, float)
    scale = np.nanmean(np.abs(y_train[m:] - y_train[:-m]))
    if not np.isfinite(scale) or scale == 0:
        return np.nan
    return mae(y_true, y_pred) / scale


def interval_score(y, lower, upper, alpha):
    """Score for ONE central (1-alpha) interval. Lower is better.
    = width of the interval  +  a penalty if the outcome falls outside it.
    A wide interval is penalised as vague; a miss is penalised as overconfident."""
    y, lower, upper = (np.asarray(v, float) for v in (y, lower, upper))
    return (upper - lower) + (2 / alpha) * (lower - y) * (y < lower) + (2 / alpha) * (y - upper) * (y > upper)


def wis(y, median, lowers, uppers, alphas):
    """Weighted interval score (Bracher et al. 2021): the median's error plus the interval
    scores of several nested intervals, each weighted by alpha/2."""
    total = 0.5 * np.abs(np.asarray(y, float) - np.asarray(median, float))
    for lo, hi, a in zip(lowers, uppers, alphas):
        total = total + (a / 2) * interval_score(y, lo, hi, a)
    return total / (len(alphas) + 0.5)
