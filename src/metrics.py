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
