import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import FEATURE_NAMES, build_row  # noqa: E402


def make_series(n=40):
    y = np.arange(n, dtype=float) + 10
    tests = np.arange(n, dtype=float) * 10 + 100
    moy = (np.arange(n) % 12) + 1
    return y, tests, moy


def test_row_has_one_value_per_feature_name():
    y, tests, moy = make_series()
    row, lvl = build_row(y, tests, 30, 2, moy, 1.0, 0)
    assert len(row) == len(FEATURE_NAMES) and lvl > 0


def test_features_ignore_everything_from_the_origin_onwards():
    """The no-leakage test: changing the FUTURE must not change the features."""
    y, tests, moy = make_series()
    row1, _ = build_row(y, tests, 30, 2, moy, 1.0, 0)
    y2, t2 = y.copy(), tests.copy()
    y2[30:] += 1000
    t2[30:] += 1000
    row2, _ = build_row(y2, t2, 30, 2, moy, 1.0, 0)
    assert row1 == row2
