import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baselines import naive, seasonal_naive  # noqa: E402
from evaluate import fill_gaps  # noqa: E402
from metrics import mase  # noqa: E402


def test_mase_matches_hand_calculation():
    train = np.array([100, 110, 105, 120, 115])      # naive error: (10+5+15+5)/4 = 8.75
    assert abs(mase([10, 20], [4, 30], train) - 8 / 8.75) < 1e-9


def test_perfect_forecast_has_zero_mase():
    assert mase([5, 6], [5, 6], np.array([1, 2, 3, 4])) == 0


def test_naive_repeats_last_value():
    assert list(naive(np.array([1, 2, 9]), 3)) == [9, 9, 9]


def test_seasonal_naive_repeats_last_year():
    train = np.arange(24, dtype=float)               # months 0..23
    assert list(seasonal_naive(train, 3)) == [12, 13, 14]


def test_fill_gaps_uses_only_the_past():
    filled = fill_gaps(np.array([1.0, np.nan, 3.0, np.nan]))
    assert list(filled) == [1.0, 2.0, 3.0, 3.0]      # trailing gap carried forward
