import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from conformal import correction  # noqa: E402


def test_correction_widens_when_intervals_miss():
    misses = np.array([0.5, 0.2, 0.0, -0.1, 0.3, 0.1, 0.4, 0.0, -0.2, 0.6])
    assert correction(misses, 0.2) > 0           # real values fell outside: widen


def test_correction_shrinks_when_intervals_are_too_wide():
    comfortably_inside = np.full(20, -0.5)
    assert correction(comfortably_inside, 0.2) < 0
