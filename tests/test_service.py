import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forecast_service import history, make_forecast, warnings_for  # noqa: E402
from generate_data import simulate  # noqa: E402


def small_data():
    return simulate(np.random.default_rng(11), 72)


def test_forecast_table_is_complete_ordered_and_in_the_future():
    df = small_data()
    table, _ = make_forecast(df)
    assert len(table) == df["district"].nunique() * 6
    assert (table[["lo95", "lo80", "forecast", "hi80", "hi95"]] >= 0).all().all()
    assert (table.lo95 <= table.lo80 + 1e-9).all() and (table.lo80 <= table.forecast + 1e-9).all()
    assert (table.forecast <= table.hi80 + 1e-9).all() and (table.hi80 <= table.hi95 + 1e-9).all()
    assert table["month"].min() == "2026-01"          # data end 2025-12; first forecast month follows


def test_warnings_flag_small_districts_and_missing_reports():
    row = dict(size_class="small", missing_share_12m=0.25)
    text = " ".join(warnings_for(row))
    assert "Small district" in text and "25%" in text and "campaigns" in text
    quiet = warnings_for(dict(size_class="large", missing_share_12m=0.0))
    assert len(quiet) == 1                              # only the campaign caution remains


def test_history_returns_last_months_in_order():
    h = history(small_data(), "Lilongwe", last=24)
    assert len(h) == 24 and list(h["month"]) == sorted(h["month"])
