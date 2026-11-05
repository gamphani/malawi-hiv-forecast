import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from charts import forecast_chart  # noqa: E402
from forecast_service import history, make_forecast  # noqa: E402
from generate_data import simulate  # noqa: E402


def test_chart_builds_for_a_district():
    df = simulate(np.random.default_rng(11), 72)
    table, _ = make_forecast(df)
    t = table[table.district == "Lilongwe"].sort_values("horizon")
    fig = forecast_chart(history(df, "Lilongwe"), t, "Lilongwe")
    assert len(fig.axes) == 1 and len(fig.axes[0].lines) >= 3
