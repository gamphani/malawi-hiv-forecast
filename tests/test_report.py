import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from report import md_table  # noqa: E402


def test_md_table_layout():
    t = md_table(pd.DataFrame({"a": [1.234]}, index=["x"]), "name", 2)
    assert t.splitlines() == ["| name | a |", "|---|---|", "| x | 1.23 |"]
