import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generate_data import DISTRICT_SIZE, simulate  # noqa: E402


def build(seed, months=72):
    return simulate(np.random.default_rng(seed), months)


def test_same_seed_gives_identical_data():
    assert build(1).equals(build(1))


def test_different_seed_gives_different_data():
    assert not build(1).equals(build(2))


def test_shape_and_districts():
    df = build(3)
    assert len(df) == len(DISTRICT_SIZE) * 72
    assert df["district"].nunique() == len(DISTRICT_SIZE)


def test_counts_are_valid():
    df = build(4).dropna(subset=["tests"])
    assert (df["tests"] >= 0).all()
    assert (df["new_diagnoses"] >= 0).all()
    assert (df["new_diagnoses"] <= df["tests"]).all()


def test_missing_reports_have_no_numbers():
    df = build(5)
    missing = df[df["reported"] == 0]
    assert missing["tests"].isna().all() and missing["new_diagnoses"].isna().all()


def test_rural_districts_miss_more_reports_than_urban():
    df = build(6)
    rate = df.groupby("urban_rural")["reported"].apply(lambda s: 1 - s.mean())
    assert rate["rural"] > rate["urban"]


def test_positivity_falls_over_time():
    df = build(7).dropna(subset=["tests"])
    df = df.assign(pos=df["new_diagnoses"] / df["tests"])
    first = df[df["month_index"] < 12]["pos"].mean()
    last = df[df["month_index"] >= 60]["pos"].mean()
    assert last < first
