"""Generate a SYNTHETIC monthly district dataset that mimics routine HIV testing
reports. No real person, facility or Ministry of Health data is used.

Run:  python src/generate_data.py          (writes data/synthetic/district_months.csv)

Why synthetic first? Real data needs access approval. A synthetic set lets us build
and test the whole workflow now, and because WE know how it was made, we can check
whether our methods recover the truth.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Illustrative RELATIVE size of each district (NOT real populations).
DISTRICT_SIZE = {
    "Lilongwe": 3.0, "Blantyre": 2.6, "Mzimba": 1.8, "Mangochi": 1.5, "Kasungu": 1.3,
    "Zomba": 1.3, "Thyolo": 1.2, "Machinga": 1.1, "Dedza": 1.1, "Mulanje": 1.1,
    "Chikwawa": 1.0, "Dowa": 1.0, "Salima": 0.9, "Ntcheu": 0.9, "Nkhotakota": 0.8,
    "Mchinji": 0.8, "Karonga": 0.8, "Nsanje": 0.7, "Balaka": 0.7, "Chiradzulu": 0.6,
    "Phalombe": 0.6, "Rumphi": 0.5, "Nkhata Bay": 0.5, "Ntchisi": 0.4, "Chitipa": 0.4,
    "Mwanza": 0.3, "Neno": 0.3, "Likoma": 0.05,
}
URBAN = {"Lilongwe", "Blantyre", "Zomba"}

BASE_TESTS = 2000          # tests per month in a district of relative size 1.0
MEAN_POSITIVITY = 0.04     # share of tests that are positive at the start
POSITIVITY_DECAY = 0.008   # monthly exponential decline in positivity
TEST_GROWTH = 0.003        # monthly growth in testing volume
SEASON_AMP = 0.12          # size of the yearly testing cycle
P_MISSING = {"urban": 0.02, "rural": 0.06}   # chance a monthly report is missing
CAMPAIGNS_PER_DISTRICT = 2
CAMPAIGN_LEN = 2           # months
CAMPAIGN_TEST_BOOST = 1.6  # tests multiplied during a campaign
CAMPAIGN_YIELD = 0.9       # positivity multiplied during a campaign (lower yield)


def size_class(x):
    return "large" if x >= 1.3 else "medium" if x >= 0.7 else "small"


def make_districts(rng):
    rows = []
    for name, size in DISTRICT_SIZE.items():
        rows.append(dict(
            district=name, relative_size=size, size_class=size_class(size),
            urban_rural="urban" if name in URBAN else "rural",
            positivity0=MEAN_POSITIVITY * rng.lognormal(0, 0.25),
        ))
    return pd.DataFrame(rows)


def simulate(rng, n_months=72, start="2020-01"):
    months = pd.period_range(start, periods=n_months, freq="M")
    t = np.arange(n_months)
    out = []
    for d in make_districts(rng).itertuples():
        campaign = np.zeros(n_months, dtype=bool)
        for _ in range(CAMPAIGNS_PER_DISTRICT):
            s = int(rng.integers(6, n_months - CAMPAIGN_LEN))
            campaign[s:s + CAMPAIGN_LEN] = True

        month_of_year = np.array([m.month for m in months])
        season = 1 + SEASON_AMP * np.cos(2 * np.pi * (month_of_year - 4) / 12)
        tests_mean = BASE_TESTS * d.relative_size * np.exp(TEST_GROWTH * t) * season
        tests_mean = tests_mean * np.where(campaign, CAMPAIGN_TEST_BOOST, 1.0)
        tests = rng.poisson(tests_mean * rng.lognormal(0, 0.08, n_months))

        positivity = d.positivity0 * np.exp(-POSITIVITY_DECAY * t)
        positivity = positivity * np.where(campaign, CAMPAIGN_YIELD, 1.0)
        # gamma-Poisson = negative binomial: counts vary more than pure chance
        lam = tests * positivity * rng.gamma(20, 1 / 20, n_months)
        positives = np.minimum(rng.poisson(lam), tests)

        reported = (rng.random(n_months) > P_MISSING[d.urban_rural]).astype(int)
        out.append(pd.DataFrame(dict(
            district=d.district, month=months.astype(str), month_index=t,
            urban_rural=d.urban_rural, size_class=d.size_class,
            relative_size=d.relative_size, campaign_month=campaign.astype(int),
            tests=tests, new_diagnoses=positives, reported=reported,
        )))
    df = pd.concat(out, ignore_index=True)
    # a missing report has no numbers (like a blank cell in a real export)
    df.loc[df["reported"] == 0, ["tests", "new_diagnoses"]] = np.nan
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=72)
    ap.add_argument("--start", default="2020-01")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/synthetic")
    a = ap.parse_args()
    df = simulate(np.random.default_rng(a.seed), a.months, a.start)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "district_months.csv", index=False)
    print(f"Wrote {len(df):,} district-month rows to {out}/ (SYNTHETIC DATA)")


if __name__ == "__main__":
    main()
