# Data card: synthetic district-month HIV testing data

**Status:** SYNTHETIC. No real person, facility or Ministry of Health data.
**File:** `data/synthetic/district_months.csv` (made by `src/generate_data.py`, seed 42)

## Purpose
Build and test the forecasting workflow before real data access is granted. Because we
know how the data were made, we can check whether methods behave sensibly.

## What it contains
28 districts x 72 months (Jan 2020 to Dec 2025) = 2,016 rows.

| Column | Meaning |
|---|---|
| district | District name. Names are real; every number is invented |
| month, month_index | Calendar month and 0-based counter |
| urban_rural, size_class, relative_size | Fixed district attributes (illustrative, not real populations) |
| campaign_month | 1 if a testing campaign ran that month |
| tests | HIV tests done (blank if the report is missing) |
| new_diagnoses | New positives (blank if the report is missing). This is the forecast target |
| reported | 1 if the monthly report exists, 0 if missing |

## How it was made
Tests = size x growth x yearly cycle x campaign boost x noise. Positivity falls over
time. Diagnoses are counts drawn around tests x positivity (negative binomial).
Missing reports are more common in rural districts (6% vs 2%).

## Target and its limits
The target is new diagnoses, a PROXY for new infections. Diagnoses depend on how much
testing is done. Campaign months raise tests, so diagnoses rise without any change in
infections. Models must use testing information and results must say so.

## Known limits
- Illustrative parameters, not calibrated to Malawi data
- Simpler than reality: no double counting, no late reports, no real district patterns
- Results on this data show the workflow works. They say NOTHING about Malawi

## Do not
- Report results from this dataset as findings about Malawi
- Use it to claim that machine learning beats baselines in practice
