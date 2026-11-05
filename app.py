"""Planner dashboard (prototype, SYNTHETIC DATA).  Run:  streamlit run app.py

Thin on purpose: all logic is in src/forecast_service.py, which is tested.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from forecast_service import explanations, history, make_forecast, warnings_for  # noqa: E402

st.set_page_config(page_title="HIV diagnoses forecast", layout="wide")
st.error("SYNTHETIC DATA. These numbers are invented to test the tool. They say nothing about Malawi.")
st.title("District HIV new-diagnosis forecast")
st.caption("Monthly new HIV diagnoses (a proxy for infections: it depends on how much testing is done). Next 6 months.")


@st.cache_data
def load():
    return pd.read_csv("data/synthetic/district_months.csv")


@st.cache_data(show_spinner="Fitting models (about 30 seconds the first time) ...")
def run(df):
    table, artifacts = make_forecast(df)
    return table, explanations(artifacts)


df = load()
table, drivers = run(df)

district = st.sidebar.selectbox("District", sorted(table["district"].unique()))
st.sidebar.markdown("Shaded bands: the darker one holds the real value about 8 times in 10, "
                    "the lighter about 19 times in 20.")
t = table[table["district"] == district].sort_values("horizon")
h = history(df, district)

first = t.iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric(f"Next month ({first['month']})", f"{first['forecast']:.0f}")
c2.metric("Likely range (80%)", f"{first['lo80']:.0f} to {first['hi80']:.0f}")
c3.metric("Wide range (95%)", f"{first['lo95']:.0f} to {first['hi95']:.0f}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(h["month"], h["actual"], color="black", marker="o", ms=3, label="Reported (past)")
ax.plot(t["month"], t["forecast"], color="#17A398", marker="o", label="Forecast")
ax.fill_between(t["month"], t["lo95"], t["hi95"], color="#17A398", alpha=0.15, label="95% range")
ax.fill_between(t["month"], t["lo80"], t["hi80"], color="#17A398", alpha=0.30, label="80% range")
ax.set_ylabel("New diagnoses per month")
ax.set_ylim(bottom=0)
ax.tick_params(axis="x", rotation=60)
ax.legend(loc="upper left")
ax.set_title(district)
st.pyplot(fig)

st.subheader("Next 6 months")
show = t[["month", "lo95", "lo80", "forecast", "hi80", "hi95"]].round(0).astype({c: int for c in ["lo95", "lo80", "forecast", "hi80", "hi95"]})
show.columns = ["Month", "Low (95%)", "Low (80%)", "Forecast", "High (80%)", "High (95%)"]
st.dataframe(show, hide_index=True)

st.subheader("What drives next month's forecast")
st.write(drivers.get(district, "Explanations are unavailable (the shap library is not installed)."))
st.subheader("Cautions")
for w in warnings_for(first):
    st.warning(w)

with st.expander("How to read this"):
    st.markdown(
        "- The forecast is the model's best single guess; the bands show how unsure it is.\n"
        "- Bands were calibrated on past errors, so on average they hold the right share of outcomes.\n"
        "- Testing campaigns are not predicted and cause the biggest misses.\n"
        "- Forecasts are not decisions: use them next to local knowledge."
    )
