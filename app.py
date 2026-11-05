"""Planner dashboard (prototype, SYNTHETIC DATA).  Run:  streamlit run app.py

Thin on purpose: logic is in src/forecast_service.py, the chart in src/charts.py, colours in src/theme.py.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
import theme  # noqa: E402
from charts import forecast_chart  # noqa: E402
from forecast_service import explanations, history, make_forecast, warnings_for  # noqa: E402

st.set_page_config(page_title="HIV diagnoses forecast", page_icon="📈", layout="wide")
st.markdown(theme.CSS, unsafe_allow_html=True)


@st.cache_data
def load():
    return pd.read_csv("data/synthetic/district_months.csv")


@st.cache_data(show_spinner="Fitting models (about 30 seconds the first time) ...")
def run(df):
    table, artifacts = make_forecast(df)
    return table, explanations(artifacts)


def kpi(label, value, sub, main=False):
    return (f'<div class="card kpi{" main" if main else ""}"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div></div>')


df = load()
table, drivers = run(df)

with st.sidebar:
    st.markdown("### Africa CDC")
    st.caption("Digital Health Division · AES-PHI capstone")
    district = st.selectbox("District", sorted(table["district"].unique()))
    st.markdown("---")
    st.markdown("**How to read the bands**")
    st.caption("The darker band should hold the real value about 8 times in 10. "
               "The lighter band should hold it about 19 times in 20.")

t = table[table["district"] == district].sort_values("horizon")
h = history(df, district)
first = t.iloc[0]

st.markdown(
    '<div class="hero"><div class="eyebrow">Africa CDC · Public Health Informatics</div>'
    "<h1>District HIV new-diagnosis forecast</h1>"
    "<p>Monthly new HIV diagnoses for the next six months, with honest ranges and plain-language reasons. "
    "Diagnoses are a proxy for infections: they depend on how much testing is done.</p></div>"
    '<div class="stripe"></div>'
    '<div class="synthetic">SYNTHETIC DATA: these numbers are invented to test the tool and say nothing about Malawi.</div>',
    unsafe_allow_html=True,
)

st.markdown(f'<div class="section">{district}: next month ({first["month"]})</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
c1.markdown(kpi("Forecast", f"{first['forecast']:.0f}", "best single estimate", main=True), unsafe_allow_html=True)
c2.markdown(kpi("Likely range (80%)", f"{first['lo80']:.0f} – {first['hi80']:.0f}", "real value expected here ~8 times in 10"),
            unsafe_allow_html=True)
c3.markdown(kpi("Wide range (95%)", f"{first['lo95']:.0f} – {first['hi95']:.0f}", "real value expected here ~19 times in 20"),
            unsafe_allow_html=True)

st.markdown('<div class="section">Past and next six months</div>', unsafe_allow_html=True)
st.pyplot(forecast_chart(h, t, district), use_container_width=True)

left, right = st.columns([3, 2])
with left:
    st.markdown('<div class="section">Next 6 months</div>', unsafe_allow_html=True)
    cols = ["lo95", "lo80", "forecast", "hi80", "hi95"]
    show = t[["month"] + cols].copy()
    show[cols] = show[cols].round(0).astype(int)
    show.columns = ["Month", "Low (95%)", "Low (80%)", "Forecast", "High (80%)", "High (95%)"]
    st.dataframe(show, hide_index=True, use_container_width=True)
with right:
    st.markdown("<div class=\"section\">What drives next month's forecast</div>", unsafe_allow_html=True)
    text = drivers.get(district, "Explanations are unavailable (the shap library is not installed).")
    st.markdown(f'<div class="card driver">{text}</div>', unsafe_allow_html=True)
    st.markdown('<div class="section">Cautions</div>', unsafe_allow_html=True)
    st.markdown("".join(f'<span class="chip">{w}</span>' for w in warnings_for(first)), unsafe_allow_html=True)

with st.expander("How to read this"):
    st.markdown(
        "- The forecast is the model's best single guess; the bands show how unsure it is.\n"
        "- Bands were calibrated on past errors, so on average they hold the right share of outcomes.\n"
        "- Testing campaigns are not predicted and cause the biggest misses.\n"
        "- Forecasts support local knowledge; they do not replace it."
    )
