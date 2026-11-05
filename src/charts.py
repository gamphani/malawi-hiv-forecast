"""The forecast chart, as a plain function so it can be tested without Streamlit."""
import matplotlib
import matplotlib.pyplot as plt

import theme

matplotlib.rcParams["font.family"] = "sans-serif"


def forecast_chart(hist, fc, district):
    """hist: columns month, actual. fc: columns month, lo95, lo80, forecast, hi80, hi95."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    n, m = len(hist), len(fc)
    xh = list(range(n))
    xf = list(range(n, n + m))   # positions, so the forecast always sits to the right of the past
    labels = list(hist["month"]) + list(fc["month"])

    ax.fill_between(xf, fc["lo95"], fc["hi95"], color=theme.AQUA, alpha=0.16, label="95% range", lw=0)
    ax.fill_between(xf, fc["lo80"], fc["hi80"], color=theme.AQUA, alpha=0.34, label="80% range", lw=0)
    ax.plot(xh, hist["actual"], color=theme.DARK, lw=2, marker="o", ms=3.5, label="Reported (past)")
    if n:   # dotted join from the last reported month to the first forecast
        ax.plot([xh[-1], xf[0]], [hist["actual"].iloc[-1], fc["forecast"].iloc[0]],
                color=theme.AQUA, lw=1.5, ls=":")
    ax.plot(xf, fc["forecast"], color=theme.AQUA, lw=2.8, marker="o", ms=6,
            markeredgecolor="white", markeredgewidth=1.5, label="Forecast")
    ax.axvline(xf[0] - 0.5, color=theme.GOLD, lw=1.2, ls="--")
    ax.text(xf[0] - 0.3, 0.02, "forecast starts", transform=ax.get_xaxis_transform(),
            color=theme.GOLD, fontsize=9, fontweight="bold", va="bottom")

    ax.set_ylim(bottom=0)
    ticks = list(range(0, n + m, max(1, (n + m) // 14)))
    ax.set_xticks(ticks)
    ax.set_xticklabels([labels[i] for i in ticks])
    ax.set_ylabel("New diagnoses per month", color=theme.INK)
    ax.tick_params(axis="x", rotation=60, colors=theme.INK, labelsize=8)
    ax.tick_params(axis="y", colors=theme.INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(theme.MIST)
    ax.grid(axis="y", color=theme.MIST, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, ncol=4, fontsize=9)
    fig.tight_layout()
    return fig
