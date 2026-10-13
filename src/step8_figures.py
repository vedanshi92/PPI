"""
step8_figures.py
----------------
Project-level summary figures for renewable projects: derisking and public finance by
IDA status, sponsor type and technology; multilateral vs bilateral support over time;
policy stacking (conditional probabilities); investment and capacity by support type.

Writes PNGs to data/processed/figures/ and the numbers behind every figure to
data/processed/figures/figure_data.xlsx (one sheet per figure).

Run from the project root:  python3 src/step8_figures.py
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OUT, PROJECT_PANEL

FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Reference palette (light mode), validated: slots 1-3 pass all checks; slot 3 (aqua) is
# below 3:1 contrast, so bars carry direct value labels and every figure has a table.
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BASELINE_GRAY = "#c3c2b7"  # the "without" group in emphasis comparisons

PERIODS = [1989, 1999, 2009, 2014, 2019, 2025]
PERIOD_LABELS = ["1990–99", "2000–09", "2010–14", "2015–19", "2020–24"]
TECH_LABELS = {"solar": "Solar", "wind": "Wind", "hydro_large": "Hydro, large",
               "hydro_small": "Hydro, small", "biomass": "Biomass", "waste": "Waste",
               "geothermal": "Geothermal", "biogas": "Biogas", "hybrid": "Hybrid"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "legend.frameon": False, "legend.fontsize": 9,
    "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": SURFACE,
})

tables: dict[str, pd.DataFrame] = {}


def finish(fig, ax, name, title, subtitle, note=None):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=24)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")
    if note:
        fig.text(0.01, -0.06, note, fontsize=8, color=MUTED, ha="left", va="top", wrap=True)
    fig.savefig(FIG_DIR / f"{name}.png")
    plt.close(fig)
    print(f"[FIG] {FIG_DIR / name}.png")


def pct_axis(ax, axis="y", top=None):
    fmt = matplotlib.ticker.PercentFormatter(xmax=1, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)
    if top is not None:
        (ax.set_ylim if axis == "y" else ax.set_xlim)(0, top)
    if axis == "y":
        ax.grid(axis="x", visible=False)
    else:
        ax.grid(axis="y", visible=False)


def grouped_bars(ax, table, colors, horizontal=False, label_fmt="{:.0%}"):
    """Bars for each column of `table`, grouped by its index; 2px surface gap between bars."""
    n = len(table.columns)
    width = 0.8 / n
    pos = np.arange(len(table))
    for i, (col, color) in enumerate(zip(table.columns, colors)):
        offset = (i - (n - 1) / 2) * width
        values = table[col].values
        if horizontal:
            bars = ax.barh(pos + offset, values, height=width, color=color, label=col,
                           edgecolor=SURFACE, linewidth=1.5)
        else:
            bars = ax.bar(pos + offset, values, width=width, color=color, label=col,
                          edgecolor=SURFACE, linewidth=1.5)
        for bar, v in zip(bars, values):
            if np.isnan(v):
                continue
            if horizontal:
                ax.text(bar.get_width() + 0.006, bar.get_y() + bar.get_height() / 2,
                        label_fmt.format(v), va="center", fontsize=8, color=INK_2)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006,
                        label_fmt.format(v), ha="center", va="bottom", fontsize=8, color=INK_2)
    (ax.set_yticks if horizontal else ax.set_xticks)(pos)
    (ax.set_yticklabels if horizontal else ax.set_xticklabels)(table.index)
    ax.tick_params(length=0)


def fig_ida_status(r):
    t = (r.groupby("IDA Status")
         .agg(n=("project_id", "size"), Derisked=("derisk", "mean"),
              **{"Public finance": ("public_finance", "mean")},
              **{"Any support": ("any_support", "mean")})
         .reindex(["Non-IDA", "Blended", "IDA"]))
    tables["1_ida_status"] = t
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    grouped_bars(ax, t[["Derisked", "Public finance", "Any support"]], SERIES)
    ax.set_xticklabels([f"{i}\nn = {n:,}" for i, n in zip(t.index, t["n"])])
    pct_axis(ax, top=max(0.8, t[["Any support"]].max().max() + 0.1))
    ax.legend(loc="upper left", ncol=3)
    finish(fig, ax, "1_ida_status", "Support rises with IDA eligibility",
           "Share of renewable projects with each kind of support, by the country's IDA status",
           "Any support = derisk, public finance, government derisk or government finance. "
           "Government fields are mostly unrecorded outside 2010–19.")


def fig_sponsor_type(r):
    order = ["domestic_only", "mixed", "foreign_only"]
    labels = {"domestic_only": "Domestic only", "mixed": "Domestic + foreign",
              "foreign_only": "Foreign only"}
    t = (r[r["sponsor_type"].isin(order)].groupby("sponsor_type")
         .agg(n=("project_id", "size"), Derisked=("derisk", "mean"),
              **{"Public finance": ("public_finance", "mean")})
         .reindex(order).rename(index=labels))
    tables["2_sponsor_type"] = t
    fig, ax = plt.subplots(figsize=(7, 4))
    grouped_bars(ax, t[["Derisked", "Public finance"]], SERIES[:2])
    ax.set_xticklabels([f"{i}\nn = {n:,}" for i, n in zip(t.index, t["n"])])
    pct_axis(ax, top=0.5)
    ax.legend(loc="upper left", ncol=2)
    finish(fig, ax, "2_sponsor_type", "Projects with foreign sponsors get more support",
           "Share of renewable projects derisked or publicly financed, by sponsor nationality",
           "Sponsor nationality compares each sponsor's country with the project's country. "
           "Projects with no known sponsor country are excluded.")


def fig_technology(r):
    t = (r.groupby("tech")
         .agg(n=("project_id", "size"), Derisked=("derisk", "mean"),
              **{"Public finance": ("public_finance", "mean")})
         .query("n >= 20").sort_values("Public finance"))
    t.index = t.index.map(TECH_LABELS)
    tables["3_technology"] = t
    fig, ax = plt.subplots(figsize=(7.5, 5))
    grouped_bars(ax, t[["Derisked", "Public finance"]], SERIES[:2], horizontal=True)
    ax.set_yticklabels([f"{i}  (n = {n:,})" for i, n in zip(t.index, t["n"])])
    pct_axis(ax, axis="x", top=max(0.6, t["Public finance"].max() + 0.08))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="lower right")
    finish(fig, ax, "3_technology", "Geothermal and solar get the most support; bioenergy almost none",
           "Share of renewable projects derisked or publicly financed, by technology",
           "Hybrid projects (n = 11) are left out.")


def fig_mls_bls(r):
    period = pd.cut(r["year"], PERIODS, labels=PERIOD_LABELS)
    mls = (r["derisk_multilateral"] == 1) | (r["public_finance_multilateral"] == 1)
    bls = (r["derisk_bilateral"] == 1) | (r["public_finance_bilateral"] == 1)
    kind = pd.Series(np.select([mls & bls, mls, bls], ["Both", "Multilateral only",
                                                       "Bilateral only"], "Neither"),
                     index=r.index)
    t = (pd.crosstab(period, kind, normalize="index")
         [["Multilateral only", "Both", "Bilateral only"]])
    t.insert(0, "n", period.value_counts().reindex(PERIOD_LABELS).values)
    tables["4_mls_bls"] = t
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bottom = np.zeros(len(t))
    pos = np.arange(len(t))
    for col, color in zip(["Multilateral only", "Both", "Bilateral only"], SERIES):
        ax.bar(pos, t[col], bottom=bottom, width=0.6, color=color, label=col,
               edgecolor=SURFACE, linewidth=1.5)
        bottom += t[col].values
    for x, total in zip(pos, bottom):
        ax.text(x, total + 0.006, f"{total:.0%}", ha="center", va="bottom", fontsize=8,
                color=INK_2)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{p}\nn = {n:,}" for p, n in zip(t.index, t["n"])])
    ax.tick_params(length=0)
    pct_axis(ax, top=0.45)
    ax.legend(loc="upper left", ncol=3)
    finish(fig, ax, "4_mls_bls", "Bilateral support appears after 2010, often alongside multilateral",
           "Share of renewable projects with multilateral (MLS) or bilateral (BLS) support, by "
           "financial-closure period",
           "Support = any derisking (guarantee, insurance, risk management) or public finance "
           "(loan, syndication, equity) in MultiLateralSupport / BiLateralSupport.")


def fig_stacking(r):
    mls = ((r["derisk_multilateral"] == 1) | (r["public_finance_multilateral"] == 1)).astype(int)
    bls = ((r["derisk_bilateral"] == 1) | (r["public_finance_bilateral"] == 1)).astype(int)
    rows = [
        ("Public finance,\ngiven derisked", r["public_finance"], r["derisk"]),
        ("Bilateral public finance,\ngiven bilateral derisk", r["public_finance_bilateral"],
         r["derisk_bilateral"]),
        ("Multilateral support,\ngiven bilateral support", mls, bls),
    ]
    t = pd.DataFrame(
        [{"comparison": label, "Without": y[x == 0].mean(), "With": y[x == 1].mean(),
          "n_with": int((x == 1).sum()), "n_without": int((x == 0).sum())}
         for label, y, x in rows]).set_index("comparison")
    tables["5_stacking"] = t
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    grouped_bars(ax, t[["Without", "With"]], [BASELINE_GRAY, SERIES[0]])
    ax.set_xticklabels([f"{i}\n(with: n = {n:,})" for i, n in zip(t.index, t["n_with"])])
    pct_axis(ax, top=0.9)
    ax.legend(["Without the first support", "With the first support"], loc="upper right")
    finish(fig, ax, "5_stacking", "Support stacks: one kind makes another far more likely",
           "Share of renewable projects receiving the second kind of support, with and without "
           "the first",
           "Raw shares. In fixed effects models (step 7) the gaps shrink but stay significant, "
           "except multilateral given bilateral within country-years.")


def fig_size(r):
    groups = ["Neither", "Public finance only", "Derisked only", "Both"]
    kind = pd.Series(np.select(
        [(r["derisk"] == 1) & (r["public_finance"] == 1), r["derisk"] == 1,
         r["public_finance"] == 1], ["Both", "Derisked only", "Public finance only"],
        "Neither"), index=r.index)
    t = pd.DataFrame({
        "n": kind.value_counts().reindex(groups),
        "median_investment_musd": r.groupby(kind)["investment_musd"].median().reindex(groups),
        "median_capacity_mw": r.groupby(kind)["capacity_mw"].median().reindex(groups),
    })
    tables["6_size"] = t
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, col, label in [(axes[0], "investment_musd", "Investment (USD million, log scale)"),
                           (axes[1], "capacity_mw", "Capacity (MW, log scale)")]:
        data = [r.loc[(kind == g) & (r[col] > 0), col].values for g in groups]
        box = ax.boxplot(data, orientation="horizontal", widths=0.5, patch_artist=True, showfliers=False,
                         medianprops=dict(color=INK, linewidth=1.5),
                         whiskerprops=dict(color=MUTED, linewidth=0.8),
                         capprops=dict(color=MUTED, linewidth=0.8))
        for patch in box["boxes"]:
            patch.set(facecolor=SERIES[0], alpha=0.35, edgecolor=SERIES[0], linewidth=1)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
        ax.set_xlabel(label)
        ax.grid(axis="y", visible=False)
        ax.tick_params(length=0)
        for i, d in enumerate(data, start=1):
            if len(d):
                ax.text(np.median(d), i + 0.33, f"{np.median(d):,.0f}", ha="center",
                        fontsize=8, color=INK_2)
    axes[0].set_yticks(range(1, len(groups) + 1))
    axes[0].set_yticklabels([f"{g}\n(n = {n:,})" for g, n in zip(groups, t["n"])])
    fig.subplots_adjust(wspace=0.08)
    finish(fig, axes[0], "6_size", "Projects with both derisking and public finance are the largest",
           "Investment and capacity of renewable projects by type of support "
           "(box = middle 50%, line = median, labelled)",
           "Outliers beyond the whiskers are hidden. Capacity uses MW projects only.")


def fig_trend(r):
    period = pd.cut(r["year"], PERIODS, labels=PERIOD_LABELS)
    t = r.groupby(period, observed=False).agg(
        n=("project_id", "size"), Derisked=("derisk", "mean"),
        **{"Public finance": ("public_finance", "mean")},
        **{"Foreign sponsor": ("has_foreign_sponsor",
                               lambda s: s.astype(str).map({"True": 1, "False": 0}).mean())})
    tables["7_trend"] = t
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(t))
    for col, color in zip(["Foreign sponsor", "Public finance", "Derisked"], SERIES):
        ax.plot(x, t[col], color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=col)
        ax.text(x[-1] + 0.12, t[col].iloc[-1], f"{col} {t[col].iloc[-1]:.0%}", va="center",
                fontsize=8.5, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\nn = {n:,}" for p, n in zip(t.index, t["n"])])
    ax.set_xlim(-0.3, len(t) - 0.3 + 1.1)
    ax.tick_params(length=0)
    pct_axis(ax, top=0.8)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=3)
    finish(fig, ax, "7_trend", "Public finance for renewables doubled after 2015",
           "Share of renewable projects with each feature, by financial-closure period",
           "Foreign sponsor share is among projects with a known sponsor country.")


def main():
    print("=" * 70)
    print("Step 8: summary figures")
    print("=" * 70)
    if not PROJECT_PANEL.exists():
        sys.exit(f"{PROJECT_PANEL} not found — run step5_merge_panel.py first.")
    df = pd.read_csv(PROJECT_PANEL, low_memory=False)
    r = df[df["technology_category"] == "renewable"].copy()

    for fig in [fig_ida_status, fig_sponsor_type, fig_technology, fig_mls_bls,
                fig_stacking, fig_size, fig_trend]:
        fig(r)

    with pd.ExcelWriter(FIG_DIR / "figure_data.xlsx") as writer:
        for name, table in tables.items():
            table.round(4).to_excel(writer, sheet_name=name[:31])
    print(f"[FIG] Figure data: {FIG_DIR / 'figure_data.xlsx'}")


if __name__ == "__main__":
    main()
