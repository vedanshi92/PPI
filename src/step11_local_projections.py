"""
step11_local_projections.py
---------------------------
Local projections on the balanced, zero-filled country-year panel: which leads, de-risking
or the market?

  ldeals = log(1 + n_ren_nd)    renewable deals without de-risking
  derisk = ren_derisk           1 if any renewable deal that year was de-risked

For each horizon h = 0..5:
  (a) derisk_{t+h} ~ ldeals_t + 2 lags of ldeals and derisk | country + year FE
  (b) ldeals_{t+h} ~ derisk_t + 2 lags of ldeals and derisk | country + year FE
SEs clustered by country. Coefficients are standardised by SD(shock) / SD(outcome), using
the full panel's SDs, so the two directions are comparable (effect in outcome SDs of a
one-SD shock).

Each regression is also run with country-year controls lagged one year relative to the
shock year t (the hazard model's definitions, without the real interest rate): ND-GAIN
vulnerability, WGI regulatory quality, log GDP per capita, currency depreciation. For
comparison it is run on that controls sample without the controls as well.

Writes data/processed/local_projections.txt, figures/8_local_projections.png and
figures/8b_local_projections_controls.png.

Run from the project root:  python3 src/step11_local_projections.py
"""

import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf

from config import COUNTRY_YEAR_PANEL, LP_TXT, OUT
from step10_hazard import CONTROLS_NO_RIR, add_lagged_controls

HORIZONS = range(0, 6)
LAGS = 2
DIRECTIONS = [
    # (label, outcome, shock)
    ("Market -> de-risking: derisk(t+h) on ldeals(t)", "derisk", "ldeals"),
    ("De-risking -> market: ldeals(t+h) on derisk(t)", "ldeals", "derisk"),
]


CONTROLS = CONTROLS_NO_RIR
VARIANTS = [
    # (name, controls sample only, include controls)
    ("baseline", False, False),
    ("controls sample, no controls", True, False),
    ("with controls", True, True),
]


def prepare(panel: pd.DataFrame) -> pd.DataFrame:
    panel = add_lagged_controls(panel)
    d = panel[["iso3", "year", "n_ren_nd", "ren_derisk", *CONTROLS]].sort_values(
        ["iso3", "year"]).copy()
    d["ldeals"] = np.log1p(d["n_ren_nd"])
    d["derisk"] = d["ren_derisk"].astype(float)
    g = d.groupby("iso3")
    for var in ["ldeals", "derisk"]:
        for k in range(1, LAGS + 1):
            d[f"{var}_l{k}"] = g[var].shift(k)
        for h in HORIZONS:
            d[f"{var}_f{h}"] = g[var].shift(-h)
    return d


def main():
    print("=" * 70)
    print("Step 11: local projections — which leads, de-risking or the market?")
    print("=" * 70)
    if not COUNTRY_YEAR_PANEL.exists():
        sys.exit(f"{COUNTRY_YEAR_PANEL} not found — run step9_country_year_panel.py first.")
    d = prepare(pd.read_csv(COUNTRY_YEAR_PANEL))
    sd = {v: d[v].std() for v in ["ldeals", "derisk"]}
    controls = " + ".join(f"{v}_l{k}" for v in ["ldeals", "derisk"] for k in range(1, LAGS + 1))

    rows = []
    controls_sample = d.dropna(subset=CONTROLS)
    for variant, restrict, with_controls in VARIANTS:
        data = controls_sample if restrict else d
        rhs_extra = controls + (" + " + " + ".join(CONTROLS) if with_controls else "")
        for label, outcome, shock in DIRECTIONS:
            scale = sd[shock] / sd[outcome]
            for h in HORIZONS:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = pf.feols(f"{outcome}_f{h} ~ {shock} + {rhs_extra} | iso3 + year",
                                   data=data, vcov={"CRV1": "iso3"})
                t = fit.tidy().loc[shock]
                rows.append({
                    "variant": variant, "direction": label, "h": h, "coef": t["Estimate"],
                    "se": t["Std. Error"], "p": t["Pr(>|t|)"], "std_coef": t["Estimate"] * scale,
                    "std_lo": t["2.5%"] * scale, "std_hi": t["97.5%"] * scale, "N": fit._N,
                })
    res = pd.DataFrame(rows)

    lines = [
        "Local projections, balanced zero-filled country-year panel",
        "ldeals = log(1 + n_ren_nd); derisk = ren_derisk (any de-risked renewable deal in t).",
        f"Controls: {LAGS} lags of ldeals and derisk; country and year FE; SEs clustered by "
        "country.",
        f"Standardised coefficient = coef x SD(shock) / SD(outcome); full-panel SDs: "
        f"SD(ldeals) = {sd['ldeals']:.3f}, SD(derisk) = {sd['derisk']:.3f}.",
        f"Means: ldeals = {d['ldeals'].mean():.3f}, derisk = {d['derisk'].mean():.3f}.",
    ]
    lines += [
        "Controls (third variant): " + ", ".join(CONTROLS) + ", each at t-1 relative to the "
        "shock year t;",
        "  real interest rate left out. Depreciation winsorised at 1%/99%; regulatory quality "
        "1997/1999/2001 interpolated.",
        f"Controls sample: {len(controls_sample):,} of {len(d):,} country-years.",
        "",
        "Standardised coefficients by variant (* p<0.1, ** p<0.05, *** p<0.01):",
    ]
    for label, _, _ in DIRECTIONS:
        summary = pd.DataFrame(index=list(HORIZONS))
        for variant, _, _ in VARIANTS:
            r = res[(res["direction"] == label) & (res["variant"] == variant)].set_index("h")
            summary[variant] = [f"{c:.3f}{stars(p)}" for c, p in zip(r["std_coef"], r["p"])]
            summary[f"N ({variant})"] = r["N"].values
        summary.index.name = "h"
        with pd.option_context("display.width", 250, "display.max_columns", 10):
            lines += ["", label, summary.to_string()]
    lines += ["", "Full output:"]
    for variant, _, _ in VARIANTS:
        for label, _, _ in DIRECTIONS:
            r = res[(res["direction"] == label) & (res["variant"] == variant)].set_index("h")
            lines += ["", f"{label} — {variant}",
                      r[["coef", "se", "p", "std_coef", "std_lo", "std_hi", "N"]]
                      .round(4).to_string()]
    report = "\n".join(lines)
    print(report)
    LP_TXT.write_text(report + "\n")
    print(f"\nSaved to {LP_TXT}")
    plot(res[res["variant"] == "baseline"])
    plot_controls(res)


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def plot_controls(res: pd.DataFrame):
    """One panel per direction: the same-sample estimate without controls (gray) and with
    controls (blue), standardised, with 95% bands."""
    surface, ink, ink2, muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    styles = [("controls sample, no controls", "#a3a29c", "Without controls (same sample)"),
              ("with controls", "#2a78d6", "With lagged controls")]
    titles = ["Market → de-risking", "De-risking → market"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, facecolor=surface)
    for ax, (label, _, _), title in zip(axes, DIRECTIONS, titles):
        ax.set_facecolor(surface)
        for (variant, color, name), offset in zip(styles, [-0.06, 0.06]):
            r = res[(res["direction"] == label) & (res["variant"] == variant)]
            x = r["h"] + offset
            ax.fill_between(x, r["std_lo"], r["std_hi"], color=color, alpha=0.15, linewidth=0)
            ax.plot(x, r["std_coef"], color=color, linewidth=2, marker="o", markersize=5,
                    markeredgecolor=surface, markeredgewidth=1.5, label=name)
        ax.axhline(0, color="#c3c2b7", linewidth=0.8)
        ax.set_xticks(list(HORIZONS))
        ax.set_xlabel("Years after the shock (h)", color=ink2)
        ax.set_title(title, loc="left", fontsize=10.5, color=ink)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color("#c3c2b7")
        ax.tick_params(colors=muted, labelcolor=ink2, length=0)
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Response, in SDs of the outcome\nper 1-SD shock", color=ink2)
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle("Lead-lag results with lagged country controls", x=0.06, ha="left",
                 fontsize=12, fontweight="bold", color=ink, y=1.04)
    fig.text(0.06, 0.965, "ND-GAIN vulnerability, regulatory quality, log GDP per capita and "
             "depreciation at t−1; country and year FE, 2 lags of both", fontsize=9,
             color=ink2, ha="left")
    out = OUT / "figures" / "8b_local_projections_controls.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=surface)
    plt.close(fig)
    print(f"[FIG] {out}")


def plot(res: pd.DataFrame):
    """Both standardised impulse responses on one axis, with 95% bands."""
    surface, ink, ink2, muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    colors = ["#2a78d6", "#eb6834"]
    labels = ["Market → de-risking\n(response of derisk to ldeals)",
              "De-risking → market\n(response of ldeals to derisk)"]
    fig, ax = plt.subplots(figsize=(7.5, 4.4), facecolor=surface)
    ax.set_facecolor(surface)
    for (label, _, _), color, name, offset in zip(DIRECTIONS, colors, labels, [-0.06, 0.06]):
        r = res[res["direction"] == label]
        x = r["h"] + offset
        ax.fill_between(x, r["std_lo"], r["std_hi"], color=color, alpha=0.12, linewidth=0)
        ax.plot(x, r["std_coef"], color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor=surface, markeredgewidth=1.5, label=name)
    ax.axhline(0, color="#c3c2b7", linewidth=0.8)
    ax.set_xticks(list(HORIZONS))
    ax.set_xlabel("Years after the shock (h)", color=ink2)
    ax.set_ylabel("Response, in SDs of the outcome\nper 1-SD shock", color=ink2)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=muted, labelcolor=ink2, length=0)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("Which leads: de-risking or the market?", loc="left", fontsize=12,
                 fontweight="bold", color=ink, pad=24)
    ax.text(0, 1.02, "Standardised local projections with 95% bands; country and year FE, "
            "2 lags of both", transform=ax.transAxes, fontsize=9, color=ink2, va="bottom")
    out = OUT / "figures" / "8_local_projections.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=surface)
    plt.close(fig)
    print(f"[FIG] {out}")


if __name__ == "__main__":
    main()
