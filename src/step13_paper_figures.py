"""
step13_paper_figures.py
-----------------------
Figures for the paper. Main text:
  fig1  The market grew, de-risking didn't keep pace (renewable deals and de-risked deals
        per year; de-risked share)
  fig2  De-risking arrives after the market starts (dumbbell: first renewable deal ->
        first de-risked renewable deal, one row per treated country; histogram of the gap)
  fig3  Recent deal flow raises the chance of de-risking (predicted annual probability of
        first de-risking against deals in t-1..t-3; LPM, logit, cloglog; other variables
        at their means; histogram of the data underneath)
  fig4  The result is stable across specifications (deal-flow coefficient / marginal
        effect across specifications)
  fig5  The market leads more strongly than it follows (local projections, both directions,
        same SD units and y-axis)
Appendix:
  figA1 Where de-risking goes (standardised control coefficients)
  figA2 Survival curves (Kaplan-Meier, landmark at 3 years after entry, split by early
        deal flow)
  figA3 Baseline hazard (year effects; years-since-entry effects)
  figA4 Map of the year of the first de-risked renewable deal
  figA5 Stacking (public-finance-on-derisk coefficient under three fixed-effect sets)

Writes PNGs to data/processed/figures/paper/ and the numbers behind them to
data/processed/figures/paper/paper_figure_data.xlsx.

Run from the project root:  python3 src/step13_paper_figures.py
"""

import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf
import statsmodels.api as sm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy import stats

import step10_hazard as hz
import step11_local_projections as lp
from config import COUNTRY_YEAR_PANEL, OUT, PROJECT_PANEL

FIG_DIR = OUT / "figures" / "paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Reference palette (light mode). Slots 1-3 validated all-pairs; aqua is below 3:1
# contrast, so it is always direct-labelled and every figure has a data table.
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY = "#a3a29c"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordinal blue, light->dark

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2, "legend.frameon": False, "legend.fontsize": 9,
    "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": SURFACE,
})

tables: dict[str, pd.DataFrame] = {}


def titled(fig, ax, title, subtitle, note=None, note_y=-0.06):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=24)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")
    if note:
        fig.text(0.01, note_y, note, fontsize=8, color=MUTED, ha="left", va="top", wrap=True)


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png")
    plt.close(fig)
    print(f"[FIG] {FIG_DIR / name}.png")


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def coef_rows(ax, rows: pd.DataFrame, color=BLUE, unit_fmt="{:+.2f}"):
    """Horizontal coefficient plot: rows has est, lo, hi, label (top to bottom)."""
    y = np.arange(len(rows))[::-1]
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.hlines(y, rows["lo"], rows["hi"], color=color, linewidth=2, alpha=0.55)
    ax.plot(rows["est"], y, "o", color=color, markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=1.5)
    for yi, (_, r) in zip(y, rows.iterrows()):
        ax.text(r["hi"], yi + 0.22, " " + unit_fmt.format(r["est"]) + r.get("stars", ""),
                fontsize=8, color=INK_2, va="bottom", ha="left")
    ax.set_yticks(y)
    ax.set_yticklabels(rows["label"])
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=0)


# ---------------------------------------------------------------------------- figure 1
def fig1(cy: pd.DataFrame):
    t = cy.groupby("year").agg(renewable=("n_ren", "sum"),
                               derisked=("n_derisk_renewable", "sum"))
    t["share"] = t["derisked"] / t["renewable"]
    tables["fig1_deals_per_year"] = t
    pre, post = t.loc[:2009], t.loc[2010:]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12})
    ax1.plot(t.index, t["renewable"], color=BLUE, linewidth=2)
    ax1.plot(t.index, t["derisked"], color=ORANGE, linewidth=2)
    ax1.text(t.index[-1] + 0.4, t["renewable"].iloc[-1], "All renewable deals",
             color=INK_2, fontsize=8.5, va="center")
    ax1.text(t.index[-1] + 0.4, t["derisked"].iloc[-1], "De-risked deals", color=INK_2,
             fontsize=8.5, va="center")
    ax1.set_ylabel("Deals per year")
    ax1.set_ylim(0, None)
    ax2.plot(t.index, 100 * t["share"], color=ORANGE, linewidth=2, marker="o", markersize=3.5,
             markeredgecolor=SURFACE, markeredgewidth=1)
    for period, (y0, y1) in [(pre, (1990, 2009)), (post, (2010, 2024))]:
        share = 100 * period["derisked"].sum() / period["renewable"].sum()
        ax2.hlines(share, y0, y1, color=MUTED, linewidth=1)
        ax2.text(2003 if y0 == 1990 else (y0 + y1) / 2, 24, f"{y0}–{y1} average: {share:.1f}%",
                 ha="center", fontsize=8, color=INK_2)
    ax2.set_ylabel("De-risked share (%)")
    ax2.set_ylim(0, max(30, 100 * t["share"].max() * 1.1))
    ax2.set_xlim(1989.5, 2024.5)
    for ax in (ax1, ax2):
        ax.grid(axis="x", visible=False)
        ax.tick_params(length=0)
    titled(fig, ax1, "The market grew, de-risking didn't keep pace",
           "Renewable PPI deals per year, and deals with multilateral or bilateral risk "
           "transfer (guarantee, insurance, risk management)",
           f"Share line: de-risked deals as % of renewable deals that year; flat lines are "
           f"period averages. Renewable deals: {pre['renewable'].sum():,} in 1990–2009, "
           f"{post['renewable'].sum():,} in 2010–2024.", note_y=0.0)
    save(fig, "fig1_market_vs_derisking")


# ---------------------------------------------------------------------------- figure 2
def fig2(cy: pd.DataFrame):
    first_ren = cy[cy["n_ren"] > 0].groupby("iso3")["year"].min()
    c = (cy.groupby("iso3")[["country_name", "treat_year"]].first()
         .dropna(subset=["treat_year"]))
    c["first_renewable"] = first_ren.reindex(c.index)
    c["gap"] = c["treat_year"] - c["first_renewable"]
    c = c.sort_values(["first_renewable", "treat_year"])
    tables["fig2_timing"] = c
    n = len(c)
    fig = plt.figure(figsize=(9, 0.2 * n + 2.2))
    gs = GridSpec(1, 2, width_ratios=[3.2, 1], wspace=0.08, figure=fig)
    ax = fig.add_subplot(gs[0])
    y = np.arange(n)[::-1]
    ax.hlines(y, c["first_renewable"], c["treat_year"], color=AXIS, linewidth=1.6)
    ax.plot(c["first_renewable"], y, "o", color=BLUE, markersize=5.5,
            markeredgecolor=SURFACE, markeredgewidth=1, label="First renewable deal")
    ax.plot(c["treat_year"], y, "o", color=ORANGE, markersize=5.5,
            markeredgecolor=SURFACE, markeredgewidth=1, label="First de-risked renewable deal")
    ax.set_yticks(y)
    ax.set_yticklabels(c["country_name"], fontsize=7.5)
    ax.set_ylim(-0.8, n - 0.2)
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8.5)
    ax.set_xlabel("Year")
    same = int((c["gap"] == 0).sum())
    ax.set_title("De-risking arrives after the market starts", loc="left", fontsize=12,
                 fontweight="bold", color=INK, pad=34)

    ax_h = fig.add_subplot(gs[1])
    bins = np.arange(-0.5, c["gap"].max() + 1.5, 2)
    ax_h.hist(c["gap"], bins=bins, color=ORANGE, edgecolor=SURFACE, linewidth=1.5)
    med = c["gap"].median()
    ax_h.axvline(med, color=INK_2, linewidth=1)
    ax_h.text(med, ax_h.get_ylim()[1] * 0.97, f"  median {med:g} years", fontsize=8,
              color=INK_2, va="top")
    ax_h.set_xlabel("Years from first renewable deal\nto first de-risked deal")
    ax_h.set_ylabel("Countries")
    ax_h.grid(axis="x", visible=False)
    ax_h.tick_params(length=0)
    ax_h.set_box_aspect(1)
    fig.text(0.01, 0.0, f"{n} countries with a de-risked renewable deal, sorted by the year "
             f"of their first renewable deal. In {same} the first renewable deal was itself "
             "de-risked (gap 0).", fontsize=8, color=MUTED, ha="left", va="top")
    save(fig, "fig2_timing_dumbbell")
    return med


# ---------------------------------------------------------------------------- figure 3
def fig3(risk: pd.DataFrame):
    """Predicted annual hazard at the means of all other variables (FE dummies included as
    sample shares), for LPM, logit and cloglog fitted on one common sample: complete cases
    in year / region / income cells that contain an event (the cells logit and cloglog can
    use)."""
    d, exog = hz.fe_design(risk, ["year", "region", "income_group"])
    groups = pd.factorize(d["iso3"])[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = {
            "LPM": sm.OLS(d["event"], exog).fit(cov_type="cluster",
                                                cov_kwds={"groups": groups}),
            "Logit": sm.Logit(d["event"], exog).fit(disp=0, cov_type="cluster",
                                                    cov_kwds={"groups": groups}),
            "Cloglog": sm.GLM(d["event"], exog, family=sm.families.Binomial(
                link=sm.families.links.CLogLog())).fit(cov_type="cluster",
                                                      cov_kwds={"groups": groups}),
        }
    inv = {"LPM": lambda e: e, "Logit": lambda e: 1 / (1 + np.exp(-e)),
           "Cloglog": lambda e: 1 - np.exp(-np.exp(e))}
    x_max = int(np.ceil(d["deals_l1_3"].quantile(0.95)))
    grid = np.arange(0, x_max + 1)
    means = exog.mean()
    out = pd.DataFrame({"deals_l1_3": grid})
    for name, fit in fits.items():
        rows = np.tile(means.values, (len(grid), 1))
        rows[:, list(exog.columns).index("log_deals_l1_3")] = np.log1p(grid)
        eta = rows @ fit.params.values
        se = np.sqrt(np.einsum("ij,jk,ik->i", rows, fit.cov_params().values, rows))
        out[name] = 100 * inv[name](eta)
        out[f"{name}_lo"] = 100 * inv[name](eta - 1.96 * se)
        out[f"{name}_hi"] = 100 * inv[name](eta + 1.96 * se)
    tables["fig3_predicted_hazard"] = out

    fig, (ax, ax_h) = plt.subplots(2, 1, figsize=(8, 5.8), sharex=True,
                                   gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08})
    ends = {name: out[name].iloc[-1] for name in fits}
    gap = 0.06 * max(ends.values())
    label_y, last = {}, -np.inf
    for name in sorted(ends, key=ends.get):  # spread end labels so they don't collide
        label_y[name] = max(ends[name], last + gap)
        last = label_y[name]
    for name, color in zip(fits, [GRAY, BLUE, ORANGE]):
        ax.fill_between(grid, out[f"{name}_lo"], out[f"{name}_hi"], color=color, alpha=0.12,
                        linewidth=0)
        ax.plot(grid, out[name], color=color, linewidth=2)
        ax.text(grid[-1] + 0.25, label_y[name], name, color=INK_2, fontsize=8.5,
                va="center")
    ax.set_ylabel("Annual probability of first\nde-risked renewable deal (%)")
    ax.set_ylim(0, None)
    ax.set_xlim(-0.4, x_max + 1.6)
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)
    counts = d.loc[d["deals_l1_3"] <= x_max, "deals_l1_3"].value_counts().sort_index()
    tail = int((d["deals_l1_3"] > x_max).sum())
    ax_h.bar(counts.index, counts.values, width=0.85, color=AXIS, edgecolor=SURFACE)
    ax_h.set_ylabel("Country-\nyears", fontsize=8)
    ax_h.set_xlabel("Renewable deals in the previous three years (t−1 to t−3)")
    ax_h.grid(axis="x", visible=False)
    ax_h.tick_params(length=0)
    zero = 100 * (d["deals_l1_3"] == 0).mean()
    titled(fig, ax, "Recent deal flow raises the chance of de-risking",
           "Predicted annual probability with 95% bands, other variables at their means",
           f"Hazard model with year, region and income-group FE, SEs clustered by country; "
           f"common sample of {len(d):,} country-years ({int(d['event'].sum())} events). "
           f"{zero:.0f}% of country-years have no deals in the previous three years; the "
           f"x-axis stops at the 95th percentile ({x_max} deals); {tail} country-years with more "
           "are not shown.", note_y=0.0)
    save(fig, "fig3_predicted_hazard")


# ---------------------------------------------------------------------------- figure 4
def glm_ame(fit, exog: pd.DataFrame, var: str, link: str):
    """Average marginal effect of `var` in a logit or cloglog GLM, with a delta-method SE."""
    def ame(params):
        eta = exog.values @ params
        dens = (np.exp(-eta) / (1 + np.exp(-eta)) ** 2 if link == "logit"
                else np.exp(eta) * np.exp(-np.exp(eta)))
        return dens.mean() * params[list(exog.columns).index(var)]
    p = fit.params.values
    est = ame(p)
    grad = np.array([(ame(p + h) - ame(p - h)) / (2 * 1e-6)
                     for h in np.eye(len(p)) * 1e-6])
    return est, float(np.sqrt(grad @ fit.cov_params().values @ grad))


def fig4(risk: pd.DataFrame):
    gov = ["reg_quality_l1", "vulnerability_l1"]
    macro = ["log_gdppc_l1", "gdp_growth_l1", "inflation_l1", "depreciation_l1",
             "private_credit_l1"]
    dur = hz.DUR_TERMS
    for term, lab in zip(dur, hz.DURATION_LABELS[1:]):
        risk[term] = (risk["dur_bin"] == lab).astype(float)
    full_sample = risk.dropna(subset=hz.X + gov + macro)
    fe = "year + region + income_group"
    lpm_specs = [
        ("Baseline", risk, [], fe),
        ("Baseline, full-controls sample", full_sample, [], fe),
        ("+ WGI and ND-GAIN", risk, gov, fe),
        ("+ macro and finance", risk, gov + macro, fe),
        ("+ years-since-entry dummies", risk, gov + macro + dur, fe),
        ("Region × year FE", risk, gov + macro + dur, "region^year"),
    ]
    rows = []
    for label, data, extra, fes in lpm_specs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = pf.feols(f"event ~ {' + '.join(hz.X + extra)} | {fes}", data=data,
                           vcov={"CRV1": "iso3"})
        t = fit.tidy().loc["log_deals_l1_3"]
        rows.append({"label": label, "est": t["Estimate"], "se": t["Std. Error"],
                     "p": t["Pr(>|t|)"], "N": fit._N, "events": int(fit._data["event"].sum())})
    # Logit and cloglog: average marginal effects, same regressors as the last LPMs
    d, exog = hz.fe_design(risk, ["year", "region", "income_group", "dur_bin"], gov + macro)
    groups = pd.factorize(d["iso3"])[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for label, family, link in [
                ("Logit, marginal effect", sm.families.Binomial(), "logit"),
                ("Cloglog, marginal effect",
                 sm.families.Binomial(link=sm.families.links.CLogLog()), "cloglog")]:
            fit = sm.GLM(d["event"], exog, family=family).fit(
                cov_type="cluster", cov_kwds={"groups": groups})
            est, se = glm_ame(fit, exog, "log_deals_l1_3", link)
            rows.append({"label": label, "est": est, "se": se,
                         "p": 2 * stats.norm.sf(abs(est / se)), "N": len(d),
                         "events": int(d["event"].sum())})
    r = pd.DataFrame(rows)
    for col in ["est", "se"]:
        r[col] *= 100
    r["lo"], r["hi"] = r["est"] - 1.96 * r["se"], r["est"] + 1.96 * r["se"]
    r["stars"] = r["p"].map(stars)
    r["label"] = [f"{lab}  (N {n:,}, {e} events)" for lab, n, e in
                  zip(r["label"], r["N"], r["events"])]
    tables["fig4_specifications"] = r
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    coef_rows(ax, r)
    ax.set_xlabel("Change in annual probability of first de-risking (percentage points)\n"
                  "per log-point of renewable deals in t−1 to t−3")
    titled(fig, ax, "The result is stable across specifications",
           "Deal-flow coefficient (LPM) or average marginal effect (logit, cloglog), "
           "95% intervals clustered by country",
           "All rows have year, region and income-group FE except 'Region × year FE'. "
           "Controls are lagged one year: WGI regulatory quality, ND-GAIN vulnerability; log "
           "GDP per capita, GDP growth, inflation, depreciation, private credit/GDP. Rows "
           "below '+ macro and finance' keep all controls. Logit and cloglog drop FE cells "
           "with no event.", note_y=-0.04)
    save(fig, "fig4_specifications")


# ---------------------------------------------------------------------------- figure 5
def fig5(cy: pd.DataFrame):
    d = lp.prepare(cy)
    sd = {v: d[v].std() for v in ["ldeals", "derisk"]}
    lags = " + ".join(f"{v}_l{k}" for v in ["ldeals", "derisk"] for k in range(1, lp.LAGS + 1))
    rows = []
    for label, outcome, shock in lp.DIRECTIONS:
        scale = sd[shock] / sd[outcome]
        for h in lp.HORIZONS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = pf.feols(f"{outcome}_f{h} ~ {shock} + {lags} | iso3 + year", data=d,
                               vcov={"CRV1": "iso3"})
            t = fit.tidy().loc[shock]
            rows.append({"direction": label, "h": h, "std_coef": t["Estimate"] * scale,
                         "std_lo": t["2.5%"] * scale, "std_hi": t["97.5%"] * scale,
                         "p": t["Pr(>|t|)"], "N": fit._N})
    res = pd.DataFrame(rows)
    tables["fig5_local_projections"] = res
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (label, _, _), title, color in zip(
            axes, lp.DIRECTIONS, ["Market → de-risking\n(response of de-risking to deals)",
                                  "De-risking → market\n(response of deals to de-risking)"],
            [BLUE, ORANGE]):
        r = res[res["direction"] == label]
        ax.fill_between(r["h"], r["std_lo"], r["std_hi"], color=color, alpha=0.15,
                        linewidth=0)
        ax.plot(r["h"], r["std_coef"], color=color, linewidth=2, marker="o", markersize=5.5,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.set_xticks(list(lp.HORIZONS))
        ax.set_xlabel("Years after the shock (h)")
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.grid(axis="x", visible=False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Response in SDs of the outcome\nper 1-SD shock")
    fig.suptitle("The market leads more strongly than it follows", x=0.06, ha="left",
                 y=1.07, fontsize=12, fontweight="bold", color=INK)
    fig.text(0.06, 0.99, "Local projections with 95% bands; same standardised units and "
             "y-axis in both panels", fontsize=9, color=INK_2, ha="left")
    fig.text(0.01, -0.04, "Deals = log(1 + renewable deals without de-risking); de-risking = "
             "any de-risked renewable deal that year. Country and year FE, two lags of both, "
             "SEs clustered by country.", fontsize=8, color=MUTED, ha="left", va="top")
    save(fig, "fig5_local_projections")


# ---------------------------------------------------------------------------- appendix
def figA1(risk: pd.DataFrame):
    controls = {"reg_quality_l1": "WGI regulatory quality",
                "vulnerability_l1": "ND-GAIN vulnerability",
                "inflation_l1": "Inflation", "depreciation_l1": "Currency depreciation",
                "private_credit_l1": "Private credit / GDP", "gdp_growth_l1": "GDP growth",
                "electricity_l1": "Electricity access"}
    data = risk.dropna(subset=hz.X + list(controls))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = pf.feols(f"event ~ {' + '.join(hz.X + list(controls))} | year + region + "
                       "income_group", data=data, vcov={"CRV1": "iso3"})
    t = fit.tidy()
    sd = data[list(controls)].std()
    r = pd.DataFrame({"label": list(controls.values()),
                      "est": 100 * t.loc[list(controls), "Estimate"].values * sd.values,
                      "se": 100 * t.loc[list(controls), "Std. Error"].values * sd.values,
                      "p": t.loc[list(controls), "Pr(>|t|)"].values,
                      "sd": sd.values})
    r["lo"], r["hi"] = r["est"] - 1.96 * r["se"], r["est"] + 1.96 * r["se"]
    r["stars"] = r["p"].map(stars)
    tables["figA1_controls"] = r
    fig, ax = plt.subplots(figsize=(8, 4.2))
    coef_rows(ax, r)
    ax.set_xlabel("Change in annual probability of first de-risking (percentage points) "
                  "per 1-SD increase")
    titled(fig, ax, "Country conditions barely predict where de-risking goes",
           "Standardised coefficients of country conditions at t−1 in the hazard model, "
           "95% intervals",
           f"LPM with the deal-flow and policy variables, year, region and income-group FE; "
           f"SEs clustered by country; N {fit._N:,}, {int(fit._data['event'].sum())} events. "
           "Inflation and depreciation are winsorised at 1%/99%.", note_y=-0.02)
    save(fig, "figA1_where_derisking_goes")


def kaplan_meier(time: np.ndarray, event: np.ndarray):
    """KM survival with Greenwood 95% bands at each distinct time."""
    order = np.unique(time)
    surv, var_sum, rows = 1.0, 0.0, []
    for t in order:
        at_risk = (time >= t).sum()
        d = ((time == t) & (event == 1)).sum()
        if d:
            surv *= 1 - d / at_risk
            var_sum += d / (at_risk * (at_risk - d)) if at_risk > d else 0
        se = surv * np.sqrt(var_sum)
        rows.append((t, surv, max(surv - 1.96 * se, 0), min(surv + 1.96 * se, 1), at_risk))
    return pd.DataFrame(rows, columns=["t", "S", "lo", "hi", "at_risk"])


def logrank(time, event, group):
    times = np.unique(time[event == 1])
    o_minus_e, var = 0.0, 0.0
    for t in times:
        at_risk = time >= t
        n, n1 = at_risk.sum(), (at_risk & (group == 1)).sum()
        d = ((time == t) & (event == 1)).sum()
        d1 = ((time == t) & (event == 1) & (group == 1)).sum()
        o_minus_e += d1 - d * n1 / n
        if n > 1:
            var += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    chi2 = o_minus_e ** 2 / var
    return chi2, stats.chi2.sf(chi2, 1)


def figA2(cy: pd.DataFrame):
    """Landmark Kaplan-Meier: countries still not de-risked at the end of their third year
    in the market, split by renewable deals in those first three years; clock from year 3."""
    last = cy["year"].max()
    c = cy.groupby("iso3")[["entry_year", "treat_year"]].first()
    c = c[c["entry_year"] <= last - 2]
    early = {iso: cy[(cy["iso3"] == iso) & cy["year"].between(e, e + 2)]["n_ren"].sum()
             for iso, e in c["entry_year"].items()}
    c["early_deals"] = pd.Series(early)
    c["time"] = np.where(c["treat_year"].notna(), c["treat_year"] - c["entry_year"],
                         last - c["entry_year"])
    c["event"] = c["treat_year"].notna().astype(int)
    c = c[c["time"] >= 3]  # landmark: still at risk after the first three years
    cut = c["early_deals"].median()
    c["high"] = (c["early_deals"] > cut).astype(int)
    labels = ({1: "At least one renewable deal\nin the first three years",
               0: "No renewable deal\nin the first three years"} if cut == 0 else
              {1: f"High early deal flow (> {cut:g} deals)",
               0: f"Low early deal flow (≤ {cut:g} deals)"})
    tables["figA2_km_countries"] = c
    chi2, p = logrank(c["time"].values, c["event"].values, c["high"].values)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    end_y = {}
    for grp, color in [(1, BLUE), (0, ORANGE)]:
        g = c[c["high"] == grp]
        km = kaplan_meier(g["time"].values, g["event"].values)
        km = pd.concat([pd.DataFrame({"t": [3], "S": [1.0], "lo": [1.0], "hi": [1.0],
                                      "at_risk": [len(g)]}), km[km["t"] >= 3]])
        tables[f"figA2_km_{'high' if grp else 'low'}"] = km
        ax.step(km["t"], 100 * km["S"], where="post", color=color, linewidth=2)
        ax.fill_between(km["t"], 100 * km["lo"], 100 * km["hi"], step="post", color=color,
                        alpha=0.12, linewidth=0)
        end_y[grp] = (km["t"].iloc[-1], 100 * km["S"].iloc[-1],
                      f"{labels[grp]}\nn = {len(g)}, {int(g['event'].sum())} de-risked")
    if abs(end_y[1][1] - end_y[0][1]) < 12:  # keep the two end labels apart
        hi_grp = 1 if end_y[1][1] >= end_y[0][1] else 0
        mid = (end_y[1][1] + end_y[0][1]) / 2
        end_y[hi_grp] = (end_y[hi_grp][0], mid + 6, end_y[hi_grp][2])
        end_y[1 - hi_grp] = (end_y[1 - hi_grp][0], mid - 6, end_y[1 - hi_grp][2])
    for grp, (x, yv, text) in end_y.items():
        ax.text(max(e[0] for e in end_y.values()) + 0.6, yv, text, fontsize=8, color=INK_2,
                va="center")
    ax.set_xlim(3, c["time"].max() + 9)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Years since entering the market (first energy PPI deal)")
    ax.set_ylabel("Countries not yet de-risked (%)")
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)
    titled(fig, ax, "Countries with early deals tend to be de-risked sooner",
           "Kaplan-Meier survival with 95% bands, from the end of the third year in the "
           "market",
           f"Landmark analysis: countries still not de-risked after their first three years "
           f"(entered by {last - 2}), split at the median of renewable deals in those three "
           f"years. Log-rank test: chi2(1) = {chi2:.2f}, p = {p:.3f} — the difference is not "
           "statistically significant.", note_y=-0.02)
    save(fig, "figA2_survival_curves")
    return chi2, p


def figA3(risk: pd.DataFrame):
    data = risk.dropna(subset=hz.X).copy()
    ref = int(data["year"].min())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = pf.feols(f"event ~ {' + '.join(hz.X)} + i(year, ref={ref}) | region + "
                       "income_group", data=data, vcov={"CRV1": "iso3"})
        for term, lab in zip(hz.DUR_TERMS, hz.DURATION_LABELS[1:]):
            data[term] = (data["dur_bin"] == lab).astype(float)
        fit_d = pf.feols(f"event ~ {' + '.join(hz.X + hz.DUR_TERMS)} | year + region + "
                         "income_group", data=data, vcov={"CRV1": "iso3"})
    t = fit.tidy()
    yr = t[t.index.str.startswith("C(year")].copy() if t.index.str.startswith("C(year").any() \
        else t[t.index.str.contains("year::")].copy()
    yr["year"] = yr.index.str.extract(r"(\d{4})", expand=False).astype(int).values
    yr = pd.concat([yr, pd.DataFrame({"year": [ref], "Estimate": [0.0], "2.5%": [0.0],
                                      "97.5%": [0.0]})]).sort_values("year")
    td = fit_d.tidy().loc[hz.DUR_TERMS]
    dur = pd.DataFrame({"label": ["0–2 (reference)", "3–5", "6–10", "11+"],
                        "est": [0, *td["Estimate"]], "lo": [0, *td["2.5%"]],
                        "hi": [0, *td["97.5%"]]})
    tables["figA3_year_effects"] = yr[["year", "Estimate", "2.5%", "97.5%"]]
    tables["figA3_duration_effects"] = dur
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"width_ratios": [3, 1.2], "wspace": 0.25})
    ax1.axvspan(2017, yr["year"].max() + 0.5, color="#f0efec", zorder=0)
    ax1.text(2016.7, 100 * yr["97.5%"].max(), "IDA Private Sector\nWindow era (2017–) →",
             fontsize=7.5, color=MUTED, va="top", ha="right")
    ax1.axhline(0, color=AXIS, linewidth=0.8)
    ax1.vlines(yr["year"], 100 * yr["2.5%"], 100 * yr["97.5%"], color=BLUE, alpha=0.5,
               linewidth=1.5)
    ax1.plot(yr["year"], 100 * yr["Estimate"], "o-", color=BLUE, markersize=4,
             linewidth=1.5, markeredgecolor=SURFACE, markeredgewidth=1)
    ax1.set_ylabel(f"Year effect vs {ref} (percentage points)")
    ax1.set_xlabel("Year")
    ax1.grid(axis="x", visible=False)
    ax1.tick_params(length=0)
    x = np.arange(len(dur))
    ax2.axhline(0, color=AXIS, linewidth=0.8)
    ax2.vlines(x, 100 * dur["lo"], 100 * dur["hi"], color=ORANGE, alpha=0.6, linewidth=2)
    ax2.plot(x, 100 * dur["est"], "o", color=ORANGE, markersize=7, markeredgecolor=SURFACE,
             markeredgewidth=1.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(dur["label"], fontsize=8)
    ax2.set_xlabel("Years since entry")
    ax2.set_title("Duration effects", loc="left", fontsize=10, color=INK)
    ax2.grid(axis="x", visible=False)
    ax2.tick_params(length=0)
    titled(fig, ax1, "Baseline hazard over calendar time and time in the market",
           "Year effects (left) and years-since-entry effects (right) from LPM hazard "
           "models, 95% intervals",
           "Left: year dummies with region and income-group FE and the deal-flow and policy "
           f"variables; reference {ref}. Right: duration dummies added to the model with year, "
           "region and income-group FE. SEs clustered by country.", note_y=-0.02)
    save(fig, "figA3_baseline_hazard")


def figA4(cy: pd.DataFrame):
    import geopandas as gpd
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    world.loc[world["name"] == "Kosovo", "iso_a3"] = "XKX"
    world = world[world["continent"] != "Antarctica"]
    c = cy.groupby("iso3")["treat_year"].first()
    bins = [(1990, 2004), (2005, 2011), (2012, 2016), (2017, 2020), (2021, 2024)]
    world["treat_year"] = world["iso_a3"].map(c)
    world["in_panel"] = world["iso_a3"].isin(c.index)
    in_map = set(world["iso_a3"])
    tables["figA4_map"] = pd.DataFrame({"treat_year": c, "on_map": c.index.isin(in_map)})
    fig, ax = plt.subplots(figsize=(11, 5.0))
    world.plot(ax=ax, color="#f3f2ee", edgecolor="#d8d6cf", linewidth=0.3)
    world[world["in_panel"] & world["treat_year"].isna()].plot(
        ax=ax, color="#bdbbb3", edgecolor=SURFACE, linewidth=0.3)
    handles = []
    for (lo, hi), color in zip(bins, RAMP):
        sel = world["treat_year"].between(lo, hi)
        if sel.any():
            world[sel].plot(ax=ax, color=color, edgecolor=SURFACE, linewidth=0.3)
        n = int(c.between(lo, hi).sum())
        handles.append(Patch(color=color, label=f"{lo}–{hi}  ({n})"))
    never = int((c.isna()).sum())
    handles += [Patch(color="#bdbbb3", label=f"In PPI, never de-risked  ({never})"),
                Patch(facecolor="#f3f2ee", edgecolor="#d8d6cf", label="Not in the PPI sample")]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5,
              title="First de-risked renewable deal", title_fontsize=9)
    ax.set_axis_off()
    missing = sorted(set(c.index) - in_map)
    titled(fig, ax, "Where and when de-risking arrived",
           "Year of each country's first de-risked renewable deal (number of countries in "
           "brackets)",
           "Boundaries: Natural Earth 1:110m (small states not shown"
           + (f": {', '.join(missing)}" if missing else "") + ").", note_y=0.1)
    save(fig, "figA4_map")


def figA5():
    p = pd.read_csv(PROJECT_PANEL, low_memory=False)
    r = p[p["technology_category"] == "renewable"].copy()
    r["tech"] = r["tech"].astype(str)
    rows = []
    for label, fe in [("Year FE", "year + tech"), ("Country + year FE", "iso3 + year + tech"),
                      ("Country × year FE", "iso3^year + tech")]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = pf.feols(f"public_finance ~ derisk | {fe}", data=r, vcov={"CRV1": "iso3"})
        t = fit.tidy().loc["derisk"]
        rows.append({"label": f"{label}  (N {fit._N:,})", "est": 100 * t["Estimate"],
                     "lo": 100 * t["2.5%"], "hi": 100 * t["97.5%"],
                     "stars": stars(t["Pr(>|t|)"])})
    raw = 100 * (r.loc[r["derisk"] == 1, "public_finance"].mean()
                 - r.loc[r["derisk"] == 0, "public_finance"].mean())
    rows.insert(0, {"label": "Raw difference", "est": raw, "lo": raw, "hi": raw, "stars": ""})
    df = pd.DataFrame(rows)
    tables["figA5_stacking"] = df
    fig, ax = plt.subplots(figsize=(8, 3.4))
    coef_rows(ax, df, unit_fmt="{:+.0f} pp")
    ax.set_xlabel("Difference in the probability of public finance, de-risked vs other "
                  "renewable deals (percentage points)")
    ax.set_xlim(0, None)
    titled(fig, ax, "How much of the gap is deal-level packaging",
           "Public finance on de-risking, renewable projects, 95% intervals",
           "All models include technology FE; SEs clustered by country. The country × year "
           "estimate compares deals in the same country and year.", note_y=-0.04)
    save(fig, "figA5_stacking")


def main():
    print("=" * 70)
    print("Step 13: paper figures")
    print("=" * 70)
    for path in (COUNTRY_YEAR_PANEL, PROJECT_PANEL):
        if not path.exists():
            sys.exit(f"{path} not found — run the earlier steps first.")
    cy = pd.read_csv(COUNTRY_YEAR_PANEL)
    risk = hz.build_sample(cy)
    fig1(cy)
    med = fig2(cy)
    fig3(risk)
    fig4(risk.copy())
    fig5(cy)
    figA1(risk)
    chi2, p = figA2(cy)
    figA3(risk.copy())
    figA4(cy)
    figA5()
    with pd.ExcelWriter(FIG_DIR / "paper_figure_data.xlsx") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name[:31])
    print(f"[FIG] Figure data: {FIG_DIR / 'paper_figure_data.xlsx'}")
    print(f"[FIG] Median gap first renewable -> first de-risked deal: {med:g} years; "
          f"KM log-rank p = {p:.3f}")


if __name__ == "__main__":
    main()
