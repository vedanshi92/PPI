"""
step12_stacked_did.py
---------------------
Shape-matched stacked difference-in-differences: does the first de-risked renewable deal
accelerate the market (renewable deals WITHOUT de-risking, n_ren_nd)?

1. Events: treated countries whose treat_year T has T-5 >= 1991 and T+5 <= 2024.
2. Candidate controls for each event: countries not treated by T+5 (or never treated) that
   entered PPI by T-1 (entry_year <= T-1).
3. Shape matching on log(1 + n_ren) over T-5..T-1, each path demeaned by its own mean;
   Euclidean distance to the treated path, +1 if the region differs. If the treated
   country had any pre-period deals, controls must also have some. Keep the 3 nearest
   (controls can serve several events).
4. Stack T-5..T+5 for the treated country and its controls: event id, unit id (event x
   country), treated, rel = year - T.
5. Poisson of n_ren_nd on rel x treated (reference rel = -1), unit FE and event x year FE,
   clustered by country; and a pooled treated x post version.
6. Pre-trend check: joint Wald test that all pre-period coefficients are zero.

Writes data/processed/stacked_did.txt, data/interim/stacked_did_matches.csv and
figures/9_stacked_did.png.

Run from the project root:  python3 src/step12_stacked_did.py
"""

import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf

from config import COUNTRY_YEAR_PANEL, INTERIM, OUT, STACKED_DID_TXT

WINDOW = 5
N_CONTROLS = 3
REGION_PENALTY = 1.0


def match_events(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    first, last = panel["year"].min(), panel["year"].max()
    countries = panel.groupby("iso3")[["treat_year", "entry_year", "region"]].first()
    ldeal = np.log1p(panel.pivot(index="iso3", columns="year", values="n_ren"))
    deals = panel.pivot(index="iso3", columns="year", values="n_ren")

    treated = countries.dropna(subset=["treat_year"])
    treated = treated[(treated["treat_year"] - WINDOW >= first + 1)
                      & (treated["treat_year"] + WINDOW <= last)]
    matches, stacks = [], []
    for event_id, (iso, row) in enumerate(treated.sort_values("treat_year").iterrows(), 1):
        T = int(row["treat_year"])
        pre = list(range(T - WINDOW, T))
        path = ldeal.loc[iso, pre]
        shape = path - path.mean()
        pool = countries[(countries.index != iso)
                         & (countries["treat_year"].isna() | (countries["treat_year"] > T + WINDOW))
                         & (countries["entry_year"] <= T - 1)]
        if deals.loc[iso, pre].sum() > 0:
            pool = pool[deals.loc[pool.index, pre].sum(axis=1) > 0]
        cand = ldeal.loc[pool.index, pre]
        cand_shape = cand.sub(cand.mean(axis=1), axis=0)
        dist = np.sqrt(((cand_shape - shape.values) ** 2).sum(axis=1))
        dist += REGION_PENALTY * (pool["region"] != row["region"]).astype(float)
        chosen = dist.nsmallest(N_CONTROLS)
        for rank, (ciso, d) in enumerate(chosen.items(), 1):
            matches.append({"event_id": event_id, "treated_iso3": iso, "treat_year": T,
                            "control_iso3": ciso, "rank": rank, "distance": round(d, 4),
                            "same_region": pool.at[ciso, "region"] == row["region"]})
        for unit_iso, is_treated in [(iso, 1)] + [(c, 0) for c in chosen.index]:
            block = panel[(panel["iso3"] == unit_iso)
                          & panel["year"].between(T - WINDOW, T + WINDOW)].copy()
            block["event_id"] = event_id
            block["unit_id"] = f"{event_id}_{unit_iso}"
            block["treated"] = is_treated
            block["rel"] = block["year"] - T
            stacks.append(block)
    return pd.DataFrame(matches), pd.concat(stacks, ignore_index=True)


def main():
    print("=" * 70)
    print("Step 12: shape-matched stacked DiD — does de-risking accelerate the market?")
    print("=" * 70)
    if not COUNTRY_YEAR_PANEL.exists():
        sys.exit(f"{COUNTRY_YEAR_PANEL} not found — run step9_country_year_panel.py first.")
    panel = pd.read_csv(COUNTRY_YEAR_PANEL)
    matches, stack = match_events(panel)
    matches.to_csv(INTERIM / "stacked_did_matches.csv", index=False)

    stack["post"] = (stack["rel"] >= 0).astype(int)
    stack["treated_post"] = stack["treated"] * stack["post"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        es = pf.fepois("n_ren_nd ~ i(rel, treated, ref=-1) | unit_id + event_id^year",
                       data=stack, vcov={"CRV1": "iso3"})
        pooled = pf.fepois("n_ren_nd ~ treated_post | unit_id + event_id^year",
                           data=stack, vcov={"CRV1": "iso3"})

    tidy = es.tidy()
    tidy.index = [int(i.split("::")[1].split(":")[0]) for i in tidy.index]
    tidy = tidy.sort_index()
    beta = pd.Series(tidy["Estimate"].values, index=[f"rel{r}" for r in tidy.index])
    vcov = pd.DataFrame(es._vcov, index=beta.index, columns=beta.index)
    pre = [r for r in tidy.index if r < -1]

    # Joint test that all pre-period coefficients are zero (Wald, chi2 with len(pre) df)
    from scipy import stats
    b_pre = beta[[f"rel{r}" for r in pre]].values
    v_pre = vcov.loc[[f"rel{r}" for r in pre], [f"rel{r}" for r in pre]].values
    wald = float(b_pre @ np.linalg.solve(v_pre, b_pre))
    wald_p = 1 - stats.chi2.cdf(wald, len(pre))

    p = pooled.tidy().loc["treated_post"]
    n_events = matches["event_id"].nunique()
    lines = [
        "Shape-matched stacked DiD: does the first de-risked renewable deal accelerate the market?",
        "Outcome: n_ren_nd (renewable deals without de-risking). Poisson, unit FE + event x year FE,",
        "SEs clustered by country. Coefficients are log changes; exp(b) - 1 is the % change.",
        "",
        f"Events: {n_events} treated countries (T-5 >= 1991, T+5 <= 2024), treat years "
        f"{matches['treat_year'].min()}-{matches['treat_year'].max()}.",
        f"Controls: {N_CONTROLS} per event, {matches['control_iso3'].nunique()} distinct countries; "
        f"{int(matches['same_region'].sum())} of {len(matches)} matches in the same region; "
        f"median distance {matches['distance'].median():.3f}.",
        f"Stacked sample: {len(stack):,} unit-years; used in Poisson: {es._N:,} (units with all-zero "
        "outcomes drop out).",
        "",
        "Pooled: n_ren_nd ~ treated x post",
        f"  coef {p['Estimate']:.4f} (SE {p['Std. Error']:.4f}), p = {p['Pr(>|t|)']:.4f}  ->  "
        f"{100 * (np.exp(p['Estimate']) - 1):+.1f}% deals after treatment",
        "",
        "Event study (reference rel = -1):",
        tidy[["Estimate", "Std. Error", "Pr(>|t|)", "2.5%", "97.5%"]].round(4).to_string(),
        "",
        f"Pre-trend check: joint Wald test that rel {pre[0]}..{pre[-1]} are all zero: "
        f"chi2({len(pre)}) = {wald:.2f}, p = {wald_p:.3f}.",
    ]
    report = "\n".join(lines)
    print(report)
    STACKED_DID_TXT.write_text(report + "\n")
    print(f"\nSaved to {STACKED_DID_TXT}")
    plot(tidy, p)


def plot(tidy: pd.DataFrame, pooled: pd.Series):
    surface, ink, ink2, muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    blue = "#2a78d6"
    rel = list(tidy.index) + [-1]
    est = list(tidy["Estimate"]) + [0.0]
    lo = list(tidy["2.5%"]) + [0.0]
    hi = list(tidy["97.5%"]) + [0.0]
    order = np.argsort(rel)
    rel, est, lo, hi = (np.array(v)[order] for v in (rel, est, lo, hi))
    fig, ax = plt.subplots(figsize=(7.5, 4.4), facecolor=surface)
    ax.set_facecolor(surface)
    ax.axvspan(-0.5, max(rel) + 0.5, color="#f0efec", zorder=0)
    ax.axhline(0, color="#c3c2b7", linewidth=0.8)
    ax.vlines(rel, lo, hi, color=blue, linewidth=1.5, alpha=0.6)
    ax.plot(rel, est, color=blue, linewidth=2, marker="o", markersize=6,
            markeredgecolor=surface, markeredgewidth=1.5)
    ax.annotate("reference", (-1, 0), textcoords="offset points", xytext=(0, -14),
                ha="center", fontsize=8, color=muted)
    ax.set_xticks(rel)
    ax.set_xlabel("Years relative to first de-risked renewable deal", color=ink2)
    ax.set_ylabel("Effect on log deals without de-risking", color=ink2)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=muted, labelcolor=ink2, length=0)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Does de-risking accelerate the market?", loc="left", fontsize=12,
                 fontweight="bold", color=ink, pad=24)
    ax.text(0, 1.02, f"Shape-matched stacked DiD, Poisson event study with 95% CIs. Pooled post "
            f"effect: {pooled['Estimate']:+.2f} (p = {pooled['Pr(>|t|)']:.3f})",
            transform=ax.transAxes, fontsize=9, color=ink2, va="bottom")
    out = OUT / "figures" / "9_stacked_did.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=surface)
    plt.close(fig)
    print(f"[FIG] {out}")


if __name__ == "__main__":
    main()
