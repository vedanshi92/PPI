"""
step10_hazard.py
----------------
Discrete-time hazard model: does de-risking follow the market? The outcome is a country's
FIRST de-risked renewable deal.

Sample: country-years from max(entry_year, 1993) up to and including treat_year. Countries
leave the risk set once treated; never-treated countries stay to the last panel year.
1993 is the first year with a full three-year deal history (deals_l1_3).

  event = 1 in treat_year, else 0
  regressors: log(1 + deals_l1_3), deals_growth, prior_energy_derisk, fit_active,
              nm_active, auction_active

Specifications (SEs clustered by country):
  (1) LPM, year FE
  (2) LPM, year + region + income-group FE
  (3) LPM, region x year FE
  (4) Logit, year FE — average marginal effects
  (5) Logit, year + region + income-group FE — average marginal effects

Rows with a missing regressor are dropped (deals_growth needs t-1..t-6, so it starts in
1996; fit_active / nm_active are blank where REN21 gives no date).

Writes data/processed/hazard_model.txt.

Run from the project root:  python3 src/step10_hazard.py
"""

import sys
import warnings

import numpy as np
import pandas as pd
import pyfixest as pf
import statsmodels.api as sm

from config import COUNTRY_YEAR_PANEL, HAZARD_TXT

START_YEAR = 1993
X = ["log_deals_l1_3", "deals_growth", "prior_energy_derisk", "fit_active", "nm_active",
     "auction_active"]


def build_sample(panel: pd.DataFrame) -> pd.DataFrame:
    start = np.maximum(panel["entry_year"].astype(float), START_YEAR)
    end = panel["treat_year"].astype(float).fillna(panel["year"].max())
    risk = panel[(panel["year"] >= start) & (panel["year"] <= end)].copy()
    risk["event"] = (risk["year"] == risk["treat_year"]).astype(int)
    risk["log_deals_l1_3"] = np.log1p(risk["deals_l1_3"].astype(float))
    for col in X:
        risk[col] = risk[col].astype(float)
    return risk


def logit_ame(data: pd.DataFrame, fe: list[str]) -> tuple[pd.DataFrame, int, int]:
    """Logit with FE dummies, SEs clustered by country; returns average marginal effects.
    FE cells with no event (e.g. a year with no first de-risking) predict 0 perfectly, so
    they are dropped first, as a conditional model would."""
    d = data.dropna(subset=["event", *X]).copy()
    for f in fe:
        d = d[d.groupby(f)["event"].transform("max") > 0]
    dummies = pd.get_dummies(d[fe].astype(str), drop_first=True, dtype=float)
    exog = sm.add_constant(pd.concat([d[X], dummies], axis=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.Logit(d["event"], exog).fit(
            disp=0, maxiter=200, cov_type="cluster",
            cov_kwds={"groups": pd.factorize(d["iso3"])[0]})
    ame = fit.get_margeff(at="overall").summary_frame()
    ame = ame.loc[X].rename(columns={"dy/dx": "AME", "Std. Err.": "Std. Error",
                                     "Pr(>|z|)": "p", "Conf. Int. Low": "2.5%",
                                     "Cont. Int. Hi.": "97.5%"})
    return ame[["AME", "Std. Error", "p", "2.5%", "97.5%"]], len(d), int(d["event"].sum())


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def main():
    print("=" * 70)
    print("Step 10: hazard model — does de-risking follow the market?")
    print("=" * 70)
    if not COUNTRY_YEAR_PANEL.exists():
        sys.exit(f"{COUNTRY_YEAR_PANEL} not found — run step9_country_year_panel.py first.")
    risk = build_sample(pd.read_csv(COUNTRY_YEAR_PANEL))
    complete = risk.dropna(subset=X)

    lines = [
        "Discrete-time hazard: first de-risked renewable deal (event = 1 in treat_year)",
        f"Risk set: country-years from max(entry_year, {START_YEAR}) to treat_year (or "
        f"{int(risk['year'].max())} if never treated). SEs clustered by country.",
        "",
        f"Risk set: {len(risk):,} country-years, {risk['iso3'].nunique()} countries, "
        f"{int(risk['event'].sum())} events (first de-risked renewable deal).",
        f"With all regressors: {len(complete):,} country-years, {complete['iso3'].nunique()} "
        f"countries, {int(complete['event'].sum())} events.",
        "Dropped for a missing regressor: "
        + ", ".join(f"{c} {int(risk[c].isna().sum())}" for c in X if risk[c].isna().any()),
        "",
        "Means in the estimation sample (event = 1 vs 0):",
        complete.groupby("event")[X].mean().round(3).T.to_string(),
    ]

    fe_specs = [("(1) LPM, year FE", "year"),
                ("(2) LPM, year + region + income FE", "year + region + income_group"),
                ("(3) LPM, region x year FE", "region^year")]
    fits = []
    for label, fe in fe_specs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = pf.feols(f"event ~ {' + '.join(X)} | {fe}", data=risk, vcov={"CRV1": "iso3"})
        fits.append((label, fit))

    logits = [("(4) Logit AME, year FE", ["year"]),
              ("(5) Logit AME, year + region + income FE", ["year", "region", "income_group"])]
    logit_results = [(label, *logit_ame(risk, fe)) for label, fe in logits]

    # Side-by-side table: coefficient (SE) with stars
    table = pd.DataFrame(index=X)
    n_row, ev_row = {}, {}
    for label, fit in fits:
        t = fit.tidy()
        table[label] = [f"{t.at[x, 'Estimate']:.4f}{stars(t.at[x, 'Pr(>|t|)'])} "
                        f"({t.at[x, 'Std. Error']:.4f})" for x in X]
        n_row[label] = f"{fit._N:,}"
        ev_row[label] = f"{int(fit._data['event'].sum())}"
    for label, ame, n, events in logit_results:
        table[label] = [f"{ame.at[x, 'AME']:.4f}{stars(ame.at[x, 'p'])} "
                        f"({ame.at[x, 'Std. Error']:.4f})" for x in X]
        n_row[label], ev_row[label] = f"{n:,}", f"{events}"
    table.loc["N"] = n_row
    table.loc["events"] = ev_row

    lines += ["", "Coefficients (LPM) / average marginal effects (logit), SE in parentheses.",
              "* p<0.1, ** p<0.05, *** p<0.01. Logit drops FE cells with no event.", ""]
    with pd.option_context("display.width", 250, "display.max_columns", 10,
                           "display.max_colwidth", 30):
        lines.append(table.T.to_string())
    lines += ["", "Full output:"]
    for label, fit in fits:
        lines += ["", f"{label}  —  N {fit._N:,}, R2 {fit._r2:.3f}, within R2 {fit._r2_within:.3f}",
                  fit.tidy().round(4).to_string()]
    for label, ame, n, events in logit_results:
        lines += ["", f"{label}  —  N {n:,}, events {events}", ame.round(4).to_string()]

    report = "\n".join(lines)
    print(report)
    HAZARD_TXT.write_text(report + "\n")
    print(f"\nSaved to {HAZARD_TXT}")


if __name__ == "__main__":
    main()
