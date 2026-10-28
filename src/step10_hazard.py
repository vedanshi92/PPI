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
  (6)-(8) as (2), (3), (5) plus duration dummies: years since entry_year (first energy PPI
          deal), binned 0-2 (reference), 3-5, 6-10, 11+. If deal flow still predicts
          de-risking, it is not just that countries get de-risked after being in the market
          for a while.
  (9)  Complementary log-log (proportional hazards), year + region + income FE
  (10) as (9) plus duration dummies
  (11) Cox proportional hazards, Efron ties, on the same multiple-record data: analysis
       time = years since entry, (start, stop] records, delayed entry for countries that
       entered before 1993; covariates X + region + income dummies

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
# Years since entry: 0-2 (reference), 3-5, 6-10, 11+
DURATION_BINS = [-1, 2, 5, 10, np.inf]
DURATION_LABELS = ["0-2", "3-5", "6-10", "11+"]
DUR_TERMS = [f"dur_{lab}" for lab in DURATION_LABELS[1:]]
X = ["log_deals_l1_3", "deals_growth", "prior_energy_derisk", "fit_active", "nm_active",
     "auction_active"]


def build_sample(panel: pd.DataFrame) -> pd.DataFrame:
    start = np.maximum(panel["entry_year"].astype(float), START_YEAR)
    end = panel["treat_year"].astype(float).fillna(panel["year"].max())
    risk = panel[(panel["year"] >= start) & (panel["year"] <= end)].copy()
    risk["event"] = (risk["year"] == risk["treat_year"]).astype(int)
    risk["log_deals_l1_3"] = np.log1p(risk["deals_l1_3"].astype(float))
    risk["years_since_entry"] = risk["year"] - risk["entry_year"].astype(float)
    risk["dur_bin"] = pd.cut(risk["years_since_entry"], DURATION_BINS, labels=DURATION_LABELS,
                             right=True).astype(str)
    for col in X:
        risk[col] = risk[col].astype(float)
    return risk


def fe_design(data: pd.DataFrame, fe: list[str], extra: list[str] | None = None):
    """Complete cases with FE cells that contain no event dropped (in a logit / cloglog
    they predict 0 perfectly, so a conditional model would drop them too), and the design
    matrix with FE dummies. `extra` regressors (e.g. duration dummies) go after X."""
    d = data.dropna(subset=["event", *X]).copy()
    for f in fe:
        d = d[d.groupby(f)["event"].transform("max") > 0]
    dummies = pd.get_dummies(d[fe].astype(str), drop_first=True, dtype=float)
    cols = [d[X]]
    if extra:
        cols.append(d[extra])
    exog = sm.add_constant(pd.concat(cols + [dummies], axis=1))
    return d, exog


def logit_ame(data: pd.DataFrame, fe: list[str], duration: bool = False):
    """Logit with FE dummies, SEs clustered by country; average marginal effects."""
    d, exog = fe_design(data, fe + (["dur_bin"] if duration else []))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.Logit(d["event"], exog).fit(
            disp=0, maxiter=200, cov_type="cluster",
            cov_kwds={"groups": pd.factorize(d["iso3"])[0]})
    ame = fit.get_margeff(at="overall").summary_frame()
    ame.index = [rename_dur(i) for i in ame.index]
    keep = X + (DUR_TERMS if duration else [])
    ame = ame.loc[keep].rename(columns={"dy/dx": "Estimate", "Std. Err.": "Std. Error",
                                        "Pr(>|z|)": "p", "Conf. Int. Low": "2.5%",
                                        "Cont. Int. Hi.": "97.5%"})
    return ame[["Estimate", "Std. Error", "p", "2.5%", "97.5%"]], len(d), int(d["event"].sum())


def rename_dur(name: str) -> str:
    """dur_bin dummy names from get_dummies ("dur_bin_3-5") -> "dur_3-5"."""
    return name.replace("dur_bin_", "dur_")


def cloglog_fit(data: pd.DataFrame, fe: list[str], duration: bool = False):
    """Binomial GLM with a complementary log-log link (the discrete-time proportional-hazards
    model), FE dummies, SEs clustered by country. Coefficients are log hazard ratios."""
    d, exog = fe_design(data, fe + (["dur_bin"] if duration else []))
    exog.columns = [rename_dur(c) for c in exog.columns]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.GLM(d["event"], exog,
                     family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(d["iso3"])[0]})
    keep = X + (DUR_TERMS if duration else [])
    ci = fit.conf_int().loc[keep]
    out = pd.DataFrame({"Estimate": fit.params[keep], "Std. Error": fit.bse[keep],
                        "p": fit.pvalues[keep], "2.5%": ci[0], "97.5%": ci[1]})
    out["hazard ratio"] = np.exp(out["Estimate"])
    return out, len(d), int(d["event"].sum())


def cox_fit(data: pd.DataFrame):
    """Cox proportional hazards with Efron ties on the multiple-record (counting-process)
    data: each country-year is a (start, stop] interval in years since entry, with delayed
    entry for countries already in the market before 1993. Covariates X + region and income
    dummies; robust SEs clustered by country."""
    d = data.dropna(subset=["event", *X]).copy()
    d["start"] = d["years_since_entry"]
    d["stop"] = d["start"] + 1
    dummies = pd.get_dummies(d[["region", "income_group"]].astype(str), drop_first=True,
                             dtype=float)
    exog = pd.concat([d[X], dummies], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.PHReg(d["stop"], exog, status=d["event"], entry=d["start"],
                       ties="efron").fit(groups=pd.factorize(d["iso3"])[0])
    names = list(exog.columns)
    params = pd.Series(fit.params, index=names)
    bse = pd.Series(fit.bse, index=names)
    pvals = pd.Series(fit.pvalues, index=names)
    out = pd.DataFrame({"Estimate": params[X], "Std. Error": bse[X], "p": pvals[X],
                        "2.5%": params[X] - 1.96 * bse[X], "97.5%": params[X] + 1.96 * bse[X]})
    out["hazard ratio"] = np.exp(out["Estimate"])
    return out, len(d), int(d["event"].sum())


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

    # Duration dummies for the LPMs (reference 0-2 years since entry)
    for term, lab in zip(DUR_TERMS, DURATION_LABELS[1:]):
        risk[term] = (risk["dur_bin"] == lab).astype(float)
    lines += ["", "Events and country-years by years since entry (estimation sample):",
              complete.assign(dur_bin=risk.loc[complete.index, "dur_bin"])
              .groupby("dur_bin")["event"].agg(country_years="size", events="sum")
              .reindex(DURATION_LABELS).to_string()]

    dur = " + ".join(DUR_TERMS)
    lpm_specs = [("(1) LPM, year FE", "", "year"),
                 ("(2) LPM, year + region + income FE", "", "year + region + income_group"),
                 ("(3) LPM, region x year FE", "", "region^year"),
                 ("(6) LPM, (2) + duration", dur, "year + region + income_group"),
                 ("(7) LPM, (3) + duration", dur, "region^year")]
    results = {}
    for label, extra, fe in lpm_specs:
        rhs = " + ".join(X + ([extra] if extra else []))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = pf.feols(f"event ~ {rhs} | {fe}", data=risk, vcov={"CRV1": "iso3"})
        t = fit.tidy().rename(columns={"Pr(>|t|)": "p"})
        results[label] = (t[["Estimate", "Std. Error", "p", "2.5%", "97.5%"]], fit._N,
                          int(fit._data["event"].sum()),
                          f"R2 {fit._r2:.3f}, within R2 {fit._r2_within:.3f}")
    nonlinear = [
        ("(4) Logit AME, year FE", logit_ame, (["year"],), {}),
        ("(5) Logit AME, year + region + income FE", logit_ame,
         (["year", "region", "income_group"],), {}),
        ("(8) Logit AME, (5) + duration", logit_ame,
         (["year", "region", "income_group"],), {"duration": True}),
        ("(9) Cloglog, year + region + income FE", cloglog_fit,
         (["year", "region", "income_group"],), {}),
        ("(10) Cloglog, (9) + duration", cloglog_fit,
         (["year", "region", "income_group"],), {"duration": True}),
        ("(11) Cox (Efron), time since entry, region + income", cox_fit, (), {}),
    ]
    for label, func, args, kwargs in nonlinear:
        est, n, events = func(risk, *args, **kwargs)
        results[label] = (est, n, events, "")
    order = ["(1) LPM, year FE", "(2) LPM, year + region + income FE",
             "(3) LPM, region x year FE", "(4) Logit AME, year FE",
             "(5) Logit AME, year + region + income FE", "(6) LPM, (2) + duration",
             "(7) LPM, (3) + duration", "(8) Logit AME, (5) + duration",
             "(9) Cloglog, year + region + income FE", "(10) Cloglog, (9) + duration",
             "(11) Cox (Efron), time since entry, region + income"]

    # Side-by-side table: estimate (SE) with stars
    rows = X + DUR_TERMS
    table = pd.DataFrame(index=rows + ["N", "events"])
    for label in order:
        est, n, events, _ = results[label]
        table[label] = [f"{est.at[r, 'Estimate']:.4f}{stars(est.at[r, 'p'])} "
                        f"({est.at[r, 'Std. Error']:.4f})" if r in est.index else ""
                        for r in rows] + [f"{n:,}", f"{events}"]

    lines += ["", "Estimates, SE in parentheses. LPM: coefficients; logit: average marginal "
              "effects;", "cloglog and Cox: coefficients = log hazard ratios (hazard ratios in "
              "the full output).", "Duration dummies: years since entry, reference 0-2. "
              "* p<0.1, ** p<0.05, *** p<0.01.", "Logit and cloglog drop FE cells with no "
              "event; Cox uses time since entry as the baseline, so no year FE.", ""]
    with pd.option_context("display.width", 300, "display.max_columns", 20,
                           "display.max_colwidth", 30):
        lines.append(table.T.to_string())
    lines += ["", "Full output:"]
    for label in order:
        est, n, events, note = results[label]
        lines += ["", f"{label}  —  N {n:,}, events {events}" + (f", {note}" if note else ""),
                  est.round(4).to_string()]

    report = "\n".join(lines)
    print(report)
    HAZARD_TXT.write_text(report + "\n")
    print(f"\nSaved to {HAZARD_TXT}")


if __name__ == "__main__":
    main()
