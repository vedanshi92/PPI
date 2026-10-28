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
       entered before 1993; covariates X + region + income dummies; SEs from a
       country-cluster bootstrap
  (A1)-(A6) with country-year controls lagged one year: ND-GAIN vulnerability, WGI
       regulatory quality, log GDP per capita, real interest rate, currency depreciation
       (A1 = same sample, no controls). (B1)-(B6) the same without the real interest rate,
       which is missing for many countries.

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
from scipy import stats

from config import COUNTRY_YEAR_PANEL, HAZARD_TXT

START_YEAR = 1993
# Years since entry: 0-2 (reference), 3-5, 6-10, 11+
DURATION_BINS = [-1, 2, 5, 10, np.inf]
DURATION_LABELS = ["0_2", "3_5", "6_10", "11plus"]  # formula-safe names
DUR_TERMS = [f"dur_{lab}" for lab in DURATION_LABELS[1:]]
COX_BOOT = 500   # country-cluster bootstrap replications for the Cox SEs
COX_SEED = 2024
X = ["log_deals_l1_3", "deals_growth", "prior_energy_derisk", "fit_active", "nm_active",
     "auction_active"]
# Country-year controls, all lagged one year (t-1)
CONTROLS = ["vulnerability_l1", "reg_quality_l1", "log_gdppc_l1", "real_interest_l1",
            "depreciation_l1"]
CONTROLS_NO_RIR = [c for c in CONTROLS if c != "real_interest_l1"]
WINSOR = (0.01, 0.99)  # depreciation is winsorised at these panel quantiles


def add_lagged_controls(panel: pd.DataFrame) -> pd.DataFrame:
    """Country-year controls at t-1, computed on the full panel (so the first risk-set year
    can use the year before it):
      vulnerability_l1  ND-GAIN vulnerability score
      reg_quality_l1    WGI regulatory quality; WGI skips 1997, 1999 and 2001, so interior
                        gaps are filled by linear interpolation within country first
      log_gdppc_l1      log GDP per capita (constant USD)
      real_interest_l1  real interest rate (%)
      depreciation_l1   change in log local-currency units per USD from t-2 to t-1
                        (positive = depreciation), winsorised at the 1st/99th percentiles
                        because currency redenominations create extreme jumps"""
    p = panel.sort_values(["iso3", "year"]).copy()
    g = p.groupby("iso3")
    rq = g["regulatory_quality"].transform(lambda s: s.interpolate(limit_area="inside"))
    log_fx = np.log(p["exchange_rate_lcu_usd"].where(p["exchange_rate_lcu_usd"] > 0))
    depreciation = log_fx.groupby(p["iso3"]).diff()
    depreciation = depreciation.clip(*depreciation.quantile(list(WINSOR)))
    current = pd.DataFrame({
        "vulnerability_l1": p["nd_gain_vulnerability"],
        "reg_quality_l1": rq,
        "log_gdppc_l1": np.log(p["gdp_per_capita"].where(p["gdp_per_capita"] > 0)),
        "real_interest_l1": p["real_interest_rate"],
        "depreciation_l1": depreciation,
    })
    for col in CONTROLS:
        p[col] = current[col].groupby(p["iso3"]).shift(1)
    return p


def build_sample(panel: pd.DataFrame) -> pd.DataFrame:
    panel = add_lagged_controls(panel)
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
    d = data.dropna(subset=["event", *X, *(extra or [])]).copy()
    for f in fe:
        d = d[d.groupby(f)["event"].transform("max") > 0]
    dummies = pd.get_dummies(d[fe].astype(str), drop_first=True, dtype=float)
    cols = [d[X]]
    if extra:
        cols.append(d[extra])
    exog = sm.add_constant(pd.concat(cols + [dummies], axis=1))
    return d, exog


def logit_ame(data: pd.DataFrame, fe: list[str], duration: bool = False,
              controls: list[str] | None = None):
    """Logit with FE dummies, SEs clustered by country; average marginal effects."""
    d, exog = fe_design(data, fe + (["dur_bin"] if duration else []), controls)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.Logit(d["event"], exog).fit(
            disp=0, maxiter=200, cov_type="cluster",
            cov_kwds={"groups": pd.factorize(d["iso3"])[0]})
    ame = fit.get_margeff(at="overall").summary_frame()
    ame.index = [rename_dur(i) for i in ame.index]
    keep = X + (DUR_TERMS if duration else []) + (controls or [])
    ame = ame.loc[keep].rename(columns={"dy/dx": "Estimate", "Std. Err.": "Std. Error",
                                        "Pr(>|z|)": "p", "Conf. Int. Low": "2.5%",
                                        "Cont. Int. Hi.": "97.5%"})
    return ame[["Estimate", "Std. Error", "p", "2.5%", "97.5%"]], len(d), int(d["event"].sum())


def rename_dur(name: str) -> str:
    """dur_bin dummy names from get_dummies ("dur_bin_3_5") -> "dur_3_5"."""
    return name.replace("dur_bin_", "dur_")


def cloglog_fit(data: pd.DataFrame, fe: list[str], duration: bool = False,
                controls: list[str] | None = None):
    """Binomial GLM with a complementary log-log link (the discrete-time proportional-hazards
    model), FE dummies, SEs clustered by country. Coefficients are log hazard ratios."""
    d, exog = fe_design(data, fe + (["dur_bin"] if duration else []), controls)
    exog.columns = [rename_dur(c) for c in exog.columns]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.GLM(d["event"], exog,
                     family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(d["iso3"])[0]})
    keep = X + (DUR_TERMS if duration else []) + (controls or [])
    ci = fit.conf_int().loc[keep]
    out = pd.DataFrame({"Estimate": fit.params[keep], "Std. Error": fit.bse[keep],
                        "p": fit.pvalues[keep], "2.5%": ci[0], "97.5%": ci[1]})
    out["hazard ratio"] = np.exp(out["Estimate"])
    return out, len(d), int(d["event"].sum())


def cox_fit(data: pd.DataFrame, controls: list[str] | None = None):
    """Cox proportional hazards with Efron ties on the multiple-record (counting-process)
    data: each country-year is a (start, stop] interval in years since entry, with delayed
    entry for countries already in the market before 1993. Covariates X + region and income
    dummies. SEs are clustered by country with a cluster bootstrap (resampling countries):
    statsmodels' robust covariance uses score residuals that ignore delayed entry (they
    don't sum to zero here), so it can't be used."""
    regs = X + (controls or [])
    d = data.dropna(subset=["event", *regs]).copy()
    d["start"] = d["years_since_entry"]
    d["stop"] = d["start"] + 1
    dummies = pd.get_dummies(d[["region", "income_group"]].astype(str), drop_first=True,
                             dtype=float)
    exog = pd.concat([d[regs], dummies], axis=1)

    def fit_params(rows: pd.Index) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sm.PHReg(d.loc[rows, "stop"], exog.loc[rows], status=d.loc[rows, "event"],
                            entry=d.loc[rows, "start"], ties="efron").fit().params

    params = pd.Series(fit_params(d.index), index=exog.columns)
    rng = np.random.default_rng(COX_SEED)
    rows_by_country = {c: idx for c, idx in d.groupby("iso3").groups.items()}
    countries = np.array(list(rows_by_country))
    draws = []
    for _ in range(COX_BOOT):
        sample = rng.choice(countries, size=len(countries), replace=True)
        rows = np.concatenate([rows_by_country[c] for c in sample])
        boot = d.loc[rows].reset_index(drop=True)
        if boot["event"].sum() == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                b = sm.PHReg(boot["stop"], exog.loc[rows].reset_index(drop=True),
                             status=boot["event"], entry=boot["start"],
                             ties="efron").fit().params
            except (np.linalg.LinAlgError, ValueError):
                continue
        if np.all(np.isfinite(b)):
            draws.append(b)
    draws = pd.DataFrame(draws, columns=exog.columns)
    bse = draws[regs].std()
    out = pd.DataFrame({"Estimate": params[regs], "Std. Error": bse,
                        "p": 2 * stats.norm.sf(np.abs(params[regs] / bse)),
                        "2.5%": draws[regs].quantile(0.025),
                        "97.5%": draws[regs].quantile(0.975)})
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

    # --- Controls (all at t-1). Each set gets a same-sample baseline without controls, so a
    # change in the deal-flow coefficient isn't confused with the change in sample.
    base_fe = "year + region + income_group"
    control_sets = [("all controls", CONTROLS, "A"),
                    ("controls excl. real interest rate", CONTROLS_NO_RIR, "B")]
    control_order = []
    for name, ctrl, tag in control_sets:
        sample = risk.dropna(subset=X + ctrl)
        specs = [
            (f"({tag}1) LPM, (6) on {name} sample, no controls", sample, [dur], base_fe),
            (f"({tag}2) LPM, (6) + {name}", sample, [dur, *ctrl], base_fe),
            (f"({tag}3) LPM, (7) + {name}", sample, [dur, *ctrl], "region^year"),
        ]
        for label, data, extra, fe in specs:
            rhs = " + ".join(X + extra)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = pf.feols(f"event ~ {rhs} | {fe}", data=data, vcov={"CRV1": "iso3"})
            t = fit.tidy().rename(columns={"Pr(>|t|)": "p"})
            results[label] = (t[["Estimate", "Std. Error", "p", "2.5%", "97.5%"]], fit._N,
                              int(fit._data["event"].sum()),
                              f"R2 {fit._r2:.3f}, within R2 {fit._r2_within:.3f}")
            control_order.append(label)
        for label, func, kwargs in [
            (f"({tag}4) Logit AME, (8) + {name}", logit_ame,
             {"fe": ["year", "region", "income_group"], "duration": True, "controls": ctrl}),
            (f"({tag}5) Cloglog, (10) + {name}", cloglog_fit,
             {"fe": ["year", "region", "income_group"], "duration": True, "controls": ctrl}),
            (f"({tag}6) Cox, (11) + {name}", cox_fit, {"controls": ctrl}),
        ]:
            est, n, events = func(risk, **kwargs)
            results[label] = (est, n, events, "")
            control_order.append(label)

    def side_by_side(labels: list[str], rows: list[str]) -> str:
        table = pd.DataFrame(index=rows + ["N", "events"])
        for label in labels:
            est, n, events, _ = results[label]
            table[label] = [f"{est.at[r, 'Estimate']:.4f}{stars(est.at[r, 'p'])} "
                            f"({est.at[r, 'Std. Error']:.4f})" if r in est.index else ""
                            for r in rows] + [f"{n:,}", f"{events}"]
        with pd.option_context("display.width", 400, "display.max_columns", 30,
                               "display.max_colwidth", 30):
            return table.T.to_string()

    lines += ["", "Estimates, SE in parentheses. LPM: coefficients; logit: average marginal "
              "effects;", "cloglog and Cox: coefficients = log hazard ratios (hazard ratios in "
              "the full output).", "Duration dummies: years since entry, reference 0-2. "
              "* p<0.1, ** p<0.05, *** p<0.01.", "Logit and cloglog drop FE cells with no "
              "event; Cox uses time since entry as the baseline, so no year FE; its SEs and "
              f"CI come from a {COX_BOOT}-replication country-cluster bootstrap.", "",
              "TABLE 1 — main specifications", side_by_side(order, X + DUR_TERMS), "",
              "TABLE 2 — with country-year controls, all lagged one year (t-1):",
              "  vulnerability_l1 = ND-GAIN vulnerability; reg_quality_l1 = WGI regulatory "
              "quality (1997/1999/2001 interpolated);",
              "  log_gdppc_l1 = log GDP per capita; real_interest_l1 = real interest rate (%); "
              "depreciation_l1 = change in log LCU per USD",
              f"  (winsorised at {WINSOR[0]:.0%}/{WINSOR[1]:.0%}). (A1)/(B1) re-run (6) on the "
              "controls sample without controls, for comparison.",
              "Missing in the main estimation sample: "
              + ", ".join(f"{c} {int(complete[c].isna().sum())}" for c in CONTROLS), "",
              side_by_side(control_order, X + CONTROLS)]
    lines += ["", "Full output:"]
    for label in order + control_order:
        est, n, events, note = results[label]
        lines += ["", f"{label}  —  N {n:,}, events {events}" + (f", {note}" if note else ""),
                  est.round(4).to_string()]

    report = "\n".join(lines)
    print(report)
    HAZARD_TXT.write_text(report + "\n")
    print(f"\nSaved to {HAZARD_TXT}")


if __name__ == "__main__":
    main()
