"""
step7_twfe_stacking.py
----------------------
Project-level fixed effects models on renewable projects (linear probability models,
standard errors clustered by country):

  1. Policy stacking:      public_finance ~ derisk
  2. Bilateral only:       public_finance_bilateral ~ derisk_bilateral
  3. MLS over BLS:         mls_support ~ bls_support
                           (any multilateral / bilateral derisk or public finance)
  4. Foreign sponsors:     derisk ~ foreign_sponsor        (sponsor country known)
  5. Award method:         derisk ~ award method dummies   (award method recorded;
                                                            reference: competitive bidding)

Each model is run with two fixed-effect sets:
  (a) iso3^year + tech                      — country-year FE; ND-GAIN and GDP growth are
                                              country-year variables, so this absorbs them
  (b) iso3 + year + tech + nd_gain_value + gdp_growth
                                            — country and year FE separately, so the
                                              controls are identified

Writes data/processed/twfe_policy_stacking.txt.

Run from the project root:  python3 src/step7_twfe_stacking.py
"""

import sys
import warnings

import pandas as pd
import pyfixest as pf

from config import PROJECT_PANEL, STACKING_TXT

CONTROLS = "nd_gain_value + gdp_growth"
AWARD_REF = "Competitive bidding"

# (name, outcome, regressors, sample filter)
MODELS = [
    ("1. Policy stacking", "public_finance", "derisk", None),
    ("2. Bilateral only", "public_finance_bilateral", "derisk_bilateral", None),
    ("3. MLS over BLS", "mls_support", "bls_support", None),
    ("4. Foreign sponsor", "derisk", "foreign_sponsor", "foreign_sponsor_known"),
    ("5. Award method", "derisk",
     f"i(award_method, ref='{AWARD_REF}')", "award_method_known"),
]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    r = df[df["technology_category"] == "renewable"].copy()
    r["tech"] = r["tech"].astype(str)
    r["mls_support"] = ((r["derisk_multilateral"] == 1)
                        | (r["public_finance_multilateral"] == 1)).astype(int)
    r["bls_support"] = ((r["derisk_bilateral"] == 1)
                        | (r["public_finance_bilateral"] == 1)).astype(int)
    r["foreign_sponsor"] = r["has_foreign_sponsor"].astype(str).map({"True": 1, "False": 0})
    r["foreign_sponsor_known"] = r["foreign_sponsor"].notna()
    r["award_method"] = r["AwardMethod"].astype(str).str.strip()
    r["award_method_known"] = ~r["award_method"].isin(["Not Available", "Not Applicable", "nan"])
    return r


def describe(r: pd.DataFrame, outcome: str, regressor: str) -> list[str]:
    """Raw means of the outcome by the regressor (for the binary regressors)."""
    if regressor not in r.columns:
        return [f"Raw mean of {outcome} by award method:",
                r.groupby("award_method")[outcome].agg(["size", "mean"]).round(3).to_string()]
    return [f"Raw mean of {outcome} by {regressor}:",
            r.groupby(regressor)[outcome].agg(["size", "mean"]).round(3).to_string()]


def main():
    print("=" * 70)
    print("Step 7: fixed effects models on renewable projects")
    print("=" * 70)
    if not PROJECT_PANEL.exists():
        sys.exit(f"{PROJECT_PANEL} not found — run step5_merge_panel.py first.")
    renewables = prepare(pd.read_csv(PROJECT_PANEL, low_memory=False))

    lines = [
        "Fixed effects models on renewable projects (linear probability models)",
        "SEs clustered by country (CRV1, iso3).",
        "(a) FE: iso3^year + tech. ND-GAIN and GDP growth vary only by country-year, so these",
        "    FE absorb them; they are left out of (a).",
        "(b) FE: iso3 + year + tech, controls: nd_gain_value + gdp_growth.",
        f"Renewable projects: {len(renewables):,}",
    ]

    for name, outcome, regressor, sample in MODELS:
        data = renewables[renewables[sample]] if sample else renewables
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # singleton-FE notices; counts are reported below
            fit_a = pf.feols(f"{outcome} ~ {regressor} | iso3^year + tech",
                             data=data, vcov={"CRV1": "iso3"})
            fit_b = pf.feols(f"{outcome} ~ {regressor} + {CONTROLS} | iso3 + year + tech",
                             data=data, vcov={"CRV1": "iso3"})

        lines += ["", "=" * 78, f"{name}: {outcome} ~ {regressor.split(',')[0].strip()}"
                  .replace("i(award_method", f"award method dummies (ref: {AWARD_REF})"),
                  "=" * 78, f"Sample: {len(data):,} projects", *describe(data, outcome, regressor)]
        for label, fit in [("(a) iso3^year + tech", fit_a),
                           ("(b) iso3 + year + tech + controls", fit_b)]:
            lines += ["", f"{label}  —  N used: {fit._N:,}, R2: {fit._r2:.3f}, "
                          f"within R2: {fit._r2_within:.3f}",
                      fit.tidy().round(4).to_string()]

    report = "\n".join(lines)
    print(report)
    STACKING_TXT.write_text(report + "\n")
    print(f"\nSaved to {STACKING_TXT}")


if __name__ == "__main__":
    main()
