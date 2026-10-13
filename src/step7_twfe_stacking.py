"""
step7_twfe_stacking.py
----------------------
Policy stacking on renewables: are derisked projects also more likely to receive public
finance? Project-level linear probability model with country-year and technology fixed
effects, standard errors clustered by country:

    public_finance ~ derisk | iso3^year + tech      (renewable projects only)

Writes data/processed/twfe_policy_stacking.txt.

Run from the project root:  python3 src/step7_twfe_stacking.py
"""

import sys

import pandas as pd
import pyfixest as pf

from config import PROJECT_PANEL, STACKING_TXT


def main():
    print("=" * 70)
    print("Step 7: two-way fixed effects, policy stacking on renewables")
    print("=" * 70)
    if not PROJECT_PANEL.exists():
        sys.exit(f"{PROJECT_PANEL} not found — run step5_merge_panel.py first.")
    df = pd.read_csv(PROJECT_PANEL, low_memory=False)

    renewables = df[df["technology_category"] == "renewable"].copy()
    renewables["tech"] = renewables["tech"].astype(str)

    fit = pf.feols("public_finance ~ derisk | iso3^year + tech",
                   data=renewables, vcov={"CRV1": "iso3"})

    # Only country-years with more than one project, and variation in derisk within them,
    # identify the coefficient. Report how much of the sample that is.
    cy = renewables.groupby(["iso3", "year"])["derisk"]
    multi = cy.transform("size") > 1
    varies = cy.transform("nunique") > 1
    lines = [
        "Policy stacking on renewables: public_finance ~ derisk | iso3^year + tech",
        "Linear probability model, SEs clustered by country (CRV1, iso3).",
        "",
        f"Renewable projects: {len(renewables):,} | country-years: "
        f"{renewables.groupby(['iso3', 'year']).ngroups:,} | countries: "
        f"{renewables['iso3'].nunique():,}",
        f"Projects in country-years with >1 project: {multi.sum():,}",
        f"Projects in country-years where derisk varies (identifying sample): {varies.sum():,}",
        f"Mean public_finance: {renewables['public_finance'].mean():.3f} | "
        f"mean derisk: {renewables['derisk'].mean():.3f}",
        "",
        "Raw means of public_finance by derisk:",
        renewables.groupby("derisk")["public_finance"].agg(["size", "mean"]).round(3).to_string(),
        "",
        fit.tidy().round(4).to_string(),
        "",
        f"Observations used: {fit._N:,} | R2: {fit._r2:.3f} | within R2: {fit._r2_within:.3f}",
    ]
    report = "\n".join(lines)
    print(report)
    STACKING_TXT.write_text(report + "\n")
    print(f"\nSaved to {STACKING_TXT}")


if __name__ == "__main__":
    main()
