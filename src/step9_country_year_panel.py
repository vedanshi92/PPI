"""
step9_country_year_panel.py
---------------------------
Collapses the project panel to a BALANCED country-year panel: every country that has at
least one PPI energy project, for every year from the first to the last financial-closure
year in the data. Country-years with no project get zero counts, zero investment and zero
capacity. A project whose investment isn't reported counts as 0 in the investment totals
(n_investment_missing counts these). ND-GAIN, WGI and WDI are merged for every country-year.

Writes data/processed/country_year_panel.csv.

Run from the project root:  python3 src/step9_country_year_panel.py
"""

import sys

import pandas as pd

from config import CONFIG, COUNTRY_YEAR_PANEL, ND_GAIN_PANEL, PROJECT_PANEL, WDI_PANEL, WGI_PANEL

# Project flags summed into counts, for all projects and for renewable projects only
FLAGS = ["derisk", "public_finance", "govt_derisk", "govt_finance", "any_support"]


def collapse(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    renewable = p["technology_category"] == "renewable"
    p["investment_filled"] = p["investment_musd"].fillna(0)
    p["capacity_mw_filled"] = p["capacity_mw"].fillna(0)
    p["foreign_sponsor"] = (p["has_foreign_sponsor"].astype(str) == "True").astype(int)

    cols = {
        "n_projects": p["project_id"].notna().astype(int),
        "n_renewable": renewable.astype(int),
        "n_non_renewable": (p["technology_category"] == "non_renewable").astype(int),
        "n_mixed": (p["technology_category"] == "mixed").astype(int),
        "investment_musd": p["investment_filled"],
        "investment_renewable_musd": p["investment_filled"].where(renewable, 0),
        "investment_non_renewable_musd": p["investment_filled"].where(
            p["technology_category"] == "non_renewable", 0),
        "n_investment_missing": p["investment_musd"].isna().astype(int),
        "capacity_mw": p["capacity_mw_filled"],
        "capacity_mw_renewable": p["capacity_mw_filled"].where(renewable, 0),
        "n_foreign_sponsor": p["foreign_sponsor"],
        "n_auction_deal": (p["auction_deal"] == 1).astype(int),
    }
    for flag in FLAGS:
        is_one = (p[flag] == 1).astype(int)
        cols[f"n_{flag}"] = is_one
        cols[f"n_{flag}_renewable"] = is_one.where(renewable, 0)

    sums = pd.DataFrame(cols).assign(iso3=p["iso3"], year=p["year"])
    return sums.groupby(["iso3", "year"], as_index=False).sum()


def main():
    print("=" * 70)
    print("Step 9: balanced country-year panel")
    print("=" * 70)
    if not PROJECT_PANEL.exists():
        sys.exit(f"{PROJECT_PANEL} not found — run step5_merge_panel.py first.")
    projects = pd.read_csv(PROJECT_PANEL, low_memory=False)

    countries = (projects.groupby("iso3")["country_name"].first().reset_index())
    years = range(int(projects["year"].min()), int(projects["year"].max()) + 1)
    grid = countries.merge(pd.DataFrame({"year": list(years)}), how="cross")

    panel = grid.merge(collapse(projects), on=["iso3", "year"], how="left")
    count_cols = [c for c in panel.columns if c not in ("iso3", "country_name", "year")]
    panel[count_cols] = panel[count_cols].fillna(0)
    int_cols = [c for c in count_cols if c.startswith("n_")]
    panel[int_cols] = panel[int_cols].astype(int)
    panel.insert(3, "has_project", (panel["n_projects"] > 0).astype(int))
    panel["renewable_investment_share"] = (
        panel["investment_renewable_musd"] / panel["investment_musd"]).where(
        panel["investment_musd"] > 0)

    # Country-year covariates for every cell, not only those with projects
    sources = [pd.read_csv(ND_GAIN_PANEL), pd.read_csv(WGI_PANEL)]
    if WDI_PANEL.exists():
        sources.append(pd.read_csv(WDI_PANEL))
    for source in sources:
        panel = panel.merge(source, on=["iso3", "year"], how="left", validate="1:1")

    assert len(panel) == len(countries) * len(years), "panel is not balanced"
    assert panel["n_projects"].sum() == len(projects), "projects lost in the collapse"
    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
    panel.to_csv(COUNTRY_YEAR_PANEL, index=False)

    covariates = ["nd_gain_value", *CONFIG["wgi_sheet_indicators"].values(),
                  *CONFIG["wdi_indicators"].values()]
    covariates = [c for c in covariates if c in panel.columns]
    print(f"[PANEL] {len(countries):,} countries x {len(years)} years ({years[0]}-{years[-1]}) "
          f"= {len(panel):,} country-years; {panel['has_project'].sum():,} with a project.")
    print(f"[PANEL] Investment: {panel['investment_musd'].sum():,.1f} USD m in total; "
          f"{panel['n_investment_missing'].sum():,} projects with unreported investment "
          f"counted as 0.")
    print("[PANEL] Missing covariates (% of country-years):")
    print((100 * panel[covariates].isna().mean()).round(1).to_string())
    print(f"[PANEL] Wrote {len(panel):,} rows, {len(panel.columns)} columns, to {COUNTRY_YEAR_PANEL}")


if __name__ == "__main__":
    main()
