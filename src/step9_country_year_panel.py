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

from common import name_to_iso3
from config import (CONFIG, COUNTRY_YEAR_PANEL, ENERGY_DEALS, ND_GAIN_PANEL, PROJECT_PANEL,
                    WDI_PANEL, WGI_PANEL)

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
        "n_ren": renewable.astype(int),
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


def first_year(panel: pd.DataFrame, mask: pd.Series) -> pd.Series:
    """For each row, the first year in which `mask` holds for that country (blank if never)."""
    first = panel.loc[mask].groupby("iso3")["year"].min()
    return panel["iso3"].map(first).astype("Int64")


def add_renewable_deal_vars(panel: pd.DataFrame) -> pd.DataFrame:
    """n_ren_nd: renewable deals without derisking. first_ren_derisk: 1 if any renewable
    deal that year was derisked."""
    panel["n_ren_nd"] = panel["n_ren"] - panel["n_derisk_renewable"]
    panel["first_ren_derisk"] = (panel["n_derisk_renewable"] > 0).astype(int)
    return panel


def add_country_dates(panel: pd.DataFrame) -> pd.DataFrame:
    """Fixed country dates (the same in every year of a country):
      treat_year       first year with a derisked renewable deal
      entry_year       first year with any energy PPI deal
      first_any_derisk first year with a derisked deal in any energy technology
      auction_year     first year with an auction-proxy deal (auction_deal = 1)
    entry_year and first_any_derisk use ALL energy deals from 1990, including those step 1
    drops for having no classifiable technology (e.g. transmission, distribution, gas)."""
    panel["treat_year"] = first_year(panel, panel["n_derisk_renewable"] > 0)
    deals = pd.read_csv(ENERGY_DEALS)
    deals = deals[deals["year"] >= panel["year"].min()]
    panel["entry_year"] = panel["iso3"].map(deals.groupby("iso3")["year"].min()).astype("Int64")
    panel["first_any_derisk"] = panel["iso3"].map(
        deals[deals["derisk"] == 1].groupby("iso3")["year"].min()).astype("Int64")
    panel["auction_year"] = first_year(panel, panel["n_auction_deal"] > 0)
    return panel


def load_ren21() -> pd.DataFrame:
    """REN21 policies as one row per country and policy (fit / nm), earliest year enacted."""
    ren = pd.read_excel(CONFIG["ren21_file"], sheet_name=CONFIG["ren21_sheet"], header=2)
    ren = ren.dropna(subset=["Country", "Policy Type"])
    ren["iso3"] = ren["Country"].apply(name_to_iso3)
    unmatched = ren.loc[ren["iso3"].isna(), "Country"].unique()
    if len(unmatched):
        print(f"[PANEL] WARNING: REN21 countries with no ISO3 (left out): {list(unmatched)}")
    ren["policy"] = ren["Policy Type"].str.strip().str.lower().map(CONFIG["ren21_policy_map"])
    ren["enacted"] = pd.to_numeric(ren["Year Enacted"], errors="coerce")
    return (ren.dropna(subset=["iso3", "policy", "enacted"])
            .groupby(["iso3", "policy"], as_index=False)["enacted"].min())


def add_policies(panel: pd.DataFrame) -> pd.DataFrame:
    """Policy controls measured at t-1:
      auction_active       an auction-proxy deal by t-1 (auction_year <= t-1)
      fit_active, nm_active REN21 feed-in tariff / net metering enacted by t-1 and not removed
      prior_energy_derisk  first_any_derisk before the current year"""
    prev = panel["year"] - 1
    panel["auction_active"] = (panel["auction_year"] <= prev).fillna(False).astype(int)
    ren = load_ren21()
    for policy in ["fit", "nm"]:
        enacted = panel["iso3"].map(ren[ren["policy"] == policy].set_index("iso3")["enacted"])
        removed = pd.Series([CONFIG["ren21_removed"].get((iso, policy)) for iso in panel["iso3"]],
                            index=panel.index, dtype="float")
        panel[f"{policy}_active"] = ((enacted <= prev) & ~(removed <= prev)).astype(int)
    panel["prior_energy_derisk"] = (panel["first_any_derisk"] < panel["year"]).fillna(False
                                                                                  ).astype(int)
    print(f"[PANEL] REN21: {ren['iso3'].nunique()} countries with a FiT or net metering policy, "
          f"{ren['iso3'].isin(panel['iso3']).groupby(ren['iso3']).first().sum()} of them in "
          f"the panel.")
    return panel


def add_deal_history(panel: pd.DataFrame) -> pd.DataFrame:
    """Renewable deal flow in the previous three years (t-1..t-3) and the three before that
    (t-4..t-6). Blank where the window reaches back before the panel's first year."""
    by_country = panel.groupby("iso3")["n_ren"]
    lags = {k: by_country.shift(k) for k in range(1, 7)}
    panel["deals_l1_3"] = lags[1] + lags[2] + lags[3]
    panel["deals_l4_6"] = lags[4] + lags[5] + lags[6]
    panel["deals_growth"] = panel["deals_l1_3"] - panel["deals_l4_6"]
    for col in ["deals_l1_3", "deals_l4_6", "deals_growth"]:
        panel[col] = panel[col].astype("Int64")
    return panel


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
        wdi = pd.read_csv(WDI_PANEL)
        wdi_cols = [c for c in CONFIG["wdi_indicators"].values() if c in wdi.columns]
        sources.append(wdi[["iso3", "year", *wdi_cols]])
    for source in sources:
        panel = panel.merge(source, on=["iso3", "year"], how="left", validate="1:1")

    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
    panel = add_renewable_deal_vars(panel)
    panel = add_country_dates(panel)
    panel = add_policies(panel)
    panel = add_deal_history(panel)

    assert len(panel) == len(countries) * len(years), "panel is not balanced"
    assert panel["n_projects"].sum() == len(projects), "projects lost in the collapse"
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
