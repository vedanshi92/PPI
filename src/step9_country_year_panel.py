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

import re
import sys

import pandas as pd

from common import name_to_iso3
from config import (CONFIG, COUNTRY_YEAR_PANEL, ENERGY_DEALS, FIT_POLICIES, ND_GAIN_PANEL,
                    PROJECT_PANEL, WDI_PANEL, WGI_PANEL)

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
    """n_ren_nd: renewable deals without derisking. ren_derisk: 1 if any renewable
    deal that year was derisked."""
    panel["n_ren_nd"] = panel["n_ren"] - panel["n_derisk_renewable"]
    panel["ren_derisk"] = (panel["n_derisk_renewable"] > 0).astype(int)
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


YEAR = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")


def year_in(value):
    """First four-digit year in a cell ("2005-2006" -> 2005, "Unknown Year" -> None)."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    match = YEAR.search(str(value))
    return int(match.group(1)) if match else None


def parse_removal(value):
    """RT5 "Year Removed" -> (removed, reintroduced, status). status is "none", "dated",
    "unknown_removal" (removed, year unknown) or "unknown_reintro" (reintroduced, year
    unknown), e.g. "2015 (reintroduced in 2020)" -> (2015, 2020, "dated")."""
    if pd.isna(value):
        return None, None, "none"
    removed = year_in(value)
    if removed is None:
        return None, None, "unknown_removal"
    reintro = re.search(r"(?:reintro|reinstat)\w*\D*(\d+)?", str(value), re.I)
    if reintro:
        year = reintro.group(1)
        if year and len(year) == 4:
            return removed, int(year), "dated"
        return removed, None, "unknown_reintro"
    return removed, None, "dated"


def load_rt5_fit(fallback: pd.DataFrame) -> pd.DataFrame:
    """Feed-in tariff dates per country from REN21 GSR 2023 table RT5: year adopted, year
    removed, year reintroduced. Only national rows are kept (state/city rows don't match a
    country name exactly). Where RT5 has no usable adoption year, the GSR 2025 Figure 9 year
    (`fallback`) is used. Writes the parsed table to data/interim for checking."""
    rt5 = pd.read_excel(CONFIG["ren21_rt5_file"], sheet_name=CONFIG["ren21_rt5_sheet"],
                        header=3).iloc[:, [1, 2, 3, 4]]
    rt5.columns = ["region", "country", "adopted_raw", "removed_raw"]
    rt5 = rt5[rt5["region"].notna() & rt5["country"].notna()].copy()
    rt5["country"] = (rt5["country"].astype(str).str.strip()
                      .str.replace(r"[\d,]+$", "", regex=True).str.strip())
    rt5["iso3"] = rt5["country"].apply(lambda n: name_to_iso3(n, fuzzy=False))
    skipped = sorted(rt5.loc[rt5["iso3"].isna(), "country"].unique())
    rt5 = rt5.dropna(subset=["iso3"]).drop_duplicates("iso3")

    rt5["adopted"] = rt5["adopted_raw"].apply(year_in)
    rt5[["removed", "reintroduced", "status"]] = pd.DataFrame(
        rt5["removed_raw"].apply(parse_removal).tolist(), index=rt5.index)
    rt5["source"] = "RT5"
    fit = rt5.set_index("iso3")[["country", "adopted_raw", "removed_raw", "adopted", "removed",
                                 "reintroduced", "status", "source"]]

    fb = fallback[fallback["policy"] == "fit"].set_index("iso3")["enacted"]
    use_fb = [iso for iso in fb.index if iso not in fit.index or pd.isna(fit.at[iso, "adopted"])]
    for iso in use_fb:
        if iso not in fit.index:
            fit.loc[iso, ["country", "status"]] = [iso, "none"]
        fit.loc[iso, ["adopted", "source"]] = [fb[iso], "GSR 2025 Figure 9"]
    fit = fit[fit["adopted"].notna() | (fit["status"] != "none")]
    fit.reset_index().to_csv(FIT_POLICIES, index=False)
    print(f"[PANEL] RT5 FiT: {(fit['source'] == 'RT5').sum()} countries; adoption year from "
          f"GSR 2025 Figure 9 for {len(use_fb)}; {len(skipped)} sub-national/unmatched rows "
          f"skipped. Parsed dates: {FIT_POLICIES}")
    return fit


def fit_status(fit: pd.DataFrame, iso3: str, prev: int):
    """1 if the FiT was in force in year `prev` (= t-1), 0 if not, blank if unknown."""
    if iso3 not in fit.index or pd.isna(fit.at[iso3, "adopted"]) or prev < fit.at[iso3, "adopted"]:
        return 0
    row = fit.loc[iso3]
    if row["status"] == "unknown_removal":
        return pd.NA
    if pd.isna(row["removed"]) or prev < row["removed"]:
        return 1
    if pd.notna(row["reintroduced"]):
        return int(prev >= row["reintroduced"])
    return pd.NA if row["status"] == "unknown_reintro" else 0


def add_policies(panel: pd.DataFrame) -> pd.DataFrame:
    """Policy controls measured at t-1:
      auction_active       an auction-proxy deal by t-1 (auction_year <= t-1)
      fit_active           feed-in tariff in force in t-1 (REN21 RT5 adoption/removal years;
                           blank where it was removed or reintroduced in an unknown year)
      nm_active            net metering or net billing enacted by t-1 (GSR 2025 Figure 9)
      prior_energy_derisk  first_any_derisk before the current year"""
    prev = panel["year"] - 1
    panel["auction_active"] = (panel["auction_year"] <= prev).fillna(False).astype(int)
    ren = load_ren21()
    fit = load_rt5_fit(ren)
    panel["fit_active"] = pd.array(
        [fit_status(fit, iso, p) for iso, p in zip(panel["iso3"], prev)], dtype="Int64")
    nm_enacted = panel["iso3"].map(ren[ren["policy"] == "nm"].set_index("iso3")["enacted"])
    panel["nm_active"] = (nm_enacted <= prev).astype(int)
    panel["prior_energy_derisk"] = (panel["first_any_derisk"] < panel["year"]).fillna(False
                                                                                  ).astype(int)
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
