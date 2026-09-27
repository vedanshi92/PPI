"""
build_project_panel.py
-----------------------
Builds a PROJECT-level panel (one row per PPI project, not aggregated to
country-year). For each project, this:
  1. Loads the raw PPI export and keeps every original column
  2. Classifies each project's `Technology` as renewable / non_renewable / mixed,
     dropping projects whose Technology is missing or unclassifiable
  3. Generates a domestic/international sponsor indicator from `Sponsors Country`,
     comparing each sponsor's country against the project's own country
  4. Merges in ND-GAIN, WGI, and live WDI macro data — a many-to-one merge, since
     many projects can share the same country-year
  5. Prints descriptive statistics
  6. Saves the result to data/processed/project_panel.csv

Country-year aggregation happens in a SEPARATE later script that reads the
output of this one — this script deliberately stops at the project level.

Run from the project root:
    python3 src/build_project_panel.py
"""

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import name_to_iso3, report_merge, load_nd_gain, load_wgi, load_wdi_via_api

# --------------------------------------------------------------------------------------
# CONFIG — edit this section to match your actual downloaded files
# --------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

CONFIG = {
    # --- PPI database ---
    "ppi_file": RAW / "ppi.xlsx",
    "ppi_sheet": "CustomQuery",
    "ppi_country_col": "Country",
    "ppi_year_col": "Financial closure year",
    "ppi_investment_col": "TotalInvestment",
    "ppi_technology_col": "Technology",
    "ppi_sponsor_country_col": "Sponsors Country",

    # Restrict to a single PPI sector. Set to None to keep all sectors.
    "ppi_sector_col": "Primary sector",
    "ppi_sector_value": "Energy",

    # Keywords used to classify the Technology field. Lowercased, substring match.
    # NOTE: "Waste" (waste-to-energy) is a genuine judgment call in the literature —
    # it's counted as non-renewable here. Move "waste" to renewable_keywords if you
    # want to count it as renewable instead.
    # NOTE: use "natural gas", not plain "gas" — "gas" would also match "biogas".
    "renewable_keywords": ["solar", "wind", "hydro", "geothermal", "biomass", "biogas"],
    "nonrenewable_keywords": ["coal", "diesel", "natural gas", "nuclear", "steam", "waste"],

    # --- World Governance Indicators (WGI) ---
    # This expects the official multi-sheet WGI Excel export (one sheet per governance
    # dimension: va, pv, ge, rq, rl, cc), already in long format — NOT a wide CSV.
    "wgi_file": RAW / "wgidata.xlsx",
    "wgi_sheet_indicators": {
        "va": "voice_accountability",
        "pv": "political_stability",
        "ge": "govt_effectiveness",
        "rq": "regulatory_quality",
        "rl": "rule_of_law",
        "cc": "control_of_corruption",
    },

    # --- ND-GAIN ---
    "nd_gain_file": RAW / "nd_gain.csv",
    "nd_gain_country_col": "ISO3",
    "nd_gain_wide_id_cols": ["ISO3", "Name"],

    # --- WDI macro indicators, pulled live from the World Bank API ---
    "wdi_indicators": {
        "NY.GDP.PCAP.KD": "gdp_per_capita",
        "NY.GDP.MKTP.KD.ZG": "gdp_growth",
        "SP.POP.TOTL": "population",
        "FP.CPI.TOTL.ZG": "inflation_cpi",
        "PA.NUS.FCRF": "exchange_rate_lcu_usd",
        "NE.TRD.GNFS.ZS": "trade_pct_gdp",
        "BX.KLT.DINV.WD.GD.ZS": "fdi_net_inflows_pct_gdp",
        "FR.INR.RINR": "real_interest_rate",
        "GC.DOD.TOTL.GD.ZS": "govt_debt_pct_gdp",
        "EG.ELC.ACCS.ZS": "electricity_access_pct",
    },

    "year_min": 1990,
    "year_max": 2025,
}


# --------------------------------------------------------------------------------------
# PPI loading + indicator construction
# --------------------------------------------------------------------------------------

def load_ppi_projects() -> pd.DataFrame:
    path = CONFIG["ppi_file"]
    if not path.exists():
        sys.exit(f"[PPI] File not found: {path}\n"
                  f"Download from https://ppi.worldbankgroup.org/en/ppi and place it there.")

    df = pd.read_excel(path, sheet_name=CONFIG["ppi_sheet"])
    print(f"[PPI] Loaded {len(df):,} raw rows, {len(df.columns)} columns.")

    required = [CONFIG["ppi_country_col"], CONFIG["ppi_year_col"], CONFIG["ppi_investment_col"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"[PPI] Missing expected column(s): {missing}\n"
                  f"Columns found: {list(df.columns)}\nUpdate CONFIG.")

    df = df.rename(columns={
        CONFIG["ppi_country_col"]: "country_name",
        CONFIG["ppi_year_col"]: "year",
        CONFIG["ppi_investment_col"]: "investment_musd",
    })

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    # Investment is kept even when unparseable (e.g. "Not Available") — this is a
    # project-level panel, not a sum, so we don't want to silently drop whole projects
    # just because their investment figure is missing.
    df["investment_musd"] = pd.to_numeric(df["investment_musd"], errors="coerce")
    df = df.dropna(subset=["year", "country_name"])
    df["year"] = df["year"].astype(int)

    # Drop known placeholder/test records regardless of year filtering — e.g. the
    # World Bank PPI export contains at least one literal "**TEST**" project name
    # (Mexico, 1988) that isn't a real project.
    if "Project name" in df.columns:
        before = len(df)
        df = df[~df["Project name"].astype(str).str.strip().str.lower().isin(["**test**", "test"])]
        if len(df) < before:
            print(f"[PPI] Dropped {before - len(df)} placeholder/test record(s) "
                  f"(e.g. a project literally named 'TEST').")

    # --- Sector filter ---
    sector_col = CONFIG.get("ppi_sector_col")
    sector_value = CONFIG.get("ppi_sector_value")
    if sector_col and sector_value:
        if sector_col not in df.columns:
            sys.exit(f"[PPI] Sector column '{sector_col}' not found.\n"
                      f"Columns found: {list(df.columns)}")
        before = len(df)
        df = df[df[sector_col].astype(str).str.strip().str.lower() == sector_value.strip().lower()].copy()
        print(f"[PPI] Sector filter '{sector_value}': kept {len(df):,} of {before:,} rows.")

    # --- Country ISO3 (needed both for the sponsor comparison below and the merges later) ---
    df["iso3"] = df["country_name"].apply(name_to_iso3)
    n_unmatched = df["iso3"].isna().sum()
    if n_unmatched:
        print(f"[PPI] WARNING: {n_unmatched:,} rows have a country name that didn't "
              f"match an ISO3 code — check common.MANUAL_ISO3_FIXES.")

    df = add_renewable_indicator(df)
    df = add_sponsor_indicator(df)

    return df


def add_renewable_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """The Technology field is genuinely ambiguous to parse: a comma can separate two
    distinct technologies ("Coal, Hydro, Large (>50MW)") or attach a qualifier to a
    single one ("Hydro, Small (<50MW)", "Solar, PV") — there's no reliable way to tell
    these apart from the raw text alone. So rather than splitting into a list (which
    would wrongly double-count "Hydro, Small (<50MW)" as two technologies), we check
    for renewable keywords as a substring across the WHOLE field. This correctly
    handles both single- and multi-technology entries without needing to parse them
    apart.

    Each project gets a `technology_category`:
      - "renewable"      — only renewable keywords (solar, wind, hydro, ...)
      - "non_renewable"  — only non-renewable keywords (coal, diesel, natural gas, ...)
      - "mixed"          — both, e.g. "Diesel, Hydro, Large (>50MW)"
    Qualifiers like "Not Applicable", "N/A", "Other" are ignored when classifying,
    since they usually mean "subtype unspecified" for an already-identified technology.

    Rows with no identifiable technology — a NaN/blank field, or one containing only
    placeholders like "Not Applicable", "N/A", "N/A, N/A", "Other" — are DROPPED, since
    they can't be placed in any of the three categories."""
    tech_col = CONFIG["ppi_technology_col"]
    renewable_kw = CONFIG["renewable_keywords"]
    nonrenewable_kw = CONFIG["nonrenewable_keywords"]

    if tech_col not in df.columns:
        sys.exit(f"[PPI] Technology column '{tech_col}' not found — can't classify "
                  f"projects as renewable/non-renewable/mixed.\n"
                  f"Columns found: {list(df.columns)}\nUpdate CONFIG.")

    tech_lower = df[tech_col].fillna("").astype(str).str.lower()

    df["n_renewable_keywords_matched"] = tech_lower.apply(
        lambda s: sum(kw in s for kw in renewable_kw))
    has_renewable = df["n_renewable_keywords_matched"] > 0
    has_nonrenewable = tech_lower.apply(lambda s: any(kw in s for kw in nonrenewable_kw))

    df["technology_category"] = np.select(
        [has_renewable & has_nonrenewable, has_renewable, has_nonrenewable],
        ["mixed", "renewable", "non_renewable"],
        default="",
    )

    unclassified = df["technology_category"] == ""
    if unclassified.any():
        print(f"[PPI] Dropped {unclassified.sum():,} project(s) with a missing or "
              f"unclassifiable Technology field. Values dropped:")
        print(df.loc[unclassified, tech_col].value_counts(dropna=False).to_string())
    df = df[~unclassified].copy()

    # Convenience booleans derived from the category
    df["is_renewable"] = df["technology_category"].isin(["renewable", "mixed"])
    df["is_pure_renewable"] = df["technology_category"] == "renewable"

    return df


def add_sponsor_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """Sponsors Country is a list of country names. IMPORTANT: entries are separated
    by a literal double line-break ("\\n\\n"), NOT by comma — a plain comma can be
    part of a single country's own name (e.g. "Hong Kong, China"). Splitting on
    comma alone would wrongly break that into two countries, so we split on the
    double line-break instead."""
    sponsor_col = CONFIG["ppi_sponsor_country_col"]

    if sponsor_col not in df.columns:
        warnings.warn(f"[PPI] Sponsor country column '{sponsor_col}' not found — "
                       f"sponsor indicators will be blank for all rows.")
        df["sponsor_countries"] = ""
        df["sponsor_type"] = "unknown"
        df["has_domestic_sponsor"] = False
        df["has_international_sponsor"] = False
        return df

    def parse_sponsor_countries(raw):
        if not isinstance(raw, str) or not raw.strip():
            return []
        parts = re.split(r"\n\n", raw)
        parts = [p.strip(" ,\n") for p in parts]
        # ".." is the World Bank PPI export's placeholder for an unknown sponsor country
        return [p for p in parts if p and p != ".."]

    def classify(raw, project_iso3):
        sponsor_names = parse_sponsor_countries(raw)
        if not sponsor_names:
            return "", "unknown", False, False

        sponsor_iso3 = [name_to_iso3(name) for name in sponsor_names]
        sponsor_iso3 = [code for code in sponsor_iso3 if code is not None]

        if not sponsor_iso3:
            # We had sponsor country text, but couldn't map any of it to an ISO3 —
            # treat as unknown rather than guessing.
            return "; ".join(sponsor_names), "unknown", False, False

        is_domestic = any(code == project_iso3 for code in sponsor_iso3)
        is_international = any(code != project_iso3 for code in sponsor_iso3)

        if is_domestic and is_international:
            sponsor_type = "mixed"
        elif is_domestic:
            sponsor_type = "domestic_only"
        else:
            sponsor_type = "international_only"

        return "; ".join(sponsor_names), sponsor_type, is_domestic, is_international

    results = df.apply(
        lambda row: classify(row[sponsor_col], row["iso3"]), axis=1, result_type="expand"
    )
    results.columns = ["sponsor_countries", "sponsor_type",
                        "has_domestic_sponsor", "has_international_sponsor"]
    df = pd.concat([df, results], axis=1)

    return df


# --------------------------------------------------------------------------------------
# Descriptive statistics
# --------------------------------------------------------------------------------------

def print_descriptive_stats(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("Descriptive statistics — project-level panel")
    print("=" * 70)

    print(f"Total projects: {len(df):,}")
    print(f"Countries: {df['iso3'].nunique():,}")
    print(f"Years: {df['year'].min()}-{df['year'].max()}")

    print("\nTechnology category:")
    print(df["technology_category"].value_counts().to_string())

    print("\nRenewable status:")
    print(f"  Renewable (any renewable tech listed): "
          f"{df['is_renewable'].sum():,} ({100 * df['is_renewable'].mean():.1f}%)")
    print(f"  Purely renewable (all listed tech renewable): "
          f"{df['is_pure_renewable'].sum():,} ({100 * df['is_pure_renewable'].mean():.1f}%)")

    print("\nSponsor type breakdown:")
    print(df["sponsor_type"].value_counts(dropna=False).to_string())

    if "investment_musd" in df.columns:
        print(f"\nTotal investment captured: ${df['investment_musd'].sum(skipna=True):,.0f}M "
              f"({df['investment_musd'].isna().sum():,} projects with missing/unparseable investment)")

    print("\nRenewable share of investment, by sponsor type:")
    inv_by_sponsor = df.groupby("sponsor_type").apply(
        lambda g: pd.Series({
            "n_projects": len(g),
            "renewable_pct": 100 * g["is_renewable"].mean(),
        })
    )
    print(inv_by_sponsor.to_string())


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Building project-level panel")
    print("=" * 70)

    ppi = load_ppi_projects()

    nd_gain_panel = load_nd_gain(
        CONFIG["nd_gain_file"],
        country_col=CONFIG["nd_gain_country_col"],
        wide_id_cols=CONFIG["nd_gain_wide_id_cols"],
    )
    wgi_panel = load_wgi(
        CONFIG["wgi_file"],
        sheet_indicators=CONFIG["wgi_sheet_indicators"],
    )
    wdi_panel = load_wdi_via_api(
        CONFIG["wdi_indicators"], CONFIG["year_min"], CONFIG["year_max"]
    )

    print("\nMerge diagnostics (many PPI project-rows -> one country-year row):")
    report_merge(ppi, nd_gain_panel, ["iso3", "year"], "PPI -> ND-GAIN")
    merged = ppi.merge(nd_gain_panel, on=["iso3", "year"], how="left", validate="m:1")

    report_merge(merged, wgi_panel, ["iso3", "year"], "+ WGI")
    merged = merged.merge(wgi_panel, on=["iso3", "year"], how="left", validate="m:1")

    report_merge(merged, wdi_panel, ["iso3", "year"], "+ WDI (live API)")
    merged = merged.merge(wdi_panel, on=["iso3", "year"], how="left", validate="m:1")

    merged = merged[
        (merged["year"] >= CONFIG["year_min"]) & (merged["year"] <= CONFIG["year_max"])
    ].sort_values(["iso3", "year"]).reset_index(drop=True)

    out_path = OUT / "project_panel.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nWrote {len(merged):,} project rows, {len(merged.columns)} columns, to {out_path}")

    print_descriptive_stats(merged)


if __name__ == "__main__":
    main()
