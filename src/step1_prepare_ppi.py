"""
step1_prepare_ppi.py
--------------------
Loads the raw PPI export (one row per project) and:
  1. Drops rows with no year/country, placeholder "TEST" records, other sectors,
     and projects outside the configured year window
  2. Maps each project's country to an ISO3 code
  3. Classifies Technology as renewable / non_renewable / mixed, DROPPING projects
     whose Technology is missing or unclassifiable
  4. Parses Sponsors Country into domestic vs. foreign sponsors by ISO3 comparison
Writes data/interim/ppi_projects.csv. Every original PPI column is kept.

Run from the project root:  python3 src/step1_prepare_ppi.py
"""

import re
import sys

import numpy as np
import pandas as pd

from common import name_to_iso3
from config import CONFIG, PPI_CLEAN


def load_ppi_projects() -> pd.DataFrame:
    path = CONFIG["ppi_file"]
    if not path.exists():
        sys.exit(f"[PPI] File not found: {path}\n"
                  f"Download from https://ppi.worldbankgroup.org/en/ppi and place it there.")

    df = pd.read_excel(path, sheet_name=CONFIG["ppi_sheet"])
    print(f"[PPI] Loaded {len(df):,} raw rows, {len(df.columns)} columns.")

    required = [CONFIG["ppi_country_col"], CONFIG["ppi_year_col"], CONFIG["ppi_investment_col"],
                CONFIG["ppi_technology_col"], CONFIG["ppi_sponsor_country_col"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"[PPI] Missing expected column(s): {missing}\n"
                  f"Columns found: {list(df.columns)}\nUpdate CONFIG in src/config.py.")

    # Stable identifier: the row number in the raw export (PPI has no project ID
    # column, and project names aren't unique)
    df.insert(0, "project_id", np.arange(1, len(df) + 1))

    df = df.rename(columns={
        CONFIG["ppi_country_col"]: "country_name",
        CONFIG["ppi_year_col"]: "year",
        CONFIG["ppi_investment_col"]: "investment_musd",
    })

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    # Investment is kept even when unparseable (e.g. "Not Available") — we don't drop
    # whole projects just because their investment figure is missing.
    df["investment_musd"] = pd.to_numeric(df["investment_musd"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["year", "country_name"])
    if len(df) < before:
        print(f"[PPI] Dropped {before - len(df):,} row(s) with no year or country.")
    df["year"] = df["year"].astype(int)
    df["country_name"] = df["country_name"].astype(str).str.strip()

    # The World Bank PPI export contains at least one literal "**TEST**" project
    if "Project name" in df.columns:
        before = len(df)
        df = df[~df["Project name"].astype(str).str.strip().str.lower().isin(["**test**", "test"])]
        if len(df) < before:
            print(f"[PPI] Dropped {before - len(df)} placeholder/test record(s).")

    # --- Sector filter ---
    sector_col = CONFIG.get("ppi_sector_col")
    sector_value = CONFIG.get("ppi_sector_value")
    if sector_col and sector_value:
        if sector_col not in df.columns:
            sys.exit(f"[PPI] Sector column '{sector_col}' not found.\n"
                      f"Columns found: {list(df.columns)}")
        before = len(df)
        df = df[df[sector_col].astype(str).str.strip().str.lower() == sector_value.strip().lower()]
        print(f"[PPI] Sector filter '{sector_value}': kept {len(df):,} of {before:,} rows.")

    # --- Year window ---
    before = len(df)
    in_window = df["year"].between(CONFIG["year_min"], CONFIG["year_max"])
    if (~in_window).any():
        print(f"[PPI] Dropped {(~in_window).sum():,} project(s) with financial-closure year "
              f"outside {CONFIG['year_min']}-{CONFIG['year_max']} "
              f"(years dropped: {sorted(df.loc[~in_window, 'year'].unique().tolist())}).")
    df = df[in_window].copy()

    # --- Country ISO3 (needed for the sponsor comparison and the merges) ---
    df["iso3"] = df["country_name"].apply(name_to_iso3)
    unmatched = df.loc[df["iso3"].isna(), "country_name"].unique()
    if len(unmatched):
        print(f"[PPI] WARNING: {df['iso3'].isna().sum():,} rows have a country name with no "
              f"ISO3 code: {list(unmatched)} — add them to common.MANUAL_ISO3_FIXES.")

    df = add_technology_category(df)
    df = add_sponsor_indicator(df)

    return df.reset_index(drop=True)


def add_technology_category(df: pd.DataFrame) -> pd.DataFrame:
    """The Technology field is genuinely ambiguous to parse: a comma can separate two
    distinct technologies ("Coal, Hydro, Large (>50MW)") or attach a qualifier to a
    single one ("Hydro, Small (<50MW)", "Solar, PV") — there's no reliable way to tell
    these apart from the raw text alone. So rather than splitting into a list, we check
    for keywords as a substring across the WHOLE field.

    Each project gets a `technology_category`:
      - "renewable"      — only renewable keywords (solar, wind, hydro, waste, ...)
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

    print(f"[PPI] Technology categories: "
          f"{df['technology_category'].value_counts().to_dict()}")
    return df


def parse_sponsor_countries(raw) -> list[str]:
    """Sponsors Country lists one country per sponsor. IMPORTANT: entries are separated
    by a double line-break ("\\n\\n"), NOT by comma — a comma can be part of a single
    country's own name (e.g. "Hong Kong, China", "Korea, Rep."). ".." is the PPI
    export's placeholder for a sponsor whose country is unknown; it's kept here so
    unknown sponsors can be counted."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    parts = [p.strip(" ,\n") for p in re.split(r"\n\n", raw)]
    return [p for p in parts if p]


def add_sponsor_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """Classifies each sponsor as domestic (same ISO3 as the project's country) or
    foreign (different ISO3). Comparing ISO3 codes rather than raw names means naming
    variants ("Cape Verde" vs "Cabo Verde") don't cause misclassification.

    Sponsors whose country is ".." or can't be mapped to an ISO3 count as unknown and
    are left out of the domestic/foreign comparison. A project where NO sponsor
    country is known gets sponsor_type "unknown" and blank (NA) domestic/foreign
    flags, rather than being counted as "no foreign sponsor"."""
    sponsor_col = CONFIG["ppi_sponsor_country_col"]
    unmapped_names = set()

    def classify(raw, project_iso3):
        names = parse_sponsor_countries(raw)
        known_names, known_iso3 = [], []
        for name in names:
            code = None if name == ".." else name_to_iso3(name)
            if code is None:
                if name != "..":
                    unmapped_names.add(name)
                continue
            known_names.append(name)
            known_iso3.append(code)

        n_unknown = len(names) - len(known_iso3)
        if not known_iso3 or project_iso3 is None:
            return ("; ".join(known_names), "; ".join(known_iso3), len(names), 0, 0,
                    n_unknown, np.nan, "unknown", pd.NA, pd.NA)

        n_dom = sum(code == project_iso3 for code in known_iso3)
        n_for = len(known_iso3) - n_dom
        if n_dom and n_for:
            sponsor_type = "mixed"
        elif n_dom:
            sponsor_type = "domestic_only"
        else:
            sponsor_type = "foreign_only"

        return ("; ".join(known_names), "; ".join(known_iso3), len(names), n_dom, n_for,
                n_unknown, n_for / len(known_iso3), sponsor_type, n_dom > 0, n_for > 0)

    cols = ["sponsor_countries", "sponsor_iso3", "n_sponsors", "n_domestic_sponsors",
            "n_foreign_sponsors", "n_unknown_country_sponsors", "foreign_sponsor_share",
            "sponsor_type", "has_domestic_sponsor", "has_foreign_sponsor"]
    results = pd.DataFrame(
        [classify(raw, iso) for raw, iso in zip(df[sponsor_col], df["iso3"])],
        columns=cols, index=df.index,
    )
    for c in ["has_domestic_sponsor", "has_foreign_sponsor"]:
        results[c] = results[c].astype("boolean")

    if unmapped_names:
        print(f"[PPI] WARNING: sponsor country names with no ISO3 (treated as unknown): "
              f"{sorted(unmapped_names)} — add them to common.MANUAL_ISO3_FIXES.")
    print(f"[PPI] Sponsor types: {results['sponsor_type'].value_counts().to_dict()}")

    return pd.concat([df, results], axis=1)


def main():
    print("=" * 70)
    print("Step 1: prepare PPI projects")
    print("=" * 70)
    df = load_ppi_projects()
    df.to_csv(PPI_CLEAN, index=False)
    print(f"[PPI] Wrote {len(df):,} projects to {PPI_CLEAN}")


if __name__ == "__main__":
    main()
