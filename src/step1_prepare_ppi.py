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
  5. Flags support types in MultiLateralSupport / BiLateralSupport: derisk (guarantee,
     risk management, insurance) and public_finance (loan, syndication, equity)
  6. Flags govt_derisk (any entry in InDirectGovtSupport) and govt_finance (any entry in
     DirectGovtSupport), and any_support if any of the four flags is 1
  7. Flags auction_deal: greenfield renewable (not large hydro) awarded by competitive
     bidding on lowest tariff, subsidy or cost
Writes data/interim/ppi_projects.csv. Every original PPI column is kept.

Run from the project root:  python3 src/step1_prepare_ppi.py
"""

import re
import sys

import numpy as np
import pandas as pd

from common import name_to_iso3
from config import CONFIG, PPI_CLEAN


def to_number(values: pd.Series, label: str) -> pd.Series:
    """Converts a PPI figure column to numbers. Placeholders such as "Not Available" and
    blank cells become blank; thousands separators ("1,200") are removed. Any other
    unreadable value is also blanked but printed, so it can be checked."""
    text = values.astype("string").str.strip().str.replace(",", "", regex=False)
    numbers = pd.to_numeric(text, errors="coerce")
    placeholder = text.isna() | text.str.lower().isin(["", *CONFIG["missing_figure_values"]])
    unreadable = numbers.isna() & ~placeholder
    if unreadable.any():
        print(f"[PPI] WARNING: {unreadable.sum():,} {label} value(s) aren't numbers and were "
              f"left blank: {values[unreadable].value_counts().head(10).to_dict()}")
    return numbers


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
        CONFIG["ppi_capacity_col"]: "capacity",
    })

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    # Investment and capacity: "Not Available" etc. become blank. Projects are kept —
    # we don't drop whole projects just because a figure is missing.
    df["investment_musd"] = to_number(df["investment_musd"], "investment_musd")
    df["capacity"] = to_number(df["capacity"], "capacity")
    # Capacity is in different units by project (MW, thousands of connections, km, ...),
    # so also keep a column with the MW figures only.
    is_mw = df[CONFIG["ppi_capacity_type_col"]].astype(str).str.strip() == "MW"
    df["capacity_mw"] = df["capacity"].where(is_mw)
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
    # derisk: guarantee, risk management or insurance from a multilateral/bilateral agency
    df = add_support_flag(df, "derisk", CONFIG["derisk_keywords"])
    # public_finance: loan, syndication or equity from a multilateral/bilateral agency
    df = add_support_flag(df, "public_finance", CONFIG["public_finance_keywords"])
    # govt_derisk: indirect government support (guarantees, tax deductions/credits)
    df = add_govt_flag(df, CONFIG["ppi_indirect_govt_col"], "govt_derisk")
    # govt_finance: direct government support (revenue/capital subsidies, in-kind)
    df = add_govt_flag(df, CONFIG["ppi_direct_govt_col"], "govt_finance")
    # any_support: any of the four support flags is 1. A blank government flag counts as
    # "not 1", so any_support is never blank.
    support_flags = ["derisk", "public_finance", "govt_derisk", "govt_finance"]
    df["any_support"] = (df[support_flags] == 1).fillna(False).any(axis=1).astype(int)
    print(f"[PPI] any_support = 1 for {df['any_support'].sum():,} of {len(df):,} projects")
    df = add_auction_deal(df)
    df = add_tech(df)

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


# Support type inside "Agency (Type / $amount / year)". The type may itself contain one
# level of parentheses: "Risk Management (including Political Risk Insurance)".
SUPPORT_TYPE_RE = re.compile(r"\(((?:[^()]|\([^()]*\))*?)\s*/\s*\$")


def support_types(raw) -> list[str]:
    """Support types listed in a MultiLateralSupport / BiLateralSupport cell. Blank and
    "No" give an empty list."""
    if pd.isna(raw) or str(raw).strip().lower() in ("", "no"):
        return []
    return [t.strip() for t in SUPPORT_TYPE_RE.findall(str(raw))]


def add_support_flag(df: pd.DataFrame, name: str, keywords: list[str]) -> pd.DataFrame:
    """`name` = 1 if any multilateral or bilateral support entry has a support type
    containing one of `keywords`, else 0. Blank or "No" counts as 0. Also adds
    `name`_multilateral and `name`_bilateral for each column separately."""
    parts = []
    for col, suffix in [(CONFIG["ppi_multilateral_col"], "multilateral"),
                        (CONFIG["ppi_bilateral_col"], "bilateral")]:
        if col not in df.columns:
            sys.exit(f"[PPI] Support column '{col}' not found.\n"
                     f"Columns found: {list(df.columns)}")
        types = df[col].apply(support_types)
        df[f"{name}_{suffix}"] = types.apply(
            lambda ts: int(any(kw in t.lower() for t in ts for kw in keywords)))
        parts.append(f"{suffix} {df[f'{name}_{suffix}'].sum():,}")

        # A cell with support listed but no type parsed would silently count as 0
        has_support = ~df[col].fillna("").astype(str).str.strip().str.lower().isin(["", "no"])
        unparsed = has_support & (types.str.len() == 0)
        if unparsed.any():
            print(f"[PPI] WARNING: {unparsed.sum():,} '{col}' value(s) list support but no "
                  f"support type could be read; counted as {name} = 0: "
                  f"{df.loc[unparsed, col].unique()[:5].tolist()}")

    df[name] = (df[f"{name}_multilateral"] | df[f"{name}_bilateral"]).astype(int)
    print(f"[PPI] {name} = 1 for {df[name].sum():,} of {len(df):,} projects "
          f"({', '.join(parts)})")
    return df


def add_govt_flag(df: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
    """`name` = 1 if government-support column `col` has any entry, else 0. Blank cells
    and placeholders such as "Not Applicable" count as 0. "Not Available" means the PPI
    didn't record the field, so it is left blank (missing) instead."""
    if col not in df.columns:
        sys.exit(f"[PPI] Column '{col}' not found.\nColumns found: {list(df.columns)}")
    placeholders = set(CONFIG["no_support_values"])
    missing = set(CONFIG["govt_support_missing_values"])

    def has_entry(raw):
        if pd.isna(raw):
            return 0
        entries = [e.strip().lower() for e in str(raw).split(",")]
        if any(e and e not in placeholders | missing for e in entries):
            return 1
        if any(e in missing for e in entries):
            return pd.NA
        return 0

    df[name] = df[col].apply(has_entry).astype("Int64")
    print(f"[PPI] {name} = 1 for {(df[name] == 1).sum():,}, "
          f"0 for {(df[name] == 0).sum():,}, "
          f"blank (not available) for {df[name].isna().sum():,} of {len(df):,} projects")
    return df


def add_auction_deal(df: pd.DataFrame) -> pd.DataFrame:
    """auction_deal = 1 for a greenfield renewable project, other than large hydro, awarded
    by competitive bidding with lowest tariff, subsidy, cost or government payments as the
    bid criterion. Such a project is left blank when its award method isn't recorded, or it
    was competitively bid but its bid criterion isn't recorded ("Not Available" or empty).
    Every other project, including non-greenfield ones, is 0."""
    for key in ["ppi_award_col", "ppi_bid_criteria_col", "ppi_type_col"]:
        if CONFIG[key] not in df.columns:
            sys.exit(f"[PPI] Column '{CONFIG[key]}' not found.\n"
                     f"Columns found: {list(df.columns)}")

    def norm(col):
        return df[CONFIG[col]].astype("string").str.strip().str.lower()

    renewable = df["technology_category"] == "renewable"
    large_hydro = (df[CONFIG["ppi_technology_col"]].astype(str).str.lower()
                   .str.contains(CONFIG["large_hydro_keyword"], regex=False))
    greenfield = norm("ppi_type_col") == CONFIG["greenfield_value"]
    competitive = norm("ppi_award_col") == CONFIG["auction_award_method"]
    def not_recorded(col):
        values = norm(col)
        return (values.isna() | values.isin(["", *CONFIG["auction_missing_values"]])).fillna(True)

    criteria = norm("ppi_bid_criteria_col").isin(CONFIG["auction_bid_criteria"]).fillna(False)
    eligible = (renewable & ~large_hydro & greenfield).fillna(False)
    award_missing = eligible & not_recorded("ppi_award_col")
    criteria_missing = eligible & competitive.fillna(False) & not_recorded("ppi_bid_criteria_col")

    df["auction_deal"] = (eligible & competitive.fillna(False) & criteria).astype("Int64")
    df.loc[award_missing | criteria_missing, "auction_deal"] = pd.NA
    print(f"[PPI] auction_deal = 1 for {(df['auction_deal'] == 1).sum():,}, "
          f"0 for {(df['auction_deal'] == 0).sum():,}, blank for "
          f"{df['auction_deal'].isna().sum():,} (award method not available "
          f"{award_missing.sum():,}, bid criteria not available {criteria_missing.sum():,}) "
          f"of {len(df):,} projects (greenfield renewable excl. large hydro: {eligible.sum():,})")
    return df


def add_tech(df: pd.DataFrame) -> pd.DataFrame:
    """tech: the main renewable technology of a renewable project (solar, wind, hydro_large,
    hydro_small, geothermal, biomass, biogas, waste), or "hybrid" if it lists more than one.
    Blank for non-renewable and mixed projects. Used as a fixed effect in step 7."""
    def classify(raw):
        t = str(raw).lower()
        found = {name for name, kw in CONFIG["tech_keywords"].items() if kw in t}
        return found.pop() if len(found) == 1 else ("hybrid" if found else pd.NA)

    renewable = df["technology_category"] == "renewable"
    df["tech"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df.loc[renewable, "tech"] = df.loc[renewable, CONFIG["ppi_technology_col"]].apply(classify)
    print(f"[PPI] tech (renewable projects): {df['tech'].value_counts().to_dict()}")
    return df


def main():
    print("=" * 70)
    print("Step 1: prepare PPI projects")
    print("=" * 70)
    df = load_ppi_projects()
    df.to_csv(PPI_CLEAN, index=False)
    print(f"[PPI] Wrote {len(df):,} projects to {PPI_CLEAN}")


if __name__ == "__main__":
    main()
