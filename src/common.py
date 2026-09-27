"""
common.py
---------
Shared building blocks used by both build_project_panel.py and the later
country-year aggregation script: country-code harmonization and loaders for
ND-GAIN, WGI, and live WDI data. Kept separate so neither script has to
duplicate this logic.
"""

import sys
import warnings

import pandas as pd

try:
    import pycountry
except ImportError:
    pycountry = None
    warnings.warn(
        "pycountry not installed — country-code harmonization will rely only on "
        "the manual fix-up dictionary below. Run `pip install pycountry` for better coverage."
    )

try:
    import wbgapi as wb
except ImportError:
    wb = None
    warnings.warn(
        "wbgapi not installed — the WDI macro-indicator step will fail. "
        "Run `pip install wbgapi` inside your virtual environment."
    )


# Common World Bank / ND-GAIN naming variants that pycountry's fuzzy search won't catch.
# Add to this as your merge diagnostics reveal more mismatches.
MANUAL_ISO3_FIXES = {
    "korea, rep.": "KOR",
    "south korea": "KOR",
    "korea, dem. people's rep.": "PRK",
    "russian federation": "RUS",
    "russia": "RUS",
    "egypt, arab rep.": "EGY",
    "iran, islamic rep.": "IRN",
    "venezuela, rb": "VEN",
    "yemen, rep.": "YEM",
    "syrian arab republic": "SYR",
    "lao pdr": "LAO",
    "congo, dem. rep.": "COD",
    "congo, rep.": "COG",
    "gambia, the": "GMB",
    "bahamas, the": "BHS",
    "kyrgyz republic": "KGZ",
    "slovak republic": "SVK",
    "st. lucia": "LCA",
    "st. vincent and the grenadines": "VCT",
    "st. kitts and nevis": "KNA",
    "turkiye": "TUR",
    "turkey": "TUR",
    "cote d'ivoire": "CIV",
    "cabo verde": "CPV",
    "brunei darussalam": "BRN",
    "micronesia, fed. sts.": "FSM",
    "bolivia": "BOL",
    "tanzania": "TZA",
    "vietnam": "VNM",
    "moldova": "MDA",
    "laos": "LAO",
    "hong kong, china": "HKG",
    "hong kong sar, china": "HKG",
    "macao sar, china": "MAC",
    "taiwan, china": "TWN",
}


def name_to_iso3(name: str) -> str | None:
    """Best-effort country name -> ISO3 conversion."""
    if not isinstance(name, str) or not name.strip():
        return None
    key = name.strip().lower()
    if key in MANUAL_ISO3_FIXES:
        return MANUAL_ISO3_FIXES[key]
    if pycountry is not None:
        try:
            match = pycountry.countries.search_fuzzy(name)
            if match:
                return match[0].alpha_3
        except LookupError:
            pass
    return None


def report_merge(left: pd.DataFrame, right: pd.DataFrame, on, label: str) -> None:
    left_keys = set(map(tuple, left[on].values))
    right_keys = set(map(tuple, right[on].values))
    matched = left_keys & right_keys
    print(f"  [{label}] left keys: {len(left_keys):,} | right keys: {len(right_keys):,} "
          f"| matched: {len(matched):,} "
          f"({100 * len(matched) / max(len(left_keys), 1):.1f}% of left)")


def load_nd_gain(path, country_col="ISO3", year_col="Year",
                  is_wide=True, wide_id_cols=("ISO3", "Name")) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"[ND-GAIN] File not found: {path}\n"
                  f"Download from https://gain.nd.edu/our-work/country-index/download-data/")

    df = pd.read_csv(path)

    if is_wide:
        id_cols = [c for c in wide_id_cols if c in df.columns]
        year_cols = [c for c in df.columns if c not in id_cols and str(c).strip().isdigit()]
        if not year_cols:
            sys.exit(f"[ND-GAIN] Expected wide format with year columns, but found none.\n"
                      f"Columns found: {list(df.columns)}. Check wide_id_cols.")
        df = df.melt(id_vars=id_cols, value_vars=year_cols,
                      var_name="year", value_name="nd_gain_value")
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        if country_col in df.columns and "iso3" not in df.columns:
            df = df.rename(columns={country_col: "iso3"})
    else:
        df = df.rename(columns={country_col: "iso3", year_col: "year"})

    if "iso3" not in df.columns:
        sys.exit(f"[ND-GAIN] Could not find a country/ISO3 column. "
                  f"Columns found: {list(df.columns)}.")

    df["iso3"] = df["iso3"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    panel = df.groupby(["iso3", "year"], as_index=False)["nd_gain_value"].mean()

    print(f"[ND-GAIN] Loaded {panel['iso3'].nunique():,} countries, "
          f"years {panel['year'].min()}-{panel['year'].max()}.")

    return panel


def load_wgi(path, sheet_indicators=None,
             country_code_col="Economy (code)", year_col="Year",
             estimate_col="Governance estimate (approx. -2.5 to +2.5)") -> pd.DataFrame:
    """Loads the official multi-sheet WGI Excel export (wgidataset.xlsx-style), where
    each governance dimension is its own sheet ('va', 'pv', 'ge', 'rq', 'rl', 'cc'),
    already in long format (one row per country-year). This is a different shape from
    a WDI-style single wide CSV, so it's handled separately rather than reusing the
    WDI-style wide-CSV parsing pattern."""
    if sheet_indicators is None:
        sheet_indicators = {
            "va": "voice_accountability",
            "pv": "political_stability",
            "ge": "govt_effectiveness",
            "rq": "regulatory_quality",
            "rl": "rule_of_law",
            "cc": "control_of_corruption",
        }
    if not path.exists():
        sys.exit(f"[WGI] File not found: {path}\n"
                  f"Download from https://info.worldbank.org/governance/wgi/")

    frames = []
    for sheet_name, friendly_name in sheet_indicators.items():
        try:
            sheet = pd.read_excel(path, sheet_name=sheet_name)
        except ValueError as exc:
            sys.exit(f"[WGI] Sheet '{sheet_name}' not found in {path}: {exc}\n"
                      f"Check the sheet names in your file and update sheet_indicators.")

        missing = [c for c in [country_code_col, year_col, estimate_col] if c not in sheet.columns]
        if missing:
            sys.exit(f"[WGI] Sheet '{sheet_name}' is missing expected column(s): {missing}\n"
                      f"Columns found: {list(sheet.columns)}")

        sheet = sheet[[country_code_col, year_col, estimate_col]].rename(columns={
            country_code_col: "iso3",
            year_col: "year",
            estimate_col: friendly_name,
        })
        frames.append(sheet)

    # Merge all six dimensions together, one column per dimension, on iso3+year
    wide = frames[0]
    for frame in frames[1:]:
        wide = wide.merge(frame, on=["iso3", "year"], how="outer")

    wide["iso3"] = wide["iso3"].astype(str).str.upper().str.strip()
    wide = wide.dropna(subset=["year"])
    wide["year"] = wide["year"].astype(int)

    print(f"[WGI] Loaded indicators {list(sheet_indicators.values())} "
          f"for {wide['iso3'].nunique():,} countries.")

    return wide


def load_wdi_via_api(indicators, year_min, year_max) -> pd.DataFrame:
    if wb is None:
        sys.exit("[WDI] wbgapi is not installed. Run `pip install wbgapi` inside your "
                  "virtual environment, then re-run this script.")

    codes = list(indicators.keys())
    print(f"[WDI] Requesting {len(codes)} indicators from the World Bank API for "
          f"{year_min}-{year_max} (needs an internet connection)...")

    try:
        raw = wb.data.DataFrame(
            codes,
            time=range(year_min, year_max + 1),
            index=["economy", "time"],
            columns="series",
            numericTimeKeys=True,
            skipBlanks=True,
            skipAggs=True,
        )
    except Exception as exc:
        sys.exit(f"[WDI] API request failed: {exc}\n"
                  f"Check your internet connection, or that the codes are valid "
                  f"World Bank series codes — look them up at "
                  f"https://databank.worldbank.org/metadataglossary")

    df = raw.reset_index().rename(columns={"economy": "iso3", "time": "year"})
    df = df.rename(columns=indicators)
    df["year"] = df["year"].astype(int)

    print(f"[WDI] Retrieved {list(indicators.values())} for {df['iso3'].nunique():,} countries.")

    return df
