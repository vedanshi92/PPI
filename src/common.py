"""
common.py
---------
Shared building blocks used by the pipeline steps: country-code harmonization and
loaders for ND-GAIN, WGI, and live WDI data. Kept separate so no step has to
duplicate this logic.
"""

import sys
import warnings
from functools import lru_cache

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


# World Bank / PPI naming variants that pycountry won't resolve (or resolves WRONGLY
# via fuzzy search — e.g. fuzzy "Niger" returns Nigeria, and "Kosovo" returns Serbia).
# Codes follow the World Bank convention (e.g. Kosovo = XKX), which is what WGI and
# WDI use. Add to this as your merge diagnostics reveal more mismatches.
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
    "côte d'ivoire": "CIV",
    "cabo verde": "CPV",
    "cape verde": "CPV",
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
    "niger": "NER",
    "kosovo": "XKX",
    "west bank and gaza": "PSE",
    "faeroe islands": "FRO",
    "faroe islands": "FRO",
    "guyana, cr": "GUY",
    "macedonia, fyr": "MKD",
    "north macedonia": "MKD",
    "czech republic": "CZE",
    "são tomé and principe": "STP",
    "sao tome and principe": "STP",
}


@lru_cache(maxsize=None)
def name_to_iso3(name: str) -> str | None:
    """Country name -> ISO3. Tries, in order: the manual dictionary, an EXACT
    pycountry lookup (name / official name / code), and only then pycountry's fuzzy
    search. Fuzzy matches are printed so they can be checked by eye, since fuzzy
    search is what produced errors like Niger -> Nigeria."""
    if not isinstance(name, str) or not name.strip():
        return None
    name = name.strip()
    key = name.lower()
    if key in MANUAL_ISO3_FIXES:
        return MANUAL_ISO3_FIXES[key]
    if pycountry is None:
        return None
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        pass
    try:
        match = pycountry.countries.search_fuzzy(name)
        if match:
            print(f"  [ISO3] fuzzy match: '{name}' -> {match[0].alpha_3} ({match[0].name}) "
                  f"— add to MANUAL_ISO3_FIXES if wrong")
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

    df = df.dropna(subset=["iso3", "year"])
    df["iso3"] = df["iso3"].astype(str).str.upper().str.strip()
    df["year"] = df["year"].astype(int)
    df["nd_gain_value"] = pd.to_numeric(df["nd_gain_value"], errors="coerce")

    panel = df.groupby(["iso3", "year"], as_index=False)["nd_gain_value"].mean()

    print(f"[ND-GAIN] Loaded {panel['iso3'].nunique():,} countries, "
          f"years {panel['year'].min()}-{panel['year'].max()}.")

    return panel


def load_wgi(path, sheet_indicators,
             country_code_col="Economy (code)", year_col="Year",
             estimate_col="Governance estimate (approx. -2.5 to +2.5)") -> pd.DataFrame:
    """Loads the official multi-sheet WGI Excel export, where each governance dimension
    is its own sheet ('va', 'pv', 'ge', 'rq', 'rl', 'cc'), already in long format (one
    row per country-year). All sheets are read in a single pass — the workbook is large
    and re-opening it once per sheet is slow."""
    if not path.exists():
        sys.exit(f"[WGI] File not found: {path}\n"
                  f"Download from https://www.worldbank.org/en/publication/worldwide-governance-indicators")

    try:
        sheets = pd.read_excel(path, sheet_name=list(sheet_indicators))
    except ValueError as exc:
        sys.exit(f"[WGI] Couldn't read sheets {list(sheet_indicators)} from {path}: {exc}\n"
                  f"Check the sheet names in your file and update wgi_sheet_indicators.")

    frames = []
    for sheet_name, friendly_name in sheet_indicators.items():
        sheet = sheets[sheet_name]
        missing = [c for c in [country_code_col, year_col, estimate_col] if c not in sheet.columns]
        if missing:
            sys.exit(f"[WGI] Sheet '{sheet_name}' is missing expected column(s): {missing}\n"
                      f"Columns found: {list(sheet.columns)}")

        sheet = sheet[[country_code_col, year_col, estimate_col]].rename(columns={
            country_code_col: "iso3",
            year_col: "year",
            estimate_col: friendly_name,
        })
        # Missing estimates can appear as ".." or blank text in some WGI releases
        sheet[friendly_name] = pd.to_numeric(sheet[friendly_name], errors="coerce")
        sheet = sheet.dropna(subset=["iso3", "year"])
        sheet["iso3"] = sheet["iso3"].astype(str).str.upper().str.strip()
        sheet["year"] = sheet["year"].astype(int)
        frames.append(sheet)

    # Merge all six dimensions together, one column per dimension, on iso3+year
    wide = frames[0]
    for frame in frames[1:]:
        wide = wide.merge(frame, on=["iso3", "year"], how="outer", validate="1:1")

    print(f"[WGI] Loaded indicators {list(sheet_indicators.values())} "
          f"for {wide['iso3'].nunique():,} countries, "
          f"years {wide['year'].min()}-{wide['year'].max()}.")

    return wide


def load_wdi_via_api(indicators, year_min, year_max) -> pd.DataFrame | None:
    """Pulls WDI series from the World Bank API. Returns None (instead of exiting) if
    the API can't be reached, so the caller can decide whether to fall back to a
    previously saved copy or carry on with WDI columns left blank."""
    if wb is None:
        warnings.warn("[WDI] wbgapi is not installed. Run `pip install wbgapi`.")
        return None

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
        warnings.warn(f"[WDI] API request failed: {type(exc).__name__}: {exc}\n"
                      f"Check your internet connection (api.worldbank.org must be reachable), "
                      f"or that the codes are valid World Bank series codes.")
        return None

    df = raw.reset_index().rename(columns={"economy": "iso3", "time": "year"})
    df = df.rename(columns=indicators)
    # A series with no data at all in the window is dropped by skipBlanks — keep the
    # column anyway so the output schema doesn't depend on what the API returned.
    for name in indicators.values():
        if name not in df.columns:
            df[name] = pd.NA
    df["year"] = df["year"].astype(int)
    df = df[["iso3", "year", *indicators.values()]]

    print(f"[WDI] Retrieved {list(indicators.values())} for {df['iso3'].nunique():,} countries.")

    return df
