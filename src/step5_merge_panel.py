"""
step5_merge_panel.py
--------------------
Merges ND-GAIN, WGI and WDI onto the cleaned PPI projects by (iso3, year). This is
a many-to-one LEFT merge: every project is kept, many projects can share one
country-year, and a project whose country-year is missing from a source simply
gets blanks for that source's columns — nothing is dropped for missing ND-GAIN,
WGI or WDI data. Writes data/processed/project_panel.csv (one row per project).

Run from the project root:  python3 src/step5_merge_panel.py
"""

import sys
import warnings

import pandas as pd

from common import report_merge
from config import CONFIG, PPI_CLEAN, ND_GAIN_PANEL, WGI_PANEL, WDI_PANEL, PROJECT_PANEL

# By default pandas reads text like "NA", "N/A" or "None" as missing, which would
# silently blank out genuine PPI text values (e.g. Technology "Solar, PV, N/A").
# Only truly empty cells should be treated as missing.
_READ = dict(keep_default_na=False, na_values=[""])


def read_country_year(path, label):
    if not path.exists():
        sys.exit(f"[{label}] {path} not found — run the earlier step first (or run_all.py).")
    df = pd.read_csv(path, **_READ)
    df["year"] = df["year"].astype(int)
    return df


def main():
    print("=" * 70)
    print("Step 5: merge into the project-country-year panel")
    print("=" * 70)

    if not PPI_CLEAN.exists():
        sys.exit(f"[PPI] {PPI_CLEAN} not found — run step1_prepare_ppi.py first.")
    ppi = pd.read_csv(PPI_CLEAN, **_READ, low_memory=False)
    for c in ["has_domestic_sponsor", "has_foreign_sponsor"]:
        ppi[c] = ppi[c].map({"True": True, "False": False, True: True, False: False}).astype("boolean")

    nd_gain = read_country_year(ND_GAIN_PANEL, "ND-GAIN")
    wgi = read_country_year(WGI_PANEL, "WGI")

    wdi_cols = list(CONFIG["wdi_indicators"].values())
    if WDI_PANEL.exists():
        wdi = read_country_year(WDI_PANEL, "WDI")
    else:
        warnings.warn("[WDI] No WDI data available (step 4 couldn't reach the API) — "
                      "WDI columns will be blank for every project.")
        wdi = pd.DataFrame(columns=["iso3", "year", *wdi_cols]).astype({"year": int})

    print("\nMerge diagnostics (PPI project-rows -> one country-year row per source):")
    merged = ppi
    for label, source in [("ND-GAIN", nd_gain), ("WGI", wgi), ("WDI", wdi)]:
        report_merge(merged, source, ["iso3", "year"], f"PPI -> {label}")
        merged = merged.merge(source, on=["iso3", "year"], how="left", validate="m:1")
    assert len(merged) == len(ppi), "merge changed the number of projects"

    # Identifier and key analysis columns first, all original PPI columns after
    front = ["project_id", "iso3", "country_name", "year", "technology_category",
             "is_renewable", "is_pure_renewable", "investment_musd", "sponsor_type",
             "has_domestic_sponsor", "has_foreign_sponsor", "n_sponsors",
             "n_domestic_sponsors", "n_foreign_sponsors", "n_unknown_country_sponsors",
             "foreign_sponsor_share", "sponsor_countries", "sponsor_iso3",
             "nd_gain_value", *CONFIG["wgi_sheet_indicators"].values(), *wdi_cols]
    merged = merged[front + [c for c in merged.columns if c not in front]]
    merged = merged.sort_values(["iso3", "year", "project_id"]).reset_index(drop=True)

    merged.to_csv(PROJECT_PANEL, index=False)
    print(f"\nWrote {len(merged):,} project rows, {len(merged.columns)} columns, to {PROJECT_PANEL}")


if __name__ == "__main__":
    main()
