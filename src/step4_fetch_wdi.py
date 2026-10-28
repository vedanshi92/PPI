"""
step4_fetch_wdi.py
------------------
Pulls the WDI macro indicators listed in CONFIG["wdi_indicators"] live from the
World Bank API and writes data/interim/wdi_panel.csv.

If the API can't be reached, this step does NOT fail the pipeline: it keeps the
copy saved by the last successful run if there is one, and otherwise writes
nothing — step 5 then adds the WDI columns as blanks and says so loudly. A single
series that fails to download is taken from that saved copy too, if it has it.

Run from the project root:  python3 src/step4_fetch_wdi.py
"""

import pandas as pd

from common import load_wdi_via_api
from config import CONFIG, WDI_PANEL


def main():
    print("=" * 70)
    print("Step 4: fetch WDI from the World Bank API")
    print("=" * 70)
    panel = load_wdi_via_api(CONFIG["wdi_indicators"], CONFIG["year_min"], CONFIG["year_max"])
    if panel is not None and WDI_PANEL.exists():
        # A series that failed to download comes back blank. Take it from the copy saved
        # by an earlier run instead of overwriting that copy with blanks.
        failed = [c for c in CONFIG["wdi_indicators"].values() if panel[c].isna().all()]
        saved = pd.read_csv(WDI_PANEL)
        failed = [c for c in failed if c in saved.columns and saved[c].notna().any()]
        if failed:
            panel = panel.drop(columns=failed).merge(
                saved[["iso3", "year", *failed]], on=["iso3", "year"], how="outer")
            print(f"[WDI] Kept {failed} from the copy saved by an earlier run.")
    if panel is not None:
        panel.to_csv(WDI_PANEL, index=False)
        print(f"[WDI] Wrote {len(panel):,} country-year rows to {WDI_PANEL}")
    elif WDI_PANEL.exists():
        print(f"[WDI] API unavailable — keeping the copy saved by an earlier run: {WDI_PANEL}")
    else:
        print("[WDI] API unavailable and no earlier copy saved — WDI columns will be blank "
              "in the final panel. Re-run with internet access to fill them in.")


if __name__ == "__main__":
    main()
