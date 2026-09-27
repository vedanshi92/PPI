"""
step4_fetch_wdi.py
------------------
Pulls the WDI macro indicators listed in CONFIG["wdi_indicators"] live from the
World Bank API and writes data/interim/wdi_panel.csv.

If the API can't be reached, this step does NOT fail the pipeline: it keeps the
copy saved by the last successful run if there is one, and otherwise writes
nothing — step 5 then adds the WDI columns as blanks and says so loudly.

Run from the project root:  python3 src/step4_fetch_wdi.py
"""

from common import load_wdi_via_api
from config import CONFIG, WDI_PANEL


def main():
    print("=" * 70)
    print("Step 4: fetch WDI from the World Bank API")
    print("=" * 70)
    panel = load_wdi_via_api(CONFIG["wdi_indicators"], CONFIG["year_min"], CONFIG["year_max"])
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
