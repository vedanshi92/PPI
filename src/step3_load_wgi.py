"""
step3_load_wgi.py
-----------------
Combines the six WGI governance dimensions (one sheet each) into a single
country-year panel with one column per dimension. Writes data/interim/wgi_panel.csv.

Run from the project root:  python3 src/step3_load_wgi.py
"""

from common import load_wgi
from config import CONFIG, WGI_PANEL


def main():
    print("=" * 70)
    print("Step 3: load WGI")
    print("=" * 70)
    panel = load_wgi(CONFIG["wgi_file"], sheet_indicators=CONFIG["wgi_sheet_indicators"])
    panel.to_csv(WGI_PANEL, index=False)
    print(f"[WGI] Wrote {len(panel):,} country-year rows to {WGI_PANEL}")


if __name__ == "__main__":
    main()
