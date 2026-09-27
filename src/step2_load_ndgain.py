"""
step2_load_ndgain.py
--------------------
Reshapes the wide ND-GAIN country index (one column per year) into a long
country-year panel. Writes data/interim/nd_gain_panel.csv.

Run from the project root:  python3 src/step2_load_ndgain.py
"""

from common import load_nd_gain
from config import CONFIG, ND_GAIN_PANEL


def main():
    print("=" * 70)
    print("Step 2: load ND-GAIN")
    print("=" * 70)
    panel = load_nd_gain(
        CONFIG["nd_gain_file"],
        country_col=CONFIG["nd_gain_country_col"],
        wide_id_cols=CONFIG["nd_gain_wide_id_cols"],
    )
    panel.to_csv(ND_GAIN_PANEL, index=False)
    print(f"[ND-GAIN] Wrote {len(panel):,} country-year rows to {ND_GAIN_PANEL}")


if __name__ == "__main__":
    main()
