"""
run_all.py
----------
Runs the whole pipeline in order. Each step can also be run on its own
(e.g. `python3 src/step6_summary_stats.py`) once the earlier steps have written
their outputs to data/interim/.

Run from the project root:  python3 run_all.py
"""

import runpy
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

STEPS = [
    "step1_prepare_ppi.py",
    "step2_load_ndgain.py",
    "step3_load_wgi.py",
    "step4_fetch_wdi.py",
    "step5_merge_panel.py",
    "step6_summary_stats.py",
]

if __name__ == "__main__":
    for step in STEPS:
        runpy.run_path(str(SRC / step), run_name="__main__")
        print()
