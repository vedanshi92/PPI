"""
config.py
---------
Paths and settings shared by every step of the pipeline. Edit this file (not the
step scripts) to point at your own downloaded files or change what gets pulled.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"      # outputs of steps 1-4, inputs to step 5
OUT = ROOT / "data" / "processed"        # final panel + summary statistics

for _d in (INTERIM, OUT):
    _d.mkdir(parents=True, exist_ok=True)

# Intermediate files written by each step
PPI_CLEAN = INTERIM / "ppi_projects.csv"
ND_GAIN_PANEL = INTERIM / "nd_gain_panel.csv"
WGI_PANEL = INTERIM / "wgi_panel.csv"
WDI_PANEL = INTERIM / "wdi_panel.csv"

# Final outputs
PROJECT_PANEL = OUT / "project_panel.csv"
SUMMARY_XLSX = OUT / "summary_stats.xlsx"
SUMMARY_TXT = OUT / "summary_stats.txt"

CONFIG = {
    # --- PPI database ---
    "ppi_file": RAW / "ppi.xlsx",
    "ppi_sheet": "CustomQuery",
    "ppi_country_col": "Country",
    "ppi_year_col": "Financial closure year",
    "ppi_investment_col": "TotalInvestment",
    "ppi_technology_col": "Technology",
    "ppi_sponsor_country_col": "Sponsors Country",

    # Restrict to a single PPI sector. Set to None to keep all sectors.
    "ppi_sector_col": "Primary sector",
    "ppi_sector_value": "Energy",

    # Keywords used to classify the Technology field. Lowercased, substring match.
    # NOTE: "Waste" (waste-to-energy) is a genuine judgment call in the literature —
    # it's counted as renewable here. Move "waste" to nonrenewable_keywords if you
    # want to count it as non-renewable instead.
    # NOTE: use "natural gas", not plain "gas" — "gas" would also match "biogas".
    "renewable_keywords": ["solar", "wind", "hydro", "geothermal", "biomass", "biogas", "waste"],
    "nonrenewable_keywords": ["coal", "diesel", "natural gas", "nuclear", "steam"],

    # --- World Governance Indicators (WGI) ---
    # The official multi-sheet WGI Excel export: one sheet per governance dimension,
    # each already in long format (one row per country-year).
    "wgi_file": RAW / "wgidata.xlsx",
    "wgi_sheet_indicators": {
        "va": "voice_accountability",
        "pv": "political_stability",
        "ge": "govt_effectiveness",
        "rq": "regulatory_quality",
        "rl": "rule_of_law",
        "cc": "control_of_corruption",
    },

    # --- ND-GAIN (wide CSV: ISO3, Name, then one column per year) ---
    "nd_gain_file": RAW / "nd_gain.csv",
    "nd_gain_country_col": "ISO3",
    "nd_gain_wide_id_cols": ["ISO3", "Name"],

    # --- WDI macro indicators, pulled live from the World Bank API ---
    # Look up series codes at https://databank.worldbank.org/metadataglossary
    "wdi_indicators": {
        "NY.GDP.PCAP.KD": "gdp_per_capita",
        "NY.GDP.MKTP.KD.ZG": "gdp_growth",
        "SP.POP.TOTL": "population",
        "FP.CPI.TOTL.ZG": "inflation_cpi",
        "PA.NUS.FCRF": "exchange_rate_lcu_usd",
        "NE.TRD.GNFS.ZS": "trade_pct_gdp",
        "BX.KLT.DINV.WD.GD.ZS": "fdi_net_inflows_pct_gdp",
        "FR.INR.RINR": "real_interest_rate",
        "GC.DOD.TOTL.GD.ZS": "govt_debt_pct_gdp",
        "EG.ELC.ACCS.ZS": "electricity_access_pct",
    },

    # Projects with a financial-closure year outside this window are dropped.
    "year_min": 1990,
    "year_max": 2025,
}
