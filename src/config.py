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
ENERGY_DEALS = INTERIM / "ppi_energy_deals_all.csv"  # all technologies, for entry dates
FIT_POLICIES = INTERIM / "ren21_fit_policies.csv"  # parsed RT5 FiT dates, for checking
NM_POLICIES = INTERIM / "ren21_nm_policies.csv"  # parsed RT5 net metering/billing dates
ND_GAIN_PANEL = INTERIM / "nd_gain_panel.csv"
WGI_PANEL = INTERIM / "wgi_panel.csv"
WDI_PANEL = INTERIM / "wdi_panel.csv"

# Final outputs
PROJECT_PANEL = OUT / "project_panel.csv"
SUMMARY_XLSX = OUT / "summary_stats.xlsx"
SUMMARY_TXT = OUT / "summary_stats.txt"
STACKING_TXT = OUT / "twfe_policy_stacking.txt"
COUNTRY_YEAR_PANEL = OUT / "country_year_panel.csv"
HAZARD_TXT = OUT / "hazard_model.txt"
LP_TXT = OUT / "local_projections.txt"
STACKED_DID_TXT = OUT / "stacked_did.txt"

CONFIG = {
    # --- PPI database ---
    "ppi_file": RAW / "ppi.xlsx",
    "ppi_sheet": "CustomQuery",
    "ppi_country_col": "Country",
    "ppi_year_col": "Financial closure year",
    "ppi_investment_col": "TotalInvestment",
    "ppi_capacity_col": "Capacity",
    "ppi_capacity_type_col": "CapacityType",
    # Values in TotalInvestment / Capacity that mean "no figure" (lowercased) -> blank
    "missing_figure_values": ["not available", "not applicable", "n/a", "na"],
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

    # De-risking: a project is derisked if any multilateral or bilateral support entry
    # is one of these support types. Lowercased, substring match on the support type
    # (e.g. "Guarantee", "Risk Management (including Political Risk Insurance)"), not
    # the agency name, so a loan from e.g. "Export Finance and Insurance Corporation"
    # doesn't count. Blank or "No" means no support, i.e. derisk = 0.
    "ppi_multilateral_col": "MultiLateralSupport",
    "ppi_bilateral_col": "BiLateralSupport",
    "derisk_keywords": ["guarantee", "risk management", "insurance"],
    # public_finance: same matching. "equity" also matches "Quasi-equity".
    "public_finance_keywords": ["loan", "syndication", "equity"],

    # govt_derisk = 1 if InDirectGovtSupport lists any support (payment, tariff, revenue,
    # debt or exchange-rate guarantees, tax deductions/credits, ...). These placeholder
    # values, lowercased, mean no support and give 0, as does a blank cell.
    # govt_finance is built the same way from DirectGovtSupport (revenue subsidy, capital
    # subsidy, in-kind).
    "ppi_indirect_govt_col": "InDirectGovtSupport",
    "ppi_direct_govt_col": "DirectGovtSupport",
    "no_support_values": ["no", "n/a", "na", "not applicable"],
    # ...while these mean the field wasn't recorded, so govt_derisk is left blank. Most
    # projects outside 2010-2019 are "Not Available".
    "govt_support_missing_values": ["not available"],

    # auction_deal = 1 for greenfield renewable projects (excluding large hydro) awarded by
    # competitive bidding on lowest tariff, subsidy, cost or government payments; 0
    # otherwise. Blank when such a project's AwardMethod is "Not Available", or it was
    # competitively bid but its BidCriteria is "Not Available" (or either is empty).
    # Exact matches, case-insensitive.
    "ppi_award_col": "AwardMethod",
    "ppi_bid_criteria_col": "BidCriteria",
    "ppi_type_col": "Type of PPI",
    "auction_award_method": "competitive bidding",
    "auction_bid_criteria": ["lowest tariff", "lowest subsidy required",
                             "lowest cost of construction or operation",
                             "lowest government payments"],
    "auction_missing_values": ["not available"],
    "greenfield_value": "greenfield project",
    "large_hydro_keyword": "hydro, large",

    # tech: main renewable technology of a renewable project, matched on Technology
    # (lowercased substring). A project matching more than one is "hybrid".
    "tech_keywords": {
        "solar": "solar", "wind": "wind", "hydro_large": "hydro, large",
        "hydro_small": "hydro, small", "geothermal": "geothermal", "biomass": "biomass",
        "biogas": "biogas", "waste": "waste",
    },

    # --- REN21 renewable policies ---
    # GSR 2025 Global Overview Figure 9, "References" sheet — one row per country and policy
    # with the year enacted (net billing counts as net metering). Used only as a fallback
    # adoption year where RT5 has none.
    "ren21_file": RAW / "ren21" / "GSR_2025_GO_Figure_9.xlsx",
    "ren21_sheet": "References",
    "ren21_policy_map": {"feed-in tariff": "fit", "net-metering": "nm", "net-billing": "nm"},
    # fit_active / nm_active: GSR 2023 Supply Data Pack, table RT5 — year adopted and year
    # removed (and reintroductions) for feed-in tariffs and for net metering/billing.
    "ren21_rt5_file": RAW / "ren21" / "GSR2023_Supply-Data-Pack.xlsx",
    "ren21_rt5_sheet": "RT5",

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
    # ND-GAIN vulnerability score (0-1, higher = more vulnerable), same wide format
    "nd_gain_vulnerability_file": RAW / "nd_gain_vulnerability.csv",
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
        "EG.ELC.ACCS.ZS": "electricity_access_pct",
        "FS.AST.PRVT.GD.ZS": "private_credit_pct_gdp",
        # Renewable energy consumption, % of total final energy consumption. Chosen over
        # fossil fuel energy consumption (EG.USE.COMM.FO.ZS), which has real values only to
        # 2014-15: the API returns 0 (not missing) for every country from 2016 on. Real
        # coverage: 90.4% of projects here vs 65.7%. This series ends in 2021-22.
        "EG.FEC.RNEW.ZS": "renewable_energy_consumption_pct",
    },

    # Projects with a financial-closure year outside this window are dropped.
    "year_min": 1990,
    "year_max": 2025,
}
