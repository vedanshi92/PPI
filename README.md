# PPI Renewable Energy Panel — Replication Code

Builds a **project-country-year panel**: one row per World Bank PPI energy project, tagged
with its country (ISO3) and financial-closure year, with country-year data from **ND-GAIN**,
the **Worldwide Governance Indicators (WGI)** and **World Bank WDI** merged on.

Step 9 then collapses it to a **balanced country-year panel** (`country_year_panel.csv`).

## 1. Folder structure

```
PPI/
├── run_all.py                  <- runs every step in order
├── src/
│   ├── config.py               <- file paths + settings (edit this, not the steps)
│   ├── common.py               <- shared loaders and country -> ISO3 matching
│   ├── step1_prepare_ppi.py    <- clean PPI, classify technology, parse sponsors
│   ├── step2_load_ndgain.py    <- ND-GAIN wide -> country-year panel
│   ├── step3_load_wgi.py       <- six WGI sheets -> one country-year panel
│   ├── step4_fetch_wdi.py      <- WDI macro indicators from the World Bank API
│   ├── step5_merge_panel.py    <- merge everything onto the projects
│   ├── step6_summary_stats.py  <- descriptive statistics
│   ├── step7_twfe_stacking.py  <- fixed effects regressions
│   ├── step8_figures.py        <- summary figures
│   └── step9_country_year_panel.py <- balanced country-year panel
├── data/                       <- not committed to git
│   ├── raw/                    <- put your downloaded files here (see step 2)
│   ├── interim/                <- outputs of steps 1-4
│   └── processed/              <- final panel + summary statistics
├── requirements.txt
└── README.md
```

## 2. Download the raw data

Place each file in `data/raw/` with the filename below (rename after downloading if needed):

| Dataset | Where to get it | Save as |
|---|---|---|
| World Bank PPI database | https://ppi.worldbankgroup.org/en/ppi -> custom query, Excel export | `data/raw/ppi.xlsx` |
| ND-GAIN Country Index | https://gain.nd.edu/our-work/country-index/download-data/ (the `gain.csv` country time series) | `data/raw/nd_gain.csv` |
| REN21 feed-in tariff / net metering policies | REN21 Global Status Report 2025, Global Overview Figure 9 (Excel, "References" sheet) | `data/raw/ren21/GSR_2025_GO_Figure_9.xlsx` |
| Worldwide Governance Indicators | https://www.worldbank.org/en/publication/worldwide-governance-indicators (the full Excel dataset, one sheet per dimension) | `data/raw/wgidata.xlsx` |

WDI indicators are **not** downloaded by hand: step 4 fetches them from the World Bank API
(`api.worldbank.org`) with the `wbgapi` package, so it needs internet access. If the API can't
be reached, the pipeline still finishes. It reuses the WDI data saved by the last successful
run, or else leaves the WDI columns blank and warns you. To change which indicators are
pulled, edit `CONFIG["wdi_indicators"]` in `src/config.py`.

The WDI series include `renewable_energy_consumption_pct` (`EG.FEC.RNEW.ZS`, renewable energy as
% of total final energy consumption), which ends in 2021 for most countries. The fossil fuel
share (`EG.USE.COMM.FO.ZS`) was not used: the API returns 0 for every country from 2016 on,
so it only has real values to 2014–15.

## 3. Run

```bash
pip install -r requirements.txt
python3 run_all.py
```

Each step can also be run on its own (e.g. `python3 src/step6_summary_stats.py`) once the
earlier steps have written their outputs.

## 4. What each step does

**Step 1 — PPI projects.** Keeps every original PPI column and adds a `project_id` (row number
in the raw export). `TotalInvestment` and `Capacity` are converted to numbers as
`investment_musd` (USD million) and `capacity`; `Not Available` / `Not Applicable` become blank.
Capacity units differ by project (`CapacityType`: MW, thousands of connections, km, ...), so
`capacity_mw` keeps only the MW figures. It drops:
- rows with no year or country
- the placeholder `**TEST**` record
- projects with a financial-closure year outside 1990–2025 (the export contains a few
  1900/1910/1970-era entries)
- projects whose `Technology` is **missing or only a placeholder** (`Not Applicable`, `N/A`,
  `Other`), since they can't be classified

It then adds:
- `technology_category`: `renewable`, `non_renewable`, or `mixed` (both kinds listed, e.g.
  `"Coal, Hydro, Large (>50MW)"`), matched on keywords in `CONFIG`. `Waste` counts as
  renewable, which is a judgment call you can change there. `is_renewable` (renewable or
  mixed) and `is_pure_renewable` (renewable only) are derived from it.
- sponsor columns, from `Sponsors Country`. Each sponsor's country is mapped to ISO3 and
  compared with the project's ISO3, giving `n_domestic_sponsors`, `n_foreign_sponsors`,
  `foreign_sponsor_share`, `has_domestic_sponsor`, `has_foreign_sponsor` and `sponsor_type`
  (`domestic_only` / `foreign_only` / `mixed` / `unknown`). Sponsors listed as `..` (country
  unknown) are counted in `n_unknown_country_sponsors` and left out of the comparison. A
  project where no sponsor country is known gets `sponsor_type = unknown` and blank flags.
  Hong Kong (HKG) counts as foreign for a project in China.

It also adds `derisk` = 1 if `MultiLateralSupport` or `BiLateralSupport` lists a support
type containing Guarantee, Risk Management or Insurance (e.g. `"IFC (Guarantee / $50 Million /
2010)"`), else 0; blank or `No` is 0. The match is on the support type, not the agency name,
so a loan from an agency like Export Finance and Insurance Corporation doesn't count.
`derisk_multilateral` and `derisk_bilateral` give the flag for each column separately.
`public_finance` is built the same way from Loan, Syndication or Equity (Equity also matches
`Quasi-equity`), with `public_finance_multilateral` and `public_finance_bilateral`.
`govt_derisk` = 1 if `InDirectGovtSupport` lists any support (payment, tariff, revenue, debt
or exchange-rate guarantees, tax deductions/credits); blank, `Not Applicable`, `No` or `N/A` is
0. `Not Available` is left **blank**: the field is mostly unrecorded outside 2010–2019, so
coding it 0 would mix "no support" with "not recorded".
`govt_finance` is built the same way from `DirectGovtSupport` (revenue subsidy, capital subsidy,
in-kind), with `Not Available` likewise left blank.
`any_support` = 1 if any of `derisk`, `public_finance`, `govt_derisk` or `govt_finance` is 1,
else 0. A blank government flag counts as not 1, so `any_support` is never blank.
`auction_deal` = 1 for a greenfield (`Type of PPI`) renewable project, other than large hydro,
with `AwardMethod` = Competitive bidding and `BidCriteria` = Lowest tariff, Lowest subsidy
required, Lowest cost of construction or operation, or Lowest government payments. A
greenfield renewable project (not large hydro) is left **blank** when its `AwardMethod` is
`Not Available`, or it was competitively bid but its `BidCriteria` is `Not Available`; every
other project, including brownfield and divestiture deals, is 0.

**Steps 2–4** reshape ND-GAIN, WGI (the governance *estimate*, about −2.5 to +2.5, for all
six dimensions) and WDI into country-year tables keyed on `iso3` + `year`.

**Step 5** left-joins them onto the projects. No project is dropped for missing ND-GAIN, WGI or
WDI values; those cells are simply left blank. Expect gaps where a source doesn't cover a
year: ND-GAIN starts in 1995, and WGI starts in 1996 and skips 1997, 1999 and 2001.
Output: `data/processed/project_panel.csv`.

**Step 6** writes descriptive statistics to `data/processed/summary_stats.xlsx` (one sheet per
table) and `summary_stats.txt`. The tables cover projects and investment by technology and
sponsor type, technology × sponsor crosstabs, projects by year, top countries, and
descriptives with missingness for every merged variable.

**Step 7** runs project-level fixed effects models (linear probability, SEs clustered by
country, pyfixest) on renewable projects: (1) `public_finance ~ derisk`; (2) the same with
bilateral support only; (3) multilateral support (any derisk or public finance) on bilateral
support; (4) `derisk` on having a foreign sponsor; (5) `derisk` on award method dummies
(reference: competitive bidding). Each is run with (a) `iso3^year + tech` fixed effects and
(b) `iso3 + year + tech` fixed effects plus ND-GAIN and GDP growth as controls. The controls
vary only by country-year, so (a) absorbs them. `tech` is the project's main renewable
technology (solar, wind, hydro_large, hydro_small, geothermal, biomass, biogas, waste, or
hybrid). Output: `data/processed/twfe_policy_stacking.txt`.

**Step 8** draws project-level summary figures for renewable projects into
`data/processed/figures/`: support by IDA status, by sponsor nationality and by technology;
multilateral vs bilateral support over time; policy stacking (raw conditional shares);
investment and capacity by type of support; and trends in public finance, derisking and foreign
sponsorship. `figure_data.xlsx` holds the numbers behind every figure.

**Step 9** collapses the project panel to a balanced country-year panel,
`data/processed/country_year_panel.csv`: every country with at least one PPI energy project
(117) for every year 1990–2024, i.e. 4,095 rows. Each row has project counts (all, renewable,
non-renewable, mixed), total investment and capacity (all and renewable), and counts of
projects with each support flag (all and renewable). Country-years with no project are 0, and
projects with unreported investment count as 0 in the totals (`n_investment_missing` counts
them). ND-GAIN, WGI and WDI are merged for every country-year, so they are only missing where
the source has no value (ND-GAIN before 1995; WGI before 1996 and in 1997, 1999, 2001).

Step 9 also builds:
- per country-year: `n_ren` (renewable deals), `n_ren_nd` (renewable deals without derisking),
  `first_ren_derisk` (1 if any renewable deal that year was derisked);
- fixed country dates (same in every year): `treat_year` (first derisked renewable deal),
  `entry_year` (first energy PPI deal), `first_any_derisk` (first derisked deal in any energy
  technology), `auction_year` (first `auction_deal`); blank if never. `entry_year` and
  `first_any_derisk` count all energy deals from 1990, including those step 1 drops for having
  no classifiable technology (transmission, distribution, gas);
- policy controls at t−1: `auction_active` (`auction_year` ≤ t−1), `fit_active` and `nm_active`
  (REN21 feed-in tariff / net metering enacted by t−1 and not removed; net billing counts as net
  metering), `prior_energy_derisk` (`first_any_derisk` < t). The REN21 sheet lists policies in
  force in 2024 and gives no removal years, so policies abolished earlier are not captured; add
  known removal years to `CONFIG["ren21_removed"]`;
- deal-flow history: `deals_l1_3` (renewable deals in t−1..t−3), `deals_l4_6` (t−4..t−6),
  `deals_growth` (the difference). Blank where the window reaches before 1990.

## 5. Country matching

Country names are matched to ISO3 codes by a manual dictionary first
(`common.MANUAL_ISO3_FIXES`), then by an exact `pycountry` lookup. pycountry's fuzzy search is
only a last resort, and every fuzzy match is printed so you can check it. Fuzzy search alone
produced errors in this data, such as `Niger` → Nigeria and `Kosovo` → Serbia. If a new PPI
export has names that don't match, the script prints them. Add them to the dictionary.

## 6. Before sharing with reviewers

- `data/` is git-ignored. Point reviewers to the download links above rather than
  redistributing the raw files, unless their licences allow it.
- Double check `data/processed/project_panel.csv` doesn't contain anything you don't want
  public before uploading it to a replication archive (e.g. Harvard Dataverse, OSF).
