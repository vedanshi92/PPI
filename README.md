# PPI Renewable Energy Panel — Replication Code

This builds a panel dataset in **two stages**:

**Stage 1 — Project-level panel** (`src/build_project_panel.py`)
Loads the raw World Bank PPI export (one row per project) and, without collapsing anything:
- Classifies each project's `Technology` as **renewable / non_renewable / mixed**, and
  drops projects whose Technology is missing or only a placeholder (`Not Applicable`, `N/A`, `Other`)
- Generates a **domestic/international sponsor indicator** from the `Sponsors Country`
  field, comparing each sponsor's country to the project's own country
- Merges in **ND-GAIN**, **World Governance Indicators (WGI)**, and **World Bank WDI
  macro indicators** (GDP, inflation, exchange rate, etc. — pulled live from the API) —
  a many-to-one merge, since many projects share the same country-year
- Prints descriptive statistics
- Saves `data/processed/project_panel.csv`

**Stage 2 — Country-year aggregation** (a later script, not yet built)
Will take `project_panel.csv` as its input and collapse it to one row per country-year
for the actual econometric analysis.

Shared loading/harmonization logic (country-code matching, ND-GAIN/WGI/WDI loaders)
lives in `src/common.py` so both stages can reuse it without duplicating code.

## 1. Folder structure

```
ppi-renewable-panel/
├── data/
│   ├── raw/              <- put your downloaded files here (see step 2)
│   └── processed/        <- scripts write output here
├── src/
│   ├── common.py          <- shared loaders, imported by both stages
│   └── build_project_panel.py  <- Stage 1 (this is what to run first)
├── requirements.txt
└── README.md
```

## 2. Download the raw data

Download each dataset manually and place it in `data/raw/` with the filenames below
(rename after downloading if needed):

| Dataset | Where to get it | Save as |
|---|---|---|
| World Bank PPI database | https://ppi.worldbankgroup.org/en/ppi -> "Custom query / download full dataset" (Excel export) | `data/raw/ppi.xlsx` |
| ND-GAIN Country Index | https://gain.nd.edu/our-work/country-index/download-data/ (download the full country-level time series, not just current-year rankings) | `data/raw/nd_gain.csv` |
| World Governance Indicators (WGI) | https://info.worldbank.org/governance/wgi/ , or search "Worldwide Governance Indicators" on https://databank.worldbank.org and export in the same wide CSV format as a WDI export | `data/raw/wgi_data.csv` |

WDI macro indicators (GDP, inflation, exchange rate, etc.) are **not** downloaded manually —
the script fetches them live from the World Bank API via the `wbgapi` package. To change which
indicators are pulled, edit the `CONFIG["wdi_indicators"]` dictionary at the top of
`src/build_project_panel.py`; look up series codes at https://databank.worldbank.org/metadataglossary.

**Important:** the exact column names in these exports can vary slightly by download method/date.
Before running the script, open each file and skim the first few rows — the `CONFIG` section at
the top of `src/build_project_panel.py` is where you tell the script what your actual column
names are. The script fails loudly with a clear message if a column it expects isn't found,
rather than silently producing wrong output.

## 3. Configure

Open `src/build_project_panel.py` and check the `CONFIG` dictionary — it's pre-filled to match
the actual PPI export column names (`Country`, `Financial closure year`, `TotalInvestment`,
`Technology`, `Sponsors Country`, etc.). If your own export has slightly different column names,
update them here.

## 4. Run Stage 1

```bash
python3 src/build_project_panel.py
```

This writes `data/processed/project_panel.csv` — one row per PPI project, with every original
column preserved, plus:
- `iso3` — harmonized country code
- `technology_category` — `renewable`, `non_renewable`, or `mixed` (both kinds listed, e.g.
  `"Coal, Hydro, Large (>50MW)"`), built from the `Technology` field using the keyword lists in
  `CONFIG`. `Waste` is counted as renewable — a judgment call you can change there.
- `is_renewable` (renewable or mixed), `is_pure_renewable` (renewable only),
  `n_renewable_keywords_matched` — convenience flags derived from the same classification
- `sponsor_countries`, `sponsor_type`, `has_domestic_sponsor`, `has_international_sponsor` —
  built from the `Sponsors Country` field, matched via ISO3 codes rather than raw string
  comparison so naming variants don't cause misclassification
- ND-GAIN, WGI, and WDI columns merged in by country-year

It also prints merge diagnostics and descriptive statistics (renewable share, sponsor-type
breakdown, total investment captured, etc.) to the terminal — worth reading closely the first
time you run it, since it also flags data-quality issues (e.g. unmatched country names,
out-of-range years).

## 5. Before sharing with reviewers

- Delete or `.gitignore` the raw files in `data/raw/` if their license doesn't permit
  redistribution — instead point reviewers to this README's download links.
- Double check `data/processed/project_panel.csv` doesn't contain anything you don't want
  public before uploading it to a replication archive (e.g. Harvard Dataverse, OSF).
