# PPI Renewable Energy Panel — Replication Code

Builds a **project-country-year panel**: one row per World Bank PPI energy project, tagged
with its country (ISO3) and financial-closure year, with country-year data from **ND-GAIN**,
the **Worldwide Governance Indicators (WGI)** and **World Bank WDI** merged on.

A later Stage 2 script (not yet built) will collapse `project_panel.csv` to one row per
country-year for the econometric analysis.

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
│   └── step6_summary_stats.py  <- descriptive statistics
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
| Worldwide Governance Indicators | https://www.worldbank.org/en/publication/worldwide-governance-indicators (the full Excel dataset, one sheet per dimension) | `data/raw/wgidata.xlsx` |

WDI indicators are **not** downloaded by hand: step 4 fetches them from the World Bank API
(`api.worldbank.org`) with the `wbgapi` package, so it needs internet access. If the API can't
be reached, the pipeline still finishes. It reuses the WDI data saved by the last successful
run, or else leaves the WDI columns blank and warns you. To change which indicators are
pulled, edit `CONFIG["wdi_indicators"]` in `src/config.py`.

## 3. Run

```bash
pip install -r requirements.txt
python3 run_all.py
```

Each step can also be run on its own (e.g. `python3 src/step6_summary_stats.py`) once the
earlier steps have written their outputs.

## 4. What each step does

**Step 1 — PPI projects.** Keeps every original PPI column and adds a `project_id` (row number
in the raw export). It drops:
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
