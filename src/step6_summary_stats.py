"""
step6_summary_stats.py
----------------------
Descriptive statistics for the final project panel. Prints them and saves them to
data/processed/summary_stats.xlsx (one sheet per table) and summary_stats.txt.

Run from the project root:  python3 src/step6_summary_stats.py
"""

import sys

import pandas as pd

from config import CONFIG, PROJECT_PANEL, SUMMARY_XLSX, SUMMARY_TXT


def build_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = {}

    tables["overview"] = pd.DataFrame({"value": {
        "Projects": len(df),
        "Countries": df["iso3"].nunique(),
        "First year": df["year"].min(),
        "Last year": df["year"].max(),
        "Total investment (USD m)": round(df["investment_musd"].sum(skipna=True), 1),
        "Projects with missing investment": int(df["investment_musd"].isna().sum()),
    }})

    by_tech = df.groupby("technology_category").agg(
        n_projects=("project_id", "size"),
        total_investment_musd=("investment_musd", "sum"),
        mean_investment_musd=("investment_musd", "mean"),
        n_countries=("iso3", "nunique"),
    )
    by_tech.insert(1, "pct_projects", 100 * by_tech["n_projects"] / len(df))
    tables["by_technology"] = by_tech.round(2)

    by_sponsor = df.groupby("sponsor_type").agg(
        n_projects=("project_id", "size"),
        total_investment_musd=("investment_musd", "sum"),
        pct_renewable=("is_pure_renewable", "mean"),
    )
    by_sponsor["pct_renewable"] *= 100
    by_sponsor.insert(1, "pct_projects", 100 * by_sponsor["n_projects"] / len(df))
    tables["by_sponsor_type"] = by_sponsor.round(2)

    tables["tech_x_sponsor_counts"] = pd.crosstab(
        df["technology_category"], df["sponsor_type"], margins=True, margins_name="total")
    tables["tech_x_sponsor_row_pct"] = (100 * pd.crosstab(
        df["technology_category"], df["sponsor_type"], normalize="index")).round(1)

    tables["tech_x_derisk_counts"] = pd.crosstab(
        df["technology_category"], df["derisk"], margins=True, margins_name="total")
    tables["derisk_by_sponsor_type"] = (df.groupby("sponsor_type")
                                        .agg(n_projects=("project_id", "size"),
                                             pct_derisked=("derisk", "mean")))
    tables["derisk_by_sponsor_type"]["pct_derisked"] = (
        100 * tables["derisk_by_sponsor_type"]["pct_derisked"]).round(1)

    tables["tech_x_public_finance_counts"] = pd.crosstab(
        df["technology_category"], df["public_finance"], margins=True, margins_name="total")
    tables["derisk_x_public_finance"] = pd.crosstab(
        df["derisk"], df["public_finance"], margins=True, margins_name="total")

    # crosstab drops missing keys, so show a blank government flag as its own category
    def with_blank(col):
        return df[col].map({0: "0", 1: "1"}).fillna("not available").rename(col)

    govt_derisk, govt_finance = with_blank("govt_derisk"), with_blank("govt_finance")
    tables["tech_x_govt_derisk_counts"] = pd.crosstab(
        df["technology_category"], govt_derisk, margins=True, margins_name="total")
    tables["derisk_x_govt_derisk"] = pd.crosstab(
        df["derisk"], govt_derisk, margins=True, margins_name="total")
    tables["tech_x_govt_finance_counts"] = pd.crosstab(
        df["technology_category"], govt_finance, margins=True, margins_name="total")
    tables["govt_derisk_x_govt_finance"] = pd.crosstab(
        govt_derisk, govt_finance, margins=True, margins_name="total")

    tables["projects_by_year"] = pd.crosstab(
        df["year"], df["technology_category"], margins=True, margins_name="total")

    tables["top_countries"] = (df.groupby(["iso3", "country_name"])
                               .agg(n_projects=("project_id", "size"),
                                    n_renewable=("is_pure_renewable", "sum"),
                                    total_investment_musd=("investment_musd", "sum"))
                               .sort_values("n_projects", ascending=False).head(20).round(1))

    numeric = ["investment_musd", "n_sponsors", "foreign_sponsor_share", "derisk", "public_finance", "govt_derisk",
               "govt_finance",
               "nd_gain_value",
               *CONFIG["wgi_sheet_indicators"].values(), *CONFIG["wdi_indicators"].values()]
    desc = df[numeric].apply(pd.to_numeric, errors="coerce").describe().T
    desc.insert(0, "n_missing", df[numeric].isna().sum())
    desc.insert(1, "pct_missing", 100 * df[numeric].isna().mean())
    tables["descriptives"] = desc.round(3)

    return tables


def main():
    print("=" * 70)
    print("Step 6: summary statistics")
    print("=" * 70)
    if not PROJECT_PANEL.exists():
        sys.exit(f"{PROJECT_PANEL} not found — run step5_merge_panel.py first.")
    df = pd.read_csv(PROJECT_PANEL, keep_default_na=False, na_values=[""], low_memory=False)
    df["is_pure_renewable"] = df["is_pure_renewable"].astype(str) == "True"

    tables = build_tables(df)

    with pd.ExcelWriter(SUMMARY_XLSX) as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name[:31])

    lines = []
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.max_rows", 200):
        for name, table in tables.items():
            lines += ["", "-" * 70, name.replace("_", " ").upper(), "-" * 70, table.to_string()]
    report = "\n".join(lines)
    SUMMARY_TXT.write_text(report + "\n")
    print(report)
    print(f"\nSaved summary tables to {SUMMARY_XLSX} and {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
