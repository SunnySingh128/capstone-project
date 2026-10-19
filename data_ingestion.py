"""
Data Ingestion Module for Subscription Revenue Forecasting Service.

Ingests invoice data from CSV, JSON, and JSONL formats, validates schema and values,
aggregates revenue at daily or monthly frequency, fills missing dates with zeros
over the global observed date range, and creates per-country and 'all' aggregate series.
"""

import argparse
import glob
import json
import logging
import os
from typing import List, Optional, Tuple, Union

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Common column name heuristics for automatic resolution
DEFAULT_DATE_CANDIDATES = ["date", "invoice_date", "invoicedate", "timestamp", "created_at"]
DEFAULT_COUNTRY_CANDIDATES = ["country", "billing_country", "billingcountry", "nation", "region"]
DEFAULT_REVENUE_CANDIDATES = ["price", "revenue", "amount", "total", "signed_revenue", "invoice_amount"]


def _detect_column(columns: List[str], candidates: List[str], explicit_name: Optional[str] = None) -> str:
    """Find the column matching explicit preference or candidates."""
    if explicit_name:
        if explicit_name in columns:
            return explicit_name
        # Case-insensitive match check
        for col in columns:
            if col.lower() == explicit_name.lower():
                return col
        raise ValueError(f"Specified column '{explicit_name}' not found in data columns: {columns}")

    for candidate in candidates:
        for col in columns:
            if col.lower() == candidate.lower():
                return col

    raise ValueError(f"Could not automatically detect column from candidates {candidates}. Available: {columns}")


def load_raw_file(file_path: str) -> pd.DataFrame:
    """Load a single CSV, JSON, or JSONL file into a pandas DataFrame."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext == ".json":
        try:
            return pd.read_json(file_path)
        except ValueError:
            # Try lines=True for multiline JSON
            return pd.read_json(file_path, lines=True)
    elif ext in [".jsonl", ".ndjson"]:
        return pd.read_json(file_path, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def load_raw_data(input_path: str) -> pd.DataFrame:
    """
    Load data from a file or directory of CSV, JSON, and JSONL files.
    """
    if os.path.isfile(input_path):
        return load_raw_file(input_path)

    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    files: List[str] = []
    for pattern in ["*.csv", "*.json", "*.jsonl", "*.ndjson"]:
        files.extend(glob.glob(os.path.join(input_path, pattern)))

    if not files:
        raise FileNotFoundError(f"No CSV, JSON, or JSONL files found in directory: {input_path}")

    dfs = [load_raw_file(f) for f in sorted(files)]
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d records across %d files from %s", len(combined_df), len(files), input_path)
    return combined_df


def validate_and_clean_data(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    country_col: Optional[str] = None,
    revenue_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, str, str]:
    """
    Validate dataframe columns and clean records.
    Returns cleaned dataframe with standardized 'date', 'country', 'revenue' columns.
    """
    if df.empty:
        raise ValueError("Input data is empty.")

    actual_date_col = _detect_column(list(df.columns), DEFAULT_DATE_CANDIDATES, date_col)
    actual_country_col = _detect_column(list(df.columns), DEFAULT_COUNTRY_CANDIDATES, country_col)
    actual_rev_col = _detect_column(list(df.columns), DEFAULT_REVENUE_CANDIDATES, revenue_col)

    initial_len = len(df)
    clean_df = df.copy()

    # Drop rows where date, country or revenue is null
    clean_df = clean_df.dropna(subset=[actual_date_col, actual_country_col, actual_rev_col])

    # Convert date column to datetime
    clean_df["clean_date"] = pd.to_datetime(clean_df[actual_date_col], errors="coerce")
    # Convert revenue to numeric
    clean_df["clean_revenue"] = pd.to_numeric(clean_df[actual_rev_col], errors="coerce")
    # Clean country strings
    clean_df["clean_country"] = clean_df[actual_country_col].astype(str).str.strip()

    # Drop invalid dates and revenues
    clean_df = clean_df.dropna(subset=["clean_date", "clean_revenue"])
    clean_df = clean_df[~clean_df["clean_country"].str.lower().isin(["", "nan", "none", "null"])]

    dropped = initial_len - len(clean_df)
    if dropped > 0:
        logger.warning("Dropped %d invalid or missing rows during validation", dropped)

    if clean_df.empty:
        raise ValueError("All records were invalid or dropped during validation.")

    return clean_df, actual_date_col, actual_country_col, actual_rev_col


def aggregate_revenue(
    df: pd.DataFrame,
    frequency: str = "monthly",
) -> pd.DataFrame:
    """
    Aggregate revenue per country and create 'all' aggregate.
    Fills missing periods within the observed global date range with zero.
    """
    clean_df = df.copy()

    if frequency == "monthly":
        clean_df["period"] = clean_df["clean_date"].dt.to_period("M").dt.to_timestamp()
        freq_alias = "MS"
        date_format = "%Y-%m-01"
    elif frequency == "daily":
        clean_df["period"] = clean_df["clean_date"].dt.floor("D")
        freq_alias = "D"
        date_format = "%Y-%m-%d"
    else:
        raise ValueError(f"Unsupported frequency: {frequency}. Choose 'monthly' or 'daily'.")

    # Group by period and country
    grouped = (
        clean_df.groupby(["period", "clean_country"])["clean_revenue"]
        .sum()
        .reset_index()
        .rename(columns={"clean_country": "country", "clean_revenue": "revenue"})
    )

    # Global date range
    global_min = clean_df["period"].min()
    global_max = clean_df["period"].max()
    full_dates = pd.date_range(start=global_min, end=global_max, freq=freq_alias)

    unique_countries = sorted(grouped["country"].unique())
    country_dfs = []

    for country in unique_countries:
        country_subset = grouped[grouped["country"] == country]
        # Reindex revenue series with full date range, filling missing periods with 0.0
        rev_series = country_subset.set_index("period")["revenue"].reindex(full_dates, fill_value=0.0)
        c_df = pd.DataFrame({
            "date": full_dates.strftime(date_format),
            "country": country,
            "revenue": rev_series.values,
        })
        country_dfs.append(c_df)

    # All aggregate series (sum of all countries)
    all_rev_series = (
        grouped.groupby("period")["revenue"]
        .sum()
        .reindex(full_dates, fill_value=0.0)
    )
    all_df = pd.DataFrame({
        "date": full_dates.strftime(date_format),
        "country": "all",
        "revenue": all_rev_series.values,
    })

    combined_result = pd.concat(country_dfs + [all_df], ignore_index=True)
    combined_result = combined_result[["date", "country", "revenue"]].sort_values(
        by=["country", "date"]
    ).reset_index(drop=True)

    # Round revenue to 2 decimal places for financial reporting
    combined_result["revenue"] = combined_result["revenue"].round(2)
    return combined_result


def ingest_data(
    input_dir: str = "data/raw",
    output_dir: Optional[str] = "data/processed",
    frequency: str = "monthly",
    date_col: Optional[str] = None,
    country_col: Optional[str] = None,
    revenue_col: Optional[str] = None,
    output_file: str = "revenue_timeseries.csv",
) -> pd.DataFrame:
    """
    Complete end-to-end ingestion pipeline.
    """
    logger.info("Starting ingestion from %s (frequency=%s)...", input_dir, frequency)
    raw_df = load_raw_data(input_dir)
    cleaned_df, _, _, _ = validate_and_clean_data(raw_df, date_col, country_col, revenue_col)
    aggregated_df = aggregate_revenue(cleaned_df, frequency=frequency)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        dest_path = os.path.join(output_dir, output_file)
        aggregated_df.to_csv(dest_path, index=False)
        logger.info("Saved aggregated time-series (%d rows) to %s", len(aggregated_df), dest_path)

    return aggregated_df


def main():
    parser = argparse.ArgumentParser(description="Ingest and aggregate subscription invoice data.")
    parser.add_argument("--input-dir", type=str, default="data/raw", help="Path to input file or directory")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Path to output directory")
    parser.add_argument("--output-file", type=str, default="revenue_timeseries.csv", help="Output CSV filename")
    parser.add_argument("--frequency", type=str, default="monthly", choices=["monthly", "daily"], help="Aggregation frequency")
    parser.add_argument("--date-column", type=str, default=None, help="Name of date column")
    parser.add_argument("--country-column", type=str, default=None, help="Name of country column")
    parser.add_argument("--revenue-column", type=str, default=None, help="Name of revenue column")

    args = parser.parse_args()

    ingest_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        frequency=args.frequency,
        date_col=args.date_column,
        country_col=args.country_column,
        revenue_col=args.revenue_column,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
