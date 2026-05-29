#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cleans and combines external macro data into a single monthly CSV.

Each section reads one raw file, extracts the relevant series,
standardises the date format to month-start, and saves a preview.

Run this script each time new raw data is downloaded.

Outputs
-------
    data/macro_data.csv : combined monthly macro data, one row per month
"""

import os
import pandas as pd
import numpy as np

# ---------------------- paths ----------------------

HOME       = os.path.expanduser("~")
PAPER_PATH = os.path.join(HOME, "Documents/github/neoval-project")
RAW_PATH   = os.path.join(PAPER_PATH, "data", "raw")
DATA_PATH  = os.path.join(PAPER_PATH, "data")

# ---------------------- utilities ----------------------

def to_month_start(s):
    """Coerce a series of dates to month-start timestamps."""
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp()


def preview(df, name):
    """Print first and last 3 rows of a cleaned series."""
    print(f"\n--- {name} ---")
    print(pd.concat([df.head(3), df.tail(3)]).to_string())


# ---------------------- rba lending rate ----------------------

def clean_rba_lending_rate():
    """
    Extracts the standard variable mortgage rate (FILRHLBVS)
    from RBA Table F5.
    Monthly series, no resampling needed.

    Expected output:
        rows:       807
        date range: 1959-02-01 to 2026-04-01
        value range: ~5% (1959) to ~8.5% (2026)
    """
    path = os.path.join(RAW_PATH, "df_rba_lending_rates.xlsx")
    df   = pd.read_excel(path, sheet_name="Data", header=10)

    series_id = "FILRHLBVS"
    if series_id not in df.columns:
        raise ValueError(f"{series_id} not found. Available: {list(df.columns)}")

    out = pd.DataFrame({
        "time":          to_month_start(df.iloc[1:, 0]),
        "mortgage_rate": pd.to_numeric(df.iloc[1:][series_id], errors="coerce"),
    }).dropna().reset_index(drop=True)

    preview(out, "mortgage_rate")
    return out


# ---------------------- rba cash rate ----------------------

def clean_rba_cash_rate():
    """
    Extracts the cash rate target (ARBAMPCNCRT) from RBA Table A2.
    Event-based series resampled to monthly by forward filling.
    When multiple changes occur in one month, keeps the last value.

    Expected output:
        rows:       430
        date range: 1990-08-01 to 2026-05-01
        value range: ~4.35% (2026) to ~14% (1990)
    """
    path = os.path.join(RAW_PATH, "df_rba_cash_rate.xlsx")
    df   = pd.read_excel(path, sheet_name="Data", header=10)

    series_id = "ARBAMPCNCRT"
    if series_id not in df.columns:
        raise ValueError(f"{series_id} not found. Available: {list(df.columns)}")

    dates  = pd.to_datetime(df.iloc[1:, 0], errors="coerce")
    values = pd.to_numeric(df.iloc[1:][series_id], errors="coerce")

    raw = pd.Series(values.values, index=dates).dropna()
    raw.index = raw.index.to_period("M").to_timestamp()

    # keep last rate change per month to remove duplicates
    raw = raw.groupby(raw.index).last()

    # resample to monthly by forward filling last known rate
    monthly_idx = pd.date_range(
        start=raw.index.min(),
        end=raw.index.max(),
        freq="MS"
    )
    out = pd.DataFrame({
        "time":      monthly_idx,
        "cash_rate": raw.reindex(monthly_idx).ffill().values,
    }).dropna().reset_index(drop=True)

    preview(out, "cash_rate")
    return out


# ---------------------- rba bond yields ----------------------

def clean_rba_bond_yields():
    """
    Extracts the 10-year government bond yield (FCMYGBAG10D)
    from RBA Table F2.
    Daily series averaged to monthly.

    Expected output:
        rows:       157
        date range: 2013-05-01 to 2026-05-01
        value range: ~3.3% (2013) to ~5.0% (2026)
    """
    path = os.path.join(RAW_PATH, "df_rba_bond_yields.xlsx")
    df   = pd.read_excel(path, sheet_name="Data", header=10)

    series_id = "FCMYGBAG10D"
    if series_id not in df.columns:
        raise ValueError(f"{series_id} not found. Available: {list(df.columns)}")

    dates  = pd.to_datetime(df.iloc[1:, 0], errors="coerce")
    values = pd.to_numeric(df.iloc[1:][series_id], errors="coerce")

    raw = pd.Series(values.values, index=dates).dropna()

    # average daily values to monthly
    raw.index = raw.index.to_period("M").to_timestamp()
    monthly   = raw.groupby(raw.index).mean()

    out = pd.DataFrame({
        "time":           monthly.index,
        "bond_yield_10y": monthly.values,
    }).reset_index(drop=True)

    preview(out, "bond_yield_10y")
    return out


# ---------------------- rba housing credit ----------------------

def clean_rba_housing_credit():
    """
    Extracts housing credit outstanding (DLCACOHS), seasonally adjusted,
    from RBA Table D2. Converts level to monthly percentage change.
    Monthly series, no resampling needed.

    Expected output:
        rows:       434
        date range: 1990-02-01 to 2026-03-01
        value range: ~0.4% to ~1.0% monthly growth
    """
    path = os.path.join(RAW_PATH, "df_rba_credit_aggregates.xlsx")
    df   = pd.read_excel(path, sheet_name="Data", header=10)

    series_id = "DLCACOHS"
    if series_id not in df.columns:
        raise ValueError(f"{series_id} not found. Available: {list(df.columns)}")

    out = pd.DataFrame({
        "time":                  to_month_start(df.iloc[1:, 0]),
        "housing_credit_growth": pd.to_numeric(df.iloc[1:][series_id], errors="coerce"),
    }).dropna().reset_index(drop=True)

    # convert level to monthly percentage change
    out["housing_credit_growth"] = out["housing_credit_growth"].pct_change() * 100
    out = out.dropna().reset_index(drop=True)

    preview(out, "housing_credit_growth")
    return out


# ---------------------- rba housing lending rate ----------------------

def clean_rba_housing_lending_rate():
    """
    Extracts owner occupier variable rate outstanding loans (FLRHOOVL)
    from RBA Table F6.
    Monthly series, no resampling needed.

    Expected output:
        rows:       80
        date range: 2019-08-01 to 2026-03-01
        value range: ~3.6% (2019) to ~6.0% (2026)
    """
    path = os.path.join(RAW_PATH, "df_rba_housing_lending_rates.xlsx")
    df   = pd.read_excel(path, sheet_name="Data", header=10)

    series_id = "FLRHOOVL"
    if series_id not in df.columns:
        raise ValueError(f"{series_id} not found. Available: {list(df.columns)}")

    out = pd.DataFrame({
        "time":                      to_month_start(df.iloc[1:, 0]),
        "housing_lending_rate":      pd.to_numeric(df.iloc[1:][series_id], errors="coerce"),
    }).dropna().reset_index(drop=True)

    preview(out, "housing_lending_rate")
    return out


# ---------------------- abs unemployment ----------------------

def clean_abs_unemployment():
    """
    Extracts unemployment rate seasonally adjusted persons (A84423050A)
    from ABS Labour Force Table 12, sheet Data2.
    Monthly series, no resampling needed.

    Expected output:
        rows:       578
        date range: 1978-02-01 to 2026-03-01
        value range: ~4% (2026) to ~11% (1993 peak)
    """
    path = os.path.join(RAW_PATH, "df_abs_unemployment_rate.xlsx")
    df   = pd.read_excel(path, sheet_name="Data2", header=9)

    series_id = "A84423050A"
    if series_id not in df.columns:
        raise ValueError(f"{series_id} not found. Available: {list(df.columns)}")

    out = pd.DataFrame({
        "time":              to_month_start(df["Series ID"]),
        "unemployment_rate": pd.to_numeric(df[series_id], errors="coerce"),
    }).dropna().reset_index(drop=True)

    preview(out, "unemployment_rate")
    return out


# ---------------------- abs trimmed mean inflation ----------------------

def clean_abs_trimmed_mean_inflation():
    """
    Extracts year-ended trimmed mean inflation (GCPIOCPMTMYP) from RBA G1.
    Quarterly series resampled to monthly by forward filling.

    Expected output:
        rows:       ~276 (Q2 2002 to Q1 2026, monthly)
        date range: 2002-06-01 to 2026-03-01
        value range: ~1% (2020) to ~8% (2022)
    """
    path = os.path.join(RAW_PATH, "df_abs_mean_inflation.xlsx")
    df   = pd.read_excel(path, sheet_name="Data", header=10)

    series_id = "GCPIOCPMTMYP"
    if series_id not in df.columns:
        raise ValueError(f"{series_id} not found. Available: {list(df.columns)}")

    dates  = pd.to_datetime(df["Series ID"], errors="coerce")
    values = pd.to_numeric(df[series_id], errors="coerce")

    raw = pd.Series(values.values, index=dates).dropna()
    raw.index = raw.index.to_period("M").to_timestamp()

    monthly_idx = pd.date_range(start=raw.index.min(), end=raw.index.max(), freq="MS")
    out = pd.DataFrame({
        "time":                   monthly_idx,
        "trimmed_mean_inflation": raw.reindex(monthly_idx).ffill().values,
    }).dropna().reset_index(drop=True)

    preview(out, "trimmed_mean_inflation")
    return out


# ---------------------- abs household disposable income ----------------------

def clean_abs_household_income():
    """
    Extracts gross household disposable income, seasonally adjusted (A2302939L),
    from ABS National Accounts (5206.0). Quarterly series resampled to monthly
    by forward filling.

    Expected output:
        rows:       ~798 (Q3 1959 to Q4 2025, monthly)
        date range: 1959-09-01 to 2025-12-01
        value range: ~$2B (1959) to ~$500B+ (2025)
    """
    path = os.path.join(RAW_PATH, "df_abs_household_income.xlsx")
    df   = pd.read_excel(path, sheet_name="Data1", header=9)

    series_id = "A2302939L"
    if series_id not in df.columns:
        raise ValueError(f"{series_id} not found. Available: {list(df.columns)}")

    dates  = pd.to_datetime(df["Series ID"], errors="coerce")
    values = pd.to_numeric(df[series_id], errors="coerce")

    raw = pd.Series(values.values, index=dates).dropna()
    raw.index = raw.index.to_period("M").to_timestamp()

    monthly_idx = pd.date_range(start=raw.index.min(), end=raw.index.max(), freq="MS")
    out = pd.DataFrame({
        "time":                        monthly_idx,
        "household_disposable_income": raw.reindex(monthly_idx).ffill().values,
    }).dropna().reset_index(drop=True)

    preview(out, "household_disposable_income")
    return out


# ---------------------- abs net overseas migration ----------------------

def clean_abs_net_overseas_migration():
    """
    Extracts net overseas migration (A2133254C) from ABS Demographic Statistics
    (3101.0). Quarterly series resampled to monthly by forward filling.

    Expected output:
        rows:       ~534 (Q2 1981 to Q3 2025, monthly)
        date range: 1981-06-01 to 2025-09-01
        value range: ~-10k (COVID 2020) to ~170k (2023 peak) per quarter
    """
    path = os.path.join(RAW_PATH, "df_abs_population.xlsx")
    df   = pd.read_excel(path, sheet_name="Data1", header=9)

    series_id = "A2133254C"
    if series_id not in df.columns:
        raise ValueError(f"{series_id} not found. Available: {list(df.columns)}")

    dates  = pd.to_datetime(df["Series ID"], errors="coerce")
    values = pd.to_numeric(df[series_id], errors="coerce")

    raw = pd.Series(values.values, index=dates).dropna()
    raw.index = raw.index.to_period("M").to_timestamp()

    monthly_idx = pd.date_range(start=raw.index.min(), end=raw.index.max(), freq="MS")
    out = pd.DataFrame({
        "time":                   monthly_idx,
        "net_overseas_migration": raw.reindex(monthly_idx).ffill().values,
    }).dropna().reset_index(drop=True)

    preview(out, "net_overseas_migration")
    return out


# ---------------------- abs building approvals ----------------------

def clean_abs_building_approvals():
    """
    Extracts total dwelling approvals, all sectors, seasonally adjusted (A422070J)
    from ABS Building Approvals (8731.0). Monthly series, no resampling needed.

    Expected output:
        rows:       ~513 (Jul 1983 to Mar 2026)
        date range: 1983-07-01 to 2026-03-01
        value range: ~8k to ~23k dwellings per month
    """
    path = os.path.join(RAW_PATH, "df_abs_building_approvals.xlsx")
    df   = pd.read_excel(path, sheet_name="Data1", header=9)

    series_id = "A422070J"
    if series_id not in df.columns:
        raise ValueError(f"{series_id} not found. Available: {list(df.columns)}")

    out = pd.DataFrame({
        "time":               to_month_start(df["Series ID"]),
        "building_approvals": pd.to_numeric(df[series_id], errors="coerce"),
    }).dropna().reset_index(drop=True)

    preview(out, "building_approvals")
    return out


# ---------------------- abs dwelling commencements ----------------------

def clean_abs_dwelling_commencements():
    """
    Extracts dwelling commencements, total sectors, seasonally adjusted (A83801544L)
    from ABS Building Activity (8752.0). Quarterly series resampled to monthly
    by forward filling.

    Expected output:
        rows:       ~726 (Q3 1965 to Q4 2025, monthly)
        date range: 1965-09-01 to 2025-12-01
        value range: ~15k to ~58k dwellings per quarter
    """
    path = os.path.join(RAW_PATH, "df_abs_dwelling_commencements.xlsx")
    df   = pd.read_excel(path, sheet_name="Data1", header=9)

    series_id = "A83801544L"
    if series_id not in df.columns:
        raise ValueError(f"{series_id} not found. Available: {list(df.columns)}")

    dates  = pd.to_datetime(df["Series ID"], errors="coerce")
    values = pd.to_numeric(df[series_id], errors="coerce")

    raw = pd.Series(values.values, index=dates).dropna()
    raw.index = raw.index.to_period("M").to_timestamp()

    monthly_idx = pd.date_range(start=raw.index.min(), end=raw.index.max(), freq="MS")
    out = pd.DataFrame({
        "time":                   monthly_idx,
        "dwelling_commencements": raw.reindex(monthly_idx).ffill().values,
    }).dropna().reset_index(drop=True)

    preview(out, "dwelling_commencements")
    return out


# ---------------------- abs cpi rents ----------------------

def clean_abs_cpi_rents():
    """
    Extracts CPI rents index, seasonally adjusted (A130400542R) from ABS CPI
    (6401.0) selected series. Monthly series available from July 2022 only.

    Expected output:
        rows:       ~45 (Jul 2022 to Mar 2026)
        date range: 2022-07-01 to 2026-03-01
        value range: index numbers ~90 to ~115
    """
    path = os.path.join(RAW_PATH, "df_abs_cpi_selected.xlsx")
    df   = pd.read_excel(path, sheet_name="Data1", header=9)

    series_id = "A130400542R"
    if series_id not in df.columns:
        raise ValueError(f"{series_id} not found. Available: {list(df.columns)}")

    out = pd.DataFrame({
        "time":      to_month_start(df["Series ID"]),
        "cpi_rents": pd.to_numeric(df[series_id], errors="coerce"),
    }).dropna().reset_index(drop=True)

    preview(out, "cpi_rents")
    return out


# ---------------------- main ----------------------

def main():
    mortgage       = clean_rba_lending_rate()
    cash_rate      = clean_rba_cash_rate()
    bond_yields    = clean_rba_bond_yields()
    housing_credit = clean_rba_housing_credit()
    housing_rate   = clean_rba_housing_lending_rate()
    unemployment   = clean_abs_unemployment()
    trimmed_mean   = clean_abs_trimmed_mean_inflation()
    hh_income      = clean_abs_household_income()
    migration      = clean_abs_net_overseas_migration()
    building_app   = clean_abs_building_approvals()
    dwelling_comm  = clean_abs_dwelling_commencements()
    cpi_rents      = clean_abs_cpi_rents()

    print("\n[done] all series cleaned")
    for name, df in [
        ("mortgage_rate",               mortgage),
        ("cash_rate",                   cash_rate),
        ("bond_yield_10y",              bond_yields),
        ("housing_credit_growth",       housing_credit),
        ("housing_lending_rate",        housing_rate),
        ("unemployment_rate",           unemployment),
        ("trimmed_mean_inflation",      trimmed_mean),
        ("household_disposable_income", hh_income),
        ("net_overseas_migration",      migration),
        ("building_approvals",          building_app),
        ("dwelling_commencements",      dwelling_comm),
        ("cpi_rents",                   cpi_rents),
    ]:
        print(f"  {name}: {len(df)} rows, {df.time.min().date()} to {df.time.max().date()}")

    from functools import reduce
    dfs = [
        mortgage, cash_rate, bond_yields, housing_credit, housing_rate,
        unemployment, trimmed_mean, hh_income, migration, building_app,
        dwelling_comm, cpi_rents,
    ]
    combined = reduce(lambda left, right: pd.merge(left, right, on="time", how="outer"), dfs)
    combined = combined.sort_values("time").reset_index(drop=True)

    out_path = os.path.join(DATA_PATH, "macro_data.csv")
    combined.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\n[ok] saved preview to {out_path}")


if __name__ == "__main__":
    main()

