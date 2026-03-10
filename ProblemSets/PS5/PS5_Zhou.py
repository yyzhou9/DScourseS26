"""
PS5_Zhou.py
Econ 5253 - Spring 2026
Tasks:
  1. Web scraping: BLS Mass Layoffs Statistics table (no API)
  2. API: BLS Public Data API v2 - unemployment + layoff series
"""

import requests
import json
import pandas as pd
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────
# TASK 1: Web Scraping (no API)
# Source: BLS Extended Mass Layoffs release table
# URL: https://www.bls.gov/news.release/mslo.t01.htm
# ─────────────────────────────────────────────

def scrape_bls_mass_layoffs():
    url = "https://www.bls.gov/news.release/mslo.t01.htm"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; academic research)"}
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # The main data table has class "regular" on BLS release pages
    table = soup.find("table", {"class": "regular"})
    if table is None:
        table = soup.find("table")

    rows = []
    headers_row = []

    for i, tr in enumerate(table.find_all("tr")):
        cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
        if i == 0:
            headers_row = cells
        elif cells:
            rows.append(cells)

    df = pd.DataFrame(rows, columns=headers_row if headers_row else None)
    df.columns = [str(c).strip() for c in df.columns]
    return df


print("=" * 60)
print("TASK 1: Web Scraping — BLS Mass Layoffs Statistics")
print("=" * 60)

try:
    df_layoffs = scrape_bls_mass_layoffs()
    print(df_layoffs.to_string(index=False))
    df_layoffs.to_csv("bls_mass_layoffs_scraped.csv", index=False)
    print("\nSaved to bls_mass_layoffs_scraped.csv")
except Exception as e:
    print(f"Scraping error: {e}")


# ─────────────────────────────────────────────
# TASK 2: BLS Public Data API v2
# Series used:
#   LNS14000000 — Unemployment Rate (seasonally adjusted)
#   JTS000000000000000LDR — Layoffs & Discharges Rate, Total Nonfarm
# No registration key needed for public access (limited to 25 yrs/10 series)
# ─────────────────────────────────────────────

def fetch_bls_api(series_ids, start_year, end_year):
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    payload = {
        "seriesid": series_ids,
        "startyear": str(start_year),
        "endyear": str(end_year),
        "calculations": True,
        "annualaverage": False,
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def parse_bls_response(data):
    records = []
    for series in data.get("Results", {}).get("series", []):
        sid = series["seriesID"]
        for obs in series.get("data", []):
            records.append({
                "series_id": sid,
                "year": int(obs["year"]),
                "period": obs["period"],
                "period_name": obs["periodName"],
                "value": float(obs["value"]),
            })
    df = pd.DataFrame(records)
    df = df[~df["period"].str.startswith("A")]   # drop annual averages
    df["month"] = df["period"].str.replace("M", "").astype(int)
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    df = df.sort_values(["series_id", "date"]).reset_index(drop=True)
    return df


print("\n" + "=" * 60)
print("TASK 2: BLS API — Unemployment Rate & Layoffs Rate (2000-2024)")
print("=" * 60)

SERIES = [
    "LNS14000000",          # Unemployment rate
    "JTS000000000000000LDR" # Layoffs & discharges rate, total nonfarm
]

try:
    raw = fetch_bls_api(SERIES, start_year=2000, end_year=2024)
    if raw.get("status") != "REQUEST_SUCCEEDED":
        print("API message:", raw.get("message"))
    df_api = parse_bls_response(raw)

    # Summary table: annual averages
    df_annual = (
        df_api.groupby(["series_id", "year"])["value"]
        .mean()
        .round(2)
        .reset_index()
    )

    label_map = {
        "LNS14000000": "Unemployment Rate (%)",
        "JTS000000000000000LDR": "Layoffs & Discharges Rate (%)",
    }
    df_annual["series_label"] = df_annual["series_id"].map(label_map)

    df_wide = df_annual.pivot(index="year", columns="series_label", values="value")
    print(df_wide.to_string())

    df_api.to_csv("bls_api_monthly.csv", index=False)
    df_wide.to_csv("bls_api_annual_summary.csv")
    print("\nSaved to bls_api_monthly.csv and bls_api_annual_summary.csv")

    # Key observations
    unemp = df_api[df_api["series_id"] == "LNS14000000"]
    peak = unemp.loc[unemp["value"].idxmax()]
    print(f"\nPeak unemployment: {peak['value']}% in {peak['period_name']} {peak['year']}")

    layoffs = df_api[df_api["series_id"] == "JTS000000000000000LDR"]
    if not layoffs.empty:
        peak_l = layoffs.loc[layoffs["value"].idxmax()]
        print(f"Peak layoffs rate: {peak_l['value']}% in {peak_l['period_name']} {peak_l['year']}")

except Exception as e:
    print(f"API error: {e}")
