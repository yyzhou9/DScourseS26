import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

# ── 0. Config ──────────────────────────────────────────────────────────────
BLS_API_KEY   = "YOUR_BLS_API_KEY"          # replace or set env var BLS_API_KEY
OUTPUT_DIR    = "."
SERIES_FILE   = "bls_series.csv"            # intermediate cache

# State-level unemployment series IDs  (LAUST + FIPS + 00000003)
STATE_FIPS = {
    "Alabama": "01", "Alaska": "02", "Arizona": "04", "Arkansas": "05",
    "California": "06", "Colorado": "08", "Connecticut": "09", "Delaware": "10",
    "Florida": "12", "Georgia": "13", "Hawaii": "15", "Idaho": "16",
    "Illinois": "17", "Indiana": "18", "Iowa": "19", "Kansas": "20",
    "Kentucky": "21", "Louisiana": "22", "Maine": "23", "Maryland": "24",
    "Massachusetts": "25", "Michigan": "26", "Minnesota": "27", "Mississippi": "28",
    "Missouri": "29", "Montana": "30", "Nebraska": "31", "Nevada": "32",
    "New Hampshire": "33", "New Jersey": "34", "New Mexico": "35", "New York": "36",
    "North Carolina": "37", "North Dakota": "38", "Ohio": "39", "Oklahoma": "40",
    "Oregon": "41", "Pennsylvania": "42", "Rhode Island": "44", "South Carolina": "45",
    "South Dakota": "46", "Tennessee": "47", "Texas": "48", "Utah": "49",
    "Vermont": "50", "Virginia": "51", "Washington": "53", "West Virginia": "54",
    "Wisconsin": "55", "Wyoming": "56",
}
SERIES_IDS = {state: f"LAUST{fips}0000000000003" for state, fips in STATE_FIPS.items()}

# ── 1. Fetch from BLS API (v2) ──────────────────────────────────────────────
def fetch_bls(series_list, start_year="2010", end_year="2024"):
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    headers = {"Content-type": "application/json"}
    all_rows = []
    # BLS v2 allows up to 50 series per request
    for i in range(0, len(series_list), 50):
        chunk = series_list[i:i+50]
        payload = {
            "seriesid": chunk,
            "startyear": start_year,
            "endyear": end_year,
            "registrationkey": BLS_API_KEY,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        for series in data.get("Results", {}).get("series", []):
            sid = series["seriesID"]
            state = next(s for s, v in SERIES_IDS.items() if v == sid)
            for item in series["data"]:
                all_rows.append({
                    "state": state,
                    "year":  int(item["year"]),
                    "period": item["period"],
                    "value": float(item["value"]),
                })
    return pd.DataFrame(all_rows)


def load_or_fetch():
    cache = os.path.join(OUTPUT_DIR, SERIES_FILE)
    if os.path.exists(cache):
        return pd.read_csv(cache)
    df = fetch_bls(list(SERIES_IDS.values()))
    df.to_csv(cache, index=False)
    return df


# ── 2. Clean ───────────────────────────────────────────────────────────────
def clean(df):
    # Keep annual averages only (period == "M13" in BLS annual series)
    # Some annual series use period "M13"; monthly series use M01-M12
    # For LAUST annual series, period == "M13" is the annual average
    df = df[df["period"] == "M13"].copy()
    df = df.drop(columns=["period"])
    df = df.rename(columns={"value": "unemp_rate"})
    # Drop rows with missing values (BLS sometimes returns "-" which we cast to NaN)
    df = df.dropna(subset=["unemp_rate"])
    # Add region grouping
    regions = {
        "Northeast": ["Connecticut","Maine","Massachusetts","New Hampshire",
                      "New Jersey","New York","Pennsylvania","Rhode Island","Vermont"],
        "Midwest":   ["Illinois","Indiana","Iowa","Kansas","Michigan","Minnesota",
                      "Missouri","Nebraska","North Dakota","Ohio","South Dakota","Wisconsin"],
        "South":     ["Alabama","Arkansas","Delaware","Florida","Georgia","Kentucky",
                      "Louisiana","Maryland","Mississippi","North Carolina","Oklahoma",
                      "South Carolina","Tennessee","Texas","Virginia","West Virginia"],
        "West":      ["Alaska","Arizona","California","Colorado","Hawaii","Idaho",
                      "Montana","Nevada","New Mexico","Oregon","Utah","Washington","Wyoming"],
    }
    region_map = {state: reg for reg, states in regions.items() for state in states}
    df["region"] = df["state"].map(region_map)
    return df.sort_values(["state", "year"]).reset_index(drop=True)


# ── 3. Visualizations ──────────────────────────────────────────────────────
STYLE = {
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
}

def fig_a(df):
    """PS6a – National annual average unemployment rate 2010–2024."""
    national = df.groupby("year")["unemp_rate"].mean().reset_index()

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(national["year"], national["unemp_rate"],
                color="#1f77b4", linewidth=2.2, marker="o", markersize=5)
        ax.fill_between(national["year"], national["unemp_rate"],
                        alpha=0.15, color="#1f77b4")
        # Annotate COVID spike
        peak = national.loc[national["unemp_rate"].idxmax()]
        ax.annotate(f"COVID-19 peak\n{peak['unemp_rate']:.1f}%",
                    xy=(peak["year"], peak["unemp_rate"]),
                    xytext=(peak["year"] - 1.5, peak["unemp_rate"] - 1.2),
                    arrowprops=dict(arrowstyle="->", color="gray"),
                    fontsize=9, color="gray")
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel("Unemployment Rate (%)", fontsize=11)
        ax.set_title("U.S. State-Average Unemployment Rate, 2010–2024", fontsize=13, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        ax.set_xticks(national["year"])
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "PS6a_Zhou.png"), bbox_inches="tight")
        plt.close(fig)
    print("Saved PS6a_Zhou.png")


def fig_b(df):
    """PS6b – Regional unemployment distribution (box plot) by year bucket."""
    # Bucket years into three periods
    bins = [2009, 2014, 2019, 2024]
    labels = ["2010–2014", "2015–2019", "2020–2024"]
    df = df.copy()
    df["period"] = pd.cut(df["year"], bins=bins, labels=labels)
    df = df.dropna(subset=["period"])

    regions = ["Northeast", "Midwest", "South", "West"]
    colors  = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
        for ax, period in zip(axes, labels):
            sub = df[df["period"] == period]
            data_by_region = [sub[sub["region"] == r]["unemp_rate"].dropna().values
                              for r in regions]
            bp = ax.boxplot(data_by_region, patch_artist=True, widths=0.5,
                            medianprops=dict(color="black", linewidth=1.8))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
            ax.set_xticklabels(regions, fontsize=9)
            ax.set_title(period, fontsize=11, fontweight="bold")
            ax.set_xlabel("Region", fontsize=10)
        axes[0].set_ylabel("Unemployment Rate (%)", fontsize=11)
        axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        fig.suptitle("State Unemployment Distribution by Region and Period",
                     fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "PS6b_Zhou.png"), bbox_inches="tight")
        plt.close(fig)
    print("Saved PS6b_Zhou.png")


def fig_c(df):
    """PS6c – Heatmap: state × year unemployment rate (2019–2024 subset)."""
    sub = df[df["year"] >= 2019].copy()
    pivot = sub.pivot_table(index="state", columns="year", values="unemp_rate")
    # Sort states by 2024 unemployment (ascending)
    pivot = pivot.sort_values(2024, ascending=False)

    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(10, 14))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                       vmin=2, vmax=10)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=10)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        cbar.set_label("Unemployment Rate (%)", fontsize=10)
        ax.set_title("State Unemployment Rates, 2019–2024\n(sorted by 2024 rate)",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "PS6c_Zhou.png"), bbox_inches="tight")
        plt.close(fig)
    print("Saved PS6c_Zhou.png")


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    raw = load_or_fetch()
    df  = clean(raw)
    print(f"Clean dataset: {len(df)} rows, {df['state'].nunique()} states, "
          f"years {df['year'].min()}–{df['year'].max()}")
    fig_a(df)
    fig_b(df)
    fig_c(df)
    print("Done.")
