"""
eda_analysis.py
----------------
Exploratory data analysis + statistical hypothesis tests.

Produces (saved to outputs/):
  1. delay_distribution.png     - histogram of arrival delay minutes
  2. delay_by_carrier.png       - bar chart, % flights delayed by carrier
  3. delay_heatmap.png          - hour-of-day x month delay-rate heatmap
  4. seasonal_trend.png         - monthly delay rate line chart

Statistical tests:
  - One-way ANOVA: do carriers differ significantly in arrival delay?
  - Welch's t-test: weekday vs weekend/Friday delay rate
  - Chi-square: is delay cause distribution independent of season?
"""
import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
DB_PATH = BASE / "data" / "flights.db"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM flights WHERE cancelled = 0", conn)
    conn.close()
    return df


def plot_delay_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df["arr_delay_minutes"].clip(upper=180), bins=60, kde=False, color="#3b6fa0")
    plt.axvline(15, color="red", linestyle="--", label="15-min delay threshold")
    plt.title("Distribution of Arrival Delay Minutes")
    plt.xlabel("Arrival Delay (minutes, clipped at 180)")
    plt.ylabel("Number of Flights")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "delay_distribution.png", dpi=140)
    plt.close()


def plot_delay_by_carrier(df):
    rate = df.groupby("carrier_code")["is_delayed_15"].mean().sort_values(ascending=False) * 100
    plt.figure(figsize=(8, 5))
    sns.barplot(x=rate.index, y=rate.values, hue=rate.index, palette="rocket", legend=False)
    plt.title("% of Flights Delayed 15+ Minutes, by Carrier")
    plt.ylabel("% Delayed")
    plt.xlabel("Carrier")
    plt.tight_layout()
    plt.savefig(OUT / "delay_by_carrier.png", dpi=140)
    plt.close()


def plot_delay_heatmap(df):
    pivot = df.pivot_table(index="crs_dep_hour", columns="month",
                            values="is_delayed_15", aggfunc="mean") * 100
    plt.figure(figsize=(10, 7))
    sns.heatmap(pivot, cmap="YlOrRd", annot=False, cbar_kws={"label": "% Delayed"})
    plt.title("Delay Rate (%) by Departure Hour and Month")
    plt.xlabel("Month")
    plt.ylabel("Scheduled Departure Hour")
    plt.tight_layout()
    plt.savefig(OUT / "delay_heatmap.png", dpi=140)
    plt.close()


def plot_seasonal_trend(df):
    monthly = df.groupby("month")["is_delayed_15"].mean() * 100
    plt.figure(figsize=(8, 5))
    plt.plot(monthly.index, monthly.values, marker="o", color="#c0392b")
    plt.title("Monthly Flight Delay Rate (%)")
    plt.xlabel("Month")
    plt.ylabel("% Delayed (15+ min)")
    plt.xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig(OUT / "seasonal_trend.png", dpi=140)
    plt.close()


def run_stat_tests(df):
    results = {}

    # ---- 1. ANOVA: arrival delay across carriers ----
    groups = [g["arr_delay_minutes"].dropna().values for _, g in df.groupby("carrier_code")]
    f_stat, p_val = stats.f_oneway(*groups)
    results["anova_carrier_delay"] = {
        "test": "One-way ANOVA - arrival delay across carriers",
        "F_statistic": round(f_stat, 3),
        "p_value": p_val,
        "conclusion": (
            "Reject H0: at least one carrier has a significantly different mean "
            "arrival delay (p < 0.05)." if p_val < 0.05 else
            "Fail to reject H0: no significant difference across carriers."
        ),
    }

    # ---- 2. Welch's t-test: weekday vs weekend/Friday delay rate ----
    weekend = df[df["day_of_week"].isin([4, 5, 6])]["is_delayed_15"].dropna()
    weekday = df[~df["day_of_week"].isin([4, 5, 6])]["is_delayed_15"].dropna()
    t_stat, p_val_t = stats.ttest_ind(weekend, weekday, equal_var=False)
    results["ttest_weekend_weekday"] = {
        "test": "Welch's t-test - delay rate: weekend/Fri vs weekday",
        "t_statistic": round(t_stat, 3),
        "p_value": p_val_t,
        "weekend_mean_pct": round(weekend.mean() * 100, 2),
        "weekday_mean_pct": round(weekday.mean() * 100, 2),
        "conclusion": (
            "Statistically significant difference in delay rate (p < 0.05)."
            if p_val_t < 0.05 else "No statistically significant difference."
        ),
    }

    # ---- 3. Chi-square: delay cause independent of season? ----
    delayed = df[(df["is_delayed_15"] == 1) & (df["delay_cause"] != "")]
    contingency = pd.crosstab(delayed["season"], delayed["delay_cause"])
    chi2, p_val_c, dof, _ = stats.chi2_contingency(contingency)
    results["chisq_cause_season"] = {
        "test": "Chi-square - delay cause vs season independence",
        "chi2_statistic": round(chi2, 3),
        "dof": dof,
        "p_value": p_val_c,
        "conclusion": (
            "Reject H0: delay cause distribution depends on season (p < 0.05) "
            "- e.g. weather-driven delays spike in winter."
            if p_val_c < 0.05 else
            "Fail to reject H0: delay cause looks independent of season."
        ),
    }

    return results


def main():
    df = load_data()
    print(f"Loaded {len(df):,} non-cancelled flights for EDA.\n")

    plot_delay_distribution(df)
    plot_delay_by_carrier(df)
    plot_delay_heatmap(df)
    plot_seasonal_trend(df)
    print("Saved 4 charts to outputs/\n")

    results = run_stat_tests(df)
    print("=" * 80)
    print("STATISTICAL TEST RESULTS")
    print("=" * 80)
    for _, r in results.items():
        print(f"\n{r['test']}")
        for k, v in r.items():
            if k == "test":
                continue
            print(f"   {k}: {v}")


if __name__ == "__main__":
    main()
