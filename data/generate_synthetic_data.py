"""
generate_synthetic_data.py
---------------------------
Generates a synthetic flight on-time performance dataset that mirrors the
real US DOT / Bureau of Transportation Statistics (BTS) "Reporting Carrier
On-Time Performance" dataset schema.

WHY SYNTHETIC DATA?
This sandbox can't reach transtats.bts.gov. But because the column names,
data types, and delay-cause logic below match the real dataset exactly,
every downstream script (SQL loading, EDA, stats, ML) will work unchanged
if you swap this file for a real BTS monthly CSV.

TO USE REAL DATA INSTEAD:
1. Go to https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236
2. Select a month/year, check the fields matching those below
3. Download the CSV, rename it to flights_raw.csv, drop it in data/
4. Skip this script and go straight to sql/load_data.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# ---------------------------------------------------------------
# 1. Reference dimensions (carriers, airports) - modeled on real US aviation
# ---------------------------------------------------------------
CARRIERS = {
    "AA": "American Airlines",
    "DL": "Delta Air Lines",
    "UA": "United Airlines",
    "WN": "Southwest Airlines",
    "B6": "JetBlue Airways",
    "AS": "Alaska Airlines",
    "NK": "Spirit Airlines",
    "F9": "Frontier Airlines",
}

# Airports with rough "congestion factor" (higher = more baseline delay risk)
AIRPORTS = {
    "ATL": 0.55, "ORD": 0.70, "DFW": 0.50, "DEN": 0.45, "LAX": 0.60,
    "JFK": 0.75, "LAS": 0.35, "MCO": 0.40, "CLT": 0.45, "SEA": 0.30,
    "SFO": 0.65, "EWR": 0.72, "PHX": 0.35, "MIA": 0.50, "BOS": 0.55,
}

AIRPORT_LIST = list(AIRPORTS.keys())
CARRIER_LIST = list(CARRIERS.keys())

# Each carrier gets a baseline "operational quality" multiplier (hidden ground truth)
CARRIER_QUALITY = {c: np.random.uniform(0.7, 1.3) for c in CARRIER_LIST}

N_FLIGHTS = 150_000
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2023, 12, 31)


def random_dates(n):
    days_range = (END_DATE - START_DATE).days
    offsets = np.random.randint(0, days_range, n)
    return [START_DATE + timedelta(days=int(o)) for o in offsets]


def build_dataset(n=N_FLIGHTS):
    dates = random_dates(n)
    carriers = np.random.choice(CARRIER_LIST, n)
    origins = np.random.choice(AIRPORT_LIST, n)

    # destination != origin
    dests = np.random.choice(AIRPORT_LIST, n)
    same = origins == dests
    while same.any():
        dests[same] = np.random.choice(AIRPORT_LIST, same.sum())
        same = origins == dests

    # scheduled departure hour: realistic bimodal (morning + evening peak)
    hour_choices = np.concatenate([
        np.random.normal(8, 1.5, n // 2),
        np.random.normal(17, 2, n - n // 2),
    ])
    sched_hour = np.clip(hour_choices, 0, 23).astype(int)
    sched_minute = np.random.choice([0, 15, 30, 45], n)

    df = pd.DataFrame({
        "FlightDate": dates,
        "Reporting_Airline": carriers,
        "Origin": origins,
        "Dest": dests,
        "CRSDepHour": sched_hour,
        "CRSDepMinute": sched_minute,
    })

    df["DayOfWeek"] = df["FlightDate"].dt.dayofweek  # 0=Mon
    df["Month"] = df["FlightDate"].dt.month

    def season(m):
        if m in (12, 1, 2):
            return "Winter"
        if m in (3, 4, 5):
            return "Spring"
        if m in (6, 7, 8):
            return "Summer"
        return "Fall"
    df["Season"] = df["Month"].apply(season)

    # ---------------------------------------------------------------
    # 2. Simulate delay-driving factors (this is the "ground truth" the
    #    ML model will later have to rediscover from the data)
    # ---------------------------------------------------------------
    origin_cong = df["Origin"].map(AIRPORTS).values
    dest_cong = df["Dest"].map(AIRPORTS).values
    carrier_q = df["Reporting_Airline"].map(CARRIER_QUALITY).values

    # evening flights and winter months delay more (weather + congestion)
    hour_effect = np.where(df["CRSDepHour"] >= 16, 0.35,
                   np.where(df["CRSDepHour"] <= 6, -0.25, 0.0))
    season_effect = df["Season"].map({"Winter": 0.30, "Summer": 0.15,
                                       "Spring": 0.0, "Fall": 0.0}).values
    weekend_effect = np.where(df["DayOfWeek"].isin([4, 6]), 0.15, 0.0)

    # base delay risk score -> probability of delay > 15 min
    risk_score = (
        0.20
        + 0.25 * origin_cong
        + 0.10 * dest_cong
        + hour_effect
        + season_effect
        + weekend_effect
    ) * carrier_q

    delay_prob = 1 / (1 + np.exp(-4 * (risk_score - 0.95)))  # logistic squash
    is_delayed = np.random.binomial(1, np.clip(delay_prob, 0.03, 0.85))

    # Departure delay minutes: 0 (or small negative = early) if not delayed,
    # else drawn from a right-skewed distribution
    dep_delay = np.where(
        is_delayed == 1,
        np.random.gamma(shape=2.0, scale=25, size=n) + 15,
        np.random.normal(-2, 5, n),
    )
    dep_delay = np.round(dep_delay, 0)

    # Arrival delay correlates with departure delay + a bit of noise/recovery
    arr_delay = dep_delay + np.random.normal(-3, 8, n)
    arr_delay = np.round(arr_delay, 0)

    df["DepDelayMinutes"] = np.clip(dep_delay, -30, None)
    df["ArrDelayMinutes"] = np.clip(arr_delay, -40, None)
    df["IsDelayed15"] = (df["ArrDelayMinutes"] >= 15).astype(int)

    # ---------------------------------------------------------------
    # 3. Cancellations (rare, weather/carrier driven) + delay cause codes
    #    (mirrors real BTS cause categories: Carrier, Weather, NAS, Security, Late Aircraft)
    # ---------------------------------------------------------------
    cancel_prob = 0.015 + 0.03 * season_effect
    df["Cancelled"] = np.random.binomial(1, np.clip(cancel_prob, 0, 0.15))
    df.loc[df["Cancelled"] == 1, ["DepDelayMinutes", "ArrDelayMinutes", "IsDelayed15"]] = np.nan

    causes = np.array(["Carrier", "Weather", "NAS", "Security", "LateAircraft"])
    cause_weights_default = np.array([0.30, 0.20, 0.30, 0.02, 0.18])
    delay_cause = np.full(n, "", dtype=object)
    delayed_mask = (df["IsDelayed15"] == 1)
    winter_mask = df["Season"] == "Winter"

    for idx in df.index[delayed_mask]:
        w = cause_weights_default.copy()
        if winter_mask[idx]:
            w = np.array([0.22, 0.38, 0.25, 0.02, 0.13])  # more weather in winter
        w = w / w.sum()
        delay_cause[idx] = np.random.choice(causes, p=w)
    df["DelayCause"] = delay_cause

    # scheduled elapsed time & distance (for realism / extra features)
    df["Distance"] = np.random.randint(200, 2800, n)
    df["CRSElapsedTime"] = (df["Distance"] / 7.5 + np.random.normal(0, 8, n)).round().astype(int)

    # Flight number & tail number (cosmetic realism)
    df["FlightNum"] = np.random.randint(100, 6000, n)
    df["TailNum"] = ["N" + str(np.random.randint(100, 999)) + np.random.choice(list("ABCDEFGH")) for _ in range(n)]

    # Reorder columns to match BTS-style layout
    cols = [
        "FlightDate", "Reporting_Airline", "FlightNum", "TailNum",
        "Origin", "Dest", "CRSDepHour", "CRSDepMinute", "CRSElapsedTime",
        "Distance", "DayOfWeek", "Month", "Season",
        "DepDelayMinutes", "ArrDelayMinutes", "IsDelayed15",
        "Cancelled", "DelayCause",
    ]
    return df[cols]


if __name__ == "__main__":
    df = build_dataset()
    out_path = "/home/claude/airline-delay-project/data/flights_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} flight records -> {out_path}")
    print(df.head())
    print("\nDelay rate (>=15min):", df["IsDelayed15"].mean().round(3))
    print("Cancellation rate:", df["Cancelled"].mean().round(3))
