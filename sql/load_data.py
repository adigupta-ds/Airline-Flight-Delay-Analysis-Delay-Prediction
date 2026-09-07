"""
load_data.py
------------
Loads flights_raw.csv into a normalized SQLite database (flights.db)
using schema.sql. This is the "SQL" layer of the project: from here on,
all business-question analysis is done with SQL, not pandas.
"""

import sqlite3
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CSV_PATH = BASE / "data" / "flights_raw.csv"
DB_PATH = BASE / "data" / "flights.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def main():
    df = pd.read_csv(CSV_PATH, parse_dates=["FlightDate"])

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 1. Build schema
    with open(SCHEMA_PATH) as f:
        cur.executescript(f.read())

    # 2. Populate dimension tables
    carriers = df[["Reporting_Airline"]].drop_duplicates().rename(
        columns={"Reporting_Airline": "carrier_code"})
    carriers["carrier_name"] = carriers["carrier_code"]  # placeholder name
    carriers.to_sql("airlines", conn, if_exists="append", index=False)

    airports = pd.DataFrame({"airport_code": pd.unique(df[["Origin", "Dest"]].values.ravel())})
    airports["congestion_factor"] = None
    airports.to_sql("airports", conn, if_exists="append", index=False)

    # 3. Populate fact table
    fact = df.rename(columns={
        "FlightDate": "flight_date",
        "Reporting_Airline": "carrier_code",
        "FlightNum": "flight_num",
        "TailNum": "tail_num",
        "Origin": "origin",
        "Dest": "dest",
        "CRSDepHour": "crs_dep_hour",
        "CRSDepMinute": "crs_dep_minute",
        "CRSElapsedTime": "crs_elapsed_time",
        "Distance": "distance",
        "DayOfWeek": "day_of_week",
        "Month": "month",
        "Season": "season",
        "DepDelayMinutes": "dep_delay_minutes",
        "ArrDelayMinutes": "arr_delay_minutes",
        "IsDelayed15": "is_delayed_15",
        "Cancelled": "cancelled",
        "DelayCause": "delay_cause",
    })
    fact["flight_date"] = fact["flight_date"].dt.strftime("%Y-%m-%d")
    fact_cols = [c for c in fact.columns if c != "flight_id"]
    fact[fact_cols].to_sql("flights", conn, if_exists="append", index=False)

    conn.commit()

    n_flights = cur.execute("SELECT COUNT(*) FROM flights").fetchone()[0]
    print(f"Loaded {n_flights:,} flights into {DB_PATH}")
    conn.close()


if __name__ == "__main__":
    main()
