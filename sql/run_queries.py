"""
run_queries.py
--------------
Splits business_queries.sql on the '-- Q' markers and runs each block,
printing a labeled pandas DataFrame. Handy for demoing SQL output without
a separate SQL client.
"""
import re
import sqlite3
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DB_PATH = BASE / "data" / "flights.db"
SQL_PATH = Path(__file__).resolve().parent / "business_queries.sql"

pd.set_option("display.max_rows", 20)
pd.set_option("display.width", 120)


def main():
    conn = sqlite3.connect(DB_PATH)
    raw = SQL_PATH.read_text()

    # split into blocks starting with "-- Q<number>:"
    blocks = re.split(r"(?=-- Q\d+:)", raw)
    for block in blocks:
        block = block.strip()
        if not block.startswith("-- Q"):
            continue
        header = block.splitlines()[0].replace("-- ", "")
        query = "\n".join(l for l in block.splitlines() if not l.strip().startswith("--"))
        query = query.strip().rstrip(";")
        if not query:
            continue
        print("=" * 90)
        print(header)
        print("=" * 90)
        try:
            df = pd.read_sql_query(query, conn)
            print(df.to_string(index=False))
        except Exception as e:
            print("ERROR running query:", e)
        print()
    conn.close()


if __name__ == "__main__":
    main()
