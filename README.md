# ✈️ Airline Flight Delay Analysis & Delay Prediction

End-to-end analytics project covering **SQL, statistics, machine learning, and business
storytelling**, built on flight on-time performance data modeled after the real
US DOT/BTS "Reporting Carrier On-Time Performance" dataset.

## Project structure

```
airline-delay-project/
├── data/
│   ├── generate_synthetic_data.py   # builds a realistic 150k-flight dataset
│   ├── flights_raw.csv              # generated data (BTS-schema compatible)
│   └── flights.db                   # SQLite database (built by sql/load_data.py)
├── sql/
│   ├── schema.sql                   # star-schema: flights fact + airline/airport dims
│   ├── load_data.py                 # loads CSV -> SQLite
│   ├── business_queries.sql         # 8 business-question SQL queries
│   └── run_queries.py               # runs & pretty-prints all queries
├── eda/
│   └── eda_analysis.py              # charts + ANOVA / t-test / chi-square tests
├── ml/
│   └── delay_prediction.py          # logistic regression + random forest classifier
├── outputs/                         # all generated charts land here
├── app.py                           # Streamlit dashboard (SQL + EDA + stats + live ML predictor)
├── requirements.txt
└── README.md
```

## How to run it (in order)

```bash
pip install -r requirements.txt

python data/generate_synthetic_data.py   # 1. generate data
python sql/load_data.py                  # 2. load into SQLite
python sql/run_queries.py                # 3. run business SQL queries
python eda/eda_analysis.py               # 4. EDA charts + statistical tests
python ml/delay_prediction.py            # 5. train & evaluate ML models
```

### Using real data instead of synthetic
Download a month (or several) of the free BTS on-time performance data from
https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236, save it as
`data/flights_raw.csv` with matching column names, and skip step 1 — everything
downstream (SQL, EDA, ML) works unchanged because the schema matches.

---

## 1. Data

150,000 synthetic flights across 2023, 8 major US carriers, and 15 major airports,
generated to match BTS's real schema and realistic delay patterns (rush-hour and
winter-weather effects, carrier-specific reliability differences, right-skewed delay
distributions, ~26.5% delay rate — consistent with real-world US aviation stats).

## 2. SQL layer

Normalized into a `flights` fact table plus `airlines`/`airports` dimension tables.
`sql/business_queries.sql` answers 8 real ops questions an airline analyst would ask,
including two window-function queries:

- **Q4** — `RANK() OVER (PARTITION BY carrier ...)` to find each carrier's 3 worst routes
- **Q5** — 3-month moving average of delay rate using `ROWS BETWEEN 2 PRECEDING AND CURRENT ROW`

### Key SQL findings
| Metric | Finding |
|---|---|
| Worst carrier | Delta (synthetic) — 38.6% of flights delayed 15+ min |
| Best carrier | Spirit (synthetic) — 15.7% delayed |
| Worst airport | JFK — 30.4% delayed departures |
| Peak delay window | 4pm–9pm, every season |

## 3. EDA + Statistical Testing

Four charts saved to `outputs/`: delay distribution, delay-by-carrier bar chart,
hour×month delay heatmap, and monthly trend line.

Three hypothesis tests, each answering a specific business question:

| Test | Question | Result |
|---|---|---|
| One-way ANOVA | Do carriers really differ in delay, or is it noise? | F=594.9, p≈0 → **Yes, significant** |
| Welch's t-test | Are weekends/Fridays worse than weekdays? | 30.1% vs 23.8% delayed, p≈0 → **Yes, significant** |
| Chi-square | Does delay *cause* shift by season (e.g., weather in winter)? | χ²=1314, p≈0 → **Yes**, weather-related delays spike in winter |

## 4. ML — Delay Prediction

**Framing:** binary classification — will this flight be delayed 15+ minutes at
arrival? Only features known *at scheduling time* are used (carrier, route, scheduled
hour, day of week, month, season, distance) — deliberately excluding actual departure
delay to avoid data leakage.

**Why these metrics:** the classes are imbalanced (~26% delayed), so accuracy alone is
misleading. Precision/recall/F1/ROC-AUC are reported, and `class_weight="balanced"` is
used in both models.

| Model | ROC-AUC | Delayed-class Recall | Notes |
|---|---|---|---|
| Logistic Regression (baseline) | 0.730 | 0.67 | Fast, interpretable |
| Random Forest | 0.744 | 0.64 | Best overall AUC |

**Top predictive features:** scheduled departure hour (by far the strongest driver),
followed by winter season, month, and specific carrier identity — directly consistent
with the EDA heatmap and ANOVA result.

## 5. Business Recommendations

1. **Schedule padding for evening departures.** Delay rates roughly triple between
   morning (6–10am, ~13%) and evening (4–9pm, ~35-45%) departures across every season.
   Airlines and travelers should treat any flight after 4pm as meaningfully higher risk.
2. **Winter ops resourcing.** Weather-attributed delays are disproportionately
   concentrated in winter months (chi-square confirms this isn't random) — de-icing
   and ground-crew staffing should scale seasonally, not stay flat year-round.
3. **Carrier-specific route audits.** The window-function query (Q4) identifies each
   carrier's three worst-performing routes — a natural starting point for an ops
   deep-dive rather than reacting to airline-wide averages.
4. **Set customer expectations by data, not policy.** A predicted delay-risk score
   (from the RF model) could feed into check-in/booking flows to proactively warn
   travelers on high-risk itineraries.

## 6. Streamlit Dashboard

```bash
streamlit run app.py
```

Four interactive pages: **Overview** (KPIs + trend), **SQL Insights** (carrier/airport/route
rankings, including the ranked-worst-routes window-function query), **EDA & Statistics**
(interactive heatmap + the three hypothesis tests), and **Delay Predictor** (live "will my
flight be delayed?" tool backed by the trained Random Forest).

## Next steps to extend this project
- Join in real NOAA weather data by airport/date for a stronger causal weather signal
- Try XGBoost/LightGBM and compare against the Random Forest
- Wrap the trained model in a small Streamlit app for an interactive demo
- Expand to multi-year data to check whether patterns hold over time
