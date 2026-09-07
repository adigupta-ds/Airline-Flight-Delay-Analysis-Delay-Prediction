# ✈️ Airline Flight Delay Analysis & Delay Prediction

End-to-end analytics project that diagnoses **why** US flights get delayed and **predicts**
delay risk before departure — combining SQL, statistics, machine learning, and an
interactive dashboard.

---

## 🎯 Why this project

Flight delays cost airlines money and cost passengers time — and today, most of that
risk is invisible until it's too late. This project asks three questions and answers
all of them with data:

1. **Where and when do delays actually happen** — which carriers, airports, hours, and
   seasons are the real problem areas? *(SQL + EDA)*
2. **Are those patterns real, or just noise?** *(Statistical hypothesis testing)*
3. **Can we predict delay risk before a flight departs**, using only information known
   at scheduling time? *(Machine learning)*

The result is packaged as an interactive dashboard so the insights are usable by
anyone — not just someone reading a Jupyter notebook.

---

## 📸 Project Overview

<!-- Replace the src paths below with your own screenshot filenames, e.g. "screenshots/overview.png" -->

### Dashboard — Overview page
![Overview dashboard showing KPI cards and monthly delay trend](PASTE_IMAGE_1_HERE.png)

### SQL Insights page
![SQL insights showing carrier and airport delay rankings](PASTE_IMAGE_2_HERE.png)

### EDA & Statistics page
![Interactive heatmap and hypothesis test results](PASTE_IMAGE_3_HERE.png)

### Delay Predictor page
![Live delay risk prediction tool](PASTE_IMAGE_4_HERE.png)

---

## 🧩 What's inside

| Layer | What it does |
|---|---|
| **Data** | 150K flights, schema-matched to the real US DOT/BTS on-time performance dataset |
| **SQL** | Star-schema database + 8 business-question queries, including window functions |
| **Statistics** | ANOVA, Welch's t-test, and Chi-square tests validating every key pattern |
| **Machine Learning** | Logistic Regression + Random Forest predicting 15+ minute delays, leakage-safe |
| **Dashboard** | Streamlit app with 4 interactive pages, including a live delay-risk predictor |

---

## 🛠️ Tech Stack

`Python` · `Pandas` / `NumPy` · `SQLite` / `SQL` · `SciPy` · `Scikit-learn` ·
`Matplotlib` / `Seaborn` · `Plotly` · `Streamlit`

---

## 📁 Project Structure

```
airline-delay-project/
├── data/
│   ├── generate_synthetic_data.py
│   ├── flights_raw.csv
│   └── flights.db
├── sql/
│   ├── schema.sql
│   ├── load_data.py
│   ├── business_queries.sql
│   └── run_queries.py
├── eda/
│   └── eda_analysis.py
├── ml/
│   └── delay_prediction.py
├── outputs/
├── app.py
├── requirements.txt
└── README.md
```

---

## 🚀 How to run it

```bash
# 1. Set up environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Build the data + database (already included, but regenerate if you edit anything)
python data/generate_synthetic_data.py
python sql/load_data.py

# 3. Explore the SQL / EDA / ML layers individually (optional)
python sql/run_queries.py
python eda/eda_analysis.py
python ml/delay_prediction.py

# 4. Launch the interactive dashboard
streamlit run app.py
```

---

## 📊 Key Findings

- **Evening departures (4–9pm) are ~2–3x more likely to be delayed** than early morning flights, across every season.
- **Carrier matters**: on-time performance ranges from ~16% to ~39% delay rate across the 8 carriers analyzed — a statistically significant difference (ANOVA, p < 0.001).
- **Weekend/Friday flights are significantly more delay-prone** than weekday flights (30.1% vs 23.8%, Welch's t-test p < 0.001).
- **Weather-caused delays spike disproportionately in winter** — confirmed independent of chance via a Chi-square test (p < 0.001).
- The Random Forest delay predictor achieves a **0.74 ROC-AUC** using only pre-departure information — realistic performance for this problem, without data leakage.

---

## 🔮 Future Improvements

- Join in real NOAA weather data by airport and date for a stronger causal signal
- Compare against XGBoost / LightGBM
- Deploy publicly via Streamlit Community Cloud
- Expand to multi-year data to test whether patterns hold over time

---

## 👤 Author

Built by [Aditya Sagar Gupta] — [https://www.linkedin.com/in/adigupta-ds] · [https://github.com/adigupta-ds/Airline-Flight-Delay-Analysis-Delay-Prediction/tree/main]
