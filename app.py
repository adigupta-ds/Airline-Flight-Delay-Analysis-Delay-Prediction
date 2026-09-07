"""
app.py
------
Streamlit dashboard for the Airline Flight Delay Analysis project.

Run with:
    streamlit run app.py

Pages (via sidebar):
  1. Overview           - dataset summary, KPIs
  2. SQL Insights        - carrier/airport/route rankings (from business_queries.sql logic)
  3. EDA & Statistics     - interactive charts + hypothesis test results
  4. Delay Predictor      - live "will my flight be delayed?" prediction using the
                            trained Random Forest model
"""
import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "data" / "flights.db"

st.set_page_config(
    page_title="Airline Delay Analytics",
    page_icon="✈️",
    layout="wide",
)

FEATURES_NUM = ["crs_dep_hour", "distance", "crs_elapsed_time", "day_of_week", "month"]
FEATURES_CAT = ["carrier_code", "origin", "dest", "season"]
TARGET = "is_delayed_15"


# ---------------------------------------------------------------
# Cached data / model loaders (so the app doesn't retrain on every click)
# ---------------------------------------------------------------
@st.cache_data
def load_flights():
    if not DB_PATH.exists():
        st.error(
            f"Database not found at {DB_PATH}.\n\n"
            "Run these first from the project root:\n"
            "1. python data/generate_synthetic_data.py\n"
            "2. python sql/load_data.py"
        )
        st.stop()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM flights", conn)
    conn.close()
    return df


@st.cache_resource
def train_model(df):
    """Train the Random Forest once, cache the fitted pipeline across reruns."""
    data = df[df["cancelled"] == 0].dropna(subset=[TARGET]).copy()
    data[TARGET] = data[TARGET].astype(int)

    X = data[FEATURES_NUM + FEATURES_CAT]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preproc = ColumnTransformer([
        ("num", StandardScaler(), FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
    ])
    pipe = Pipeline([
        ("preprocess", preproc),
        ("model", RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
    ])
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, pipe.predict(X_test), target_names=["On-time", "Delayed"])

    return pipe, auc, report


df = load_flights()

# ---------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------
st.sidebar.title("✈️ Airline Delay Analytics")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "SQL Insights", "EDA & Statistics", "Delay Predictor"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Dataset: {len(df):,} flights | "
    f"{df['flight_date'].min()} to {df['flight_date'].max()}"
)

# =================================================================
# PAGE 1: OVERVIEW
# =================================================================
if page == "Overview":
    st.title("✈️ Airline Flight Delay Analysis & Prediction")
    st.markdown(
        "End-to-end analytics project: SQL business queries, exploratory data "
        "analysis, statistical hypothesis testing, and a machine learning model "
        "that predicts flight delays before they happen."
    )

    non_cancelled = df[df["cancelled"] == 0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Flights", f"{len(df):,}")
    c2.metric("% Delayed 15+ min", f"{non_cancelled['is_delayed_15'].mean()*100:.1f}%")
    c3.metric("% Cancelled", f"{df['cancelled'].mean()*100:.2f}%")
    c4.metric("Avg Arrival Delay", f"{non_cancelled['arr_delay_minutes'].mean():.1f} min")

    st.markdown("### Sample of the underlying data")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("### Monthly delay trend")
    monthly = non_cancelled.groupby("month")["is_delayed_15"].mean().reset_index()
    monthly["pct_delayed"] = monthly["is_delayed_15"] * 100
    fig = px.line(monthly, x="month", y="pct_delayed", markers=True,
                  labels={"month": "Month", "pct_delayed": "% Delayed"})
    st.plotly_chart(fig, use_container_width=True)

# =================================================================
# PAGE 2: SQL INSIGHTS  (business_queries.sql logic, made interactive)
# =================================================================
elif page == "SQL Insights":
    st.title("📊 SQL Business Insights")
    non_cancelled = df[df["cancelled"] == 0]

    st.markdown("### Q1 — Which airlines have the worst on-time performance?")
    carrier_stats = (
        non_cancelled.groupby("carrier_code")
        .agg(total_flights=("carrier_code", "count"),
             pct_delayed=("is_delayed_15", lambda x: round(x.mean() * 100, 1)),
             avg_delay_min=("arr_delay_minutes", lambda x: round(x.mean(), 1)))
        .sort_values("pct_delayed", ascending=False)
        .reset_index()
    )
    fig1 = px.bar(carrier_stats, x="carrier_code", y="pct_delayed",
                  color="pct_delayed", color_continuous_scale="Reds",
                  labels={"pct_delayed": "% Delayed", "carrier_code": "Carrier"})
    st.plotly_chart(fig1, use_container_width=True)
    st.dataframe(carrier_stats, use_container_width=True)

    st.markdown("### Q2 — Which origin airports create the most delay risk?")
    airport_stats = (
        non_cancelled.groupby("origin")
        .agg(departures=("origin", "count"),
             pct_delayed=("is_delayed_15", lambda x: round(x.mean() * 100, 1)))
        .sort_values("pct_delayed", ascending=False)
        .head(10)
        .reset_index()
    )
    fig2 = px.bar(airport_stats, x="origin", y="pct_delayed",
                  color="pct_delayed", color_continuous_scale="Oranges",
                  labels={"pct_delayed": "% Delayed", "origin": "Airport"})
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Q4 — Each carrier's 3 worst routes (window function: `RANK() OVER PARTITION BY`)")
    selected_carrier = st.selectbox("Choose a carrier", sorted(df["carrier_code"].unique()))
    route_stats = (
        non_cancelled[non_cancelled["carrier_code"] == selected_carrier]
        .groupby(["origin", "dest"])
        .agg(n_flights=("origin", "count"), delay_rate=("is_delayed_15", "mean"))
        .query("n_flights >= 30")
        .sort_values("delay_rate", ascending=False)
        .head(3)
        .reset_index()
    )
    route_stats["pct_delayed"] = (route_stats["delay_rate"] * 100).round(1)
    st.table(route_stats[["origin", "dest", "n_flights", "pct_delayed"]])

# =================================================================
# PAGE 3: EDA & STATISTICS
# =================================================================
elif page == "EDA & Statistics":
    st.title("🔍 Exploratory Data Analysis & Hypothesis Testing")
    non_cancelled = df[df["cancelled"] == 0]

    st.markdown("### Delay rate by departure hour and month")
    pivot = non_cancelled.pivot_table(
        index="crs_dep_hour", columns="month", values="is_delayed_15", aggfunc="mean"
    ) * 100
    fig_heat = px.imshow(
        pivot, color_continuous_scale="YlOrRd", aspect="auto",
        labels=dict(x="Month", y="Scheduled Departure Hour", color="% Delayed"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("### Arrival delay distribution")
    fig_hist = px.histogram(
        non_cancelled, x="arr_delay_minutes", nbins=60,
        range_x=[-40, 180], labels={"arr_delay_minutes": "Arrival Delay (minutes)"},
    )
    fig_hist.add_vline(x=15, line_dash="dash", line_color="red")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")
    st.markdown("### Statistical hypothesis tests")

    # ANOVA
    groups = [g["arr_delay_minutes"].dropna().values for _, g in non_cancelled.groupby("carrier_code")]
    f_stat, p_anova = stats.f_oneway(*groups)
    with st.expander("① One-way ANOVA — do carriers really differ in arrival delay?", expanded=True):
        st.write(f"**F-statistic:** {f_stat:.2f}  |  **p-value:** {p_anova:.2e}")
        st.success("Significant difference across carriers (p < 0.05)." if p_anova < 0.05
                   else "No significant difference across carriers.")

    # t-test
    weekend = non_cancelled[non_cancelled["day_of_week"].isin([4, 5, 6])]["is_delayed_15"].dropna()
    weekday = non_cancelled[~non_cancelled["day_of_week"].isin([4, 5, 6])]["is_delayed_15"].dropna()
    t_stat, p_t = stats.ttest_ind(weekend, weekday, equal_var=False)
    with st.expander("② Welch's t-test — weekend/Friday vs weekday delay rate"):
        st.write(f"Weekend/Fri: **{weekend.mean()*100:.1f}%** delayed  |  "
                 f"Weekday: **{weekday.mean()*100:.1f}%** delayed")
        st.write(f"**t-statistic:** {t_stat:.2f}  |  **p-value:** {p_t:.2e}")
        st.success("Statistically significant difference (p < 0.05)." if p_t < 0.05
                   else "No statistically significant difference.")

    # chi-square
    delayed = non_cancelled[(non_cancelled["is_delayed_15"] == 1) & (non_cancelled["delay_cause"] != "")]
    contingency = pd.crosstab(delayed["season"], delayed["delay_cause"])
    chi2, p_chi, dof, _ = stats.chi2_contingency(contingency)
    with st.expander("③ Chi-square — is delay cause independent of season?"):
        st.write(f"**χ²:** {chi2:.1f}  |  **dof:** {dof}  |  **p-value:** {p_chi:.2e}")
        st.success("Delay cause depends on season — e.g. weather spikes in winter (p < 0.05)."
                   if p_chi < 0.05 else "Delay cause looks independent of season.")
        st.dataframe(contingency, use_container_width=True)

# =================================================================
# PAGE 4: DELAY PREDICTOR (live ML inference)
# =================================================================
elif page == "Delay Predictor":
    st.title("🤖 Flight Delay Predictor")
    st.markdown(
        "Enter a hypothetical flight's details below. The model (Random Forest, "
        "trained only on information known **before** departure — no data leakage) "
        "estimates the probability of a 15+ minute arrival delay."
    )

    with st.spinner("Training model (cached after first run)..."):
        pipe, auc, report = train_model(df)

    st.caption(f"Model test-set ROC-AUC: **{auc:.3f}**")
    with st.expander("Full classification report"):
        st.text(report)

    st.markdown("### Flight details")
    col1, col2, col3 = st.columns(3)
    with col1:
        carrier = st.selectbox("Carrier", sorted(df["carrier_code"].unique()))
        origin = st.selectbox("Origin airport", sorted(df["origin"].unique()))
    with col2:
        dest_options = sorted([a for a in df["origin"].unique() if a != origin])
        dest = st.selectbox("Destination airport", dest_options)
        season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])
    with col3:
        dep_hour = st.slider("Scheduled departure hour", 0, 23, 17)
        day_of_week = st.selectbox(
            "Day of week",
            options=list(range(7)),
            format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
        )

    month_map = {"Winter": 1, "Spring": 4, "Summer": 7, "Fall": 10}
    input_df = pd.DataFrame([{
        "crs_dep_hour": dep_hour,
        "distance": 1000,
        "crs_elapsed_time": 150,
        "day_of_week": day_of_week,
        "month": month_map[season],
        "carrier_code": carrier,
        "origin": origin,
        "dest": dest,
        "season": season,
    }])

    if st.button("Predict delay risk", type="primary"):
        proba = pipe.predict_proba(input_df)[0, 1]
        st.markdown(f"## Predicted delay probability: **{proba*100:.1f}%**")
        if proba >= 0.5:
            st.error("⚠️ High risk of a 15+ minute delay.")
        elif proba >= 0.3:
            st.warning("⚡ Moderate delay risk.")
        else:
            st.success("✅ Low delay risk.")
        st.progress(min(proba, 1.0))
