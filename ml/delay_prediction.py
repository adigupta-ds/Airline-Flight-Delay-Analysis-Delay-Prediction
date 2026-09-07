"""
delay_prediction.py
--------------------
Predicts whether a flight will be delayed 15+ minutes at arrival, using
only information KNOWN AT SCHEDULING TIME (no data leakage - we deliberately
exclude dep_delay_minutes/arr_delay_minutes as features, since those are
only known after the flight already happened).

Models:
  1. Logistic Regression (baseline, interpretable)
  2. Random Forest (main model)

Since delays are imbalanced (~26% positive class), we report precision,
recall, F1, and ROC-AUC rather than accuracy alone, and use
class_weight="balanced".
"""
import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
)

BASE = Path(__file__).resolve().parent.parent
DB_PATH = BASE / "data" / "flights.db"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)


FEATURES_NUM = ["crs_dep_hour", "distance", "crs_elapsed_time", "day_of_week", "month"]
FEATURES_CAT = ["carrier_code", "origin", "dest", "season"]
TARGET = "is_delayed_15"


def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        f"""SELECT {', '.join(FEATURES_NUM + FEATURES_CAT)}, {TARGET}
            FROM flights WHERE cancelled = 0""",
        conn,
    )
    conn.close()
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    return df


def build_pipeline(model):
    preproc = ColumnTransformer([
        ("num", StandardScaler(), FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
    ])
    return Pipeline([("preprocess", preproc), ("model", model)])


def evaluate(name, pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    print("=" * 80)
    print(f"{name} - Test Set Performance")
    print("=" * 80)
    print(classification_report(y_test, y_pred, target_names=["On-time", "Delayed"]))
    print(f"ROC-AUC: {auc:.3f}\n")

    return y_pred, y_proba, auc


def main():
    df = load_data()
    print(f"Loaded {len(df):,} flights. Delay rate: {df[TARGET].mean():.1%}\n")

    X = df[FEATURES_NUM + FEATURES_CAT]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- Model 1: Logistic Regression baseline ----
    logreg = build_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"))
    logreg.fit(X_train, y_train)
    _, proba_lr, auc_lr = evaluate("Logistic Regression (baseline)", logreg, X_test, y_test)

    # ---- Model 2: Random Forest ----
    rf = build_pipeline(RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    ))
    rf.fit(X_train, y_train)
    pred_rf, proba_rf, auc_rf = evaluate("Random Forest", rf, X_test, y_test)

    # ---- ROC curve comparison ----
    plt.figure(figsize=(7, 6))
    for name, proba, auc in [("Logistic Regression", proba_lr, auc_lr),
                              ("Random Forest", proba_rf, auc_rf)]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Flight Delay Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "roc_curve.png", dpi=140)
    plt.close()

    # ---- Confusion matrix (Random Forest) ----
    cm = confusion_matrix(y_test, pred_rf)
    disp = ConfusionMatrixDisplay(cm, display_labels=["On-time", "Delayed"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Random Forest - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=140)
    plt.close()

    # ---- Feature importance (business insight payoff) ----
    ohe = rf.named_steps["preprocess"].named_transformers_["cat"]
    cat_feature_names = ohe.get_feature_names_out(FEATURES_CAT)
    all_feature_names = np.concatenate([FEATURES_NUM, cat_feature_names])
    importances = rf.named_steps["model"].feature_importances_

    imp_df = pd.DataFrame({"feature": all_feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#2e7d32")
    plt.title("Top 15 Feature Importances - Random Forest")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUT / "feature_importance.png", dpi=140)
    plt.close()

    print("Top 10 most important features:")
    print(imp_df.head(10).to_string(index=False))
    print("\nSaved roc_curve.png, confusion_matrix.png, feature_importance.png to outputs/")


if __name__ == "__main__":
    main()
