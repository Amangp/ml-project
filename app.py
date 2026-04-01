import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Ethereum Fraud Detector", layout="wide")

# ── Load & train models ─────────────────────────
@st.cache_resource
def load_models():
    df = pd.read_csv("Data/labeled_data.csv")

    features = ["Total Value_z", "Net Value_z", "Fee Ratio_z", "Time Gap_z", "Block Gap_z"]
    X = df[features]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "SVM": SVC(class_weight="balanced", probability=True),
    }

    for m in models.values():
        m.fit(X_train.values, y_train)

    model_test_results = {}
    for name, m in models.items():
        preds = m.predict(X_test.values)
        report = classification_report(y_test, preds, output_dict=True)
        cm = confusion_matrix(y_test, preds)
        model_test_results[name] = {"report": report, "cm": cm}

    stats = df[["Total Value", "Net Value", "Fee Ratio", "Time Gap", "Block Gap"]].agg(["mean", "std"])

    return models, stats, model_test_results


# ── MF-UFS Score ─────────────────────────
def compute_mfufs(row_raw, stats):
    epsilon = 1e-9
    z = {}

    for f in row_raw:
        mean = stats[f]["mean"]
        std = stats[f]["std"]
        z[f] = (row_raw[f] - mean) / (std + epsilon)

    stat_score = (abs(z["Total Value"]) + abs(z["Fee Ratio"]) + abs(z["Time Gap"])) / 10
    stat_score = min(stat_score, 1)

    return stat_score, z


models, stats, model_test_results = load_models()

# ── UI ─────────────────────────
st.title("Ethereum Fraud Detector")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Enter Transaction Details")

    value_in = st.number_input(
    "Value IN (ETH)",
    min_value=0.0,
    value=0.0,
    step=1e-12,
    format="%.12f"
    )

    value_out = st.number_input(
        "Value OUT (ETH)",
        min_value=0.0,
        value=0.0,
        step=1e-12,
        format="%.12f"
    )

    fee = st.number_input(
        "Fee (ETH)",
        min_value=0.0,
        value=0.0,
        step=1e-12,
        format="%.12f"
    )
    time_gap = st.number_input("Time Gap", 0.0)
    block_gap = st.number_input("Block Gap", 0.0)

with col2:
    st.subheader("Analysis")

    if st.button("Analyze"):

        total_value = value_in + value_out
        net_value = value_in - value_out
        fee_ratio = fee / (total_value + 1e-9)

        raw = {
            "Total Value": total_value,
            "Net Value": net_value,
            "Fee Ratio": fee_ratio,
            "Time Gap": time_gap,
            "Block Gap": block_gap,
        }

        score, z = compute_mfufs(raw, stats)

       
        z_input = np.array([[
            z["Total Value"],
            z["Net Value"],
            z["Fee Ratio"],
            z["Time Gap"],
            z["Block Gap"],
        ]])

        st.metric("Risk Score", f"{int(score * 100)}%")

        results = {}
        for name, model in models.items():
            pred = model.predict(z_input)[0]
            prob = model.predict_proba(z_input)[0][1] if hasattr(model, "predict_proba") else None
            results[name] = (pred, prob)

        st.subheader("Model Votes")

        for name, (pred, prob) in results.items():
            label = "Fraud" if pred == 1 else "Normal"
            prob_txt = f" ({prob*100:.1f}%)" if prob is not None else ""
            st.write(f"{name}: {label}{prob_txt}")

        st.subheader("Z-Scores")

        for k, v in z.items():
            if abs(v) > 3:
                alert = "🚨"
            elif abs(v) > 2:
                alert = "⚠️"
            else:
                alert = ""
            st.write(f"{k}: {round(v,3)} {alert}")