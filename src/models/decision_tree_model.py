# src/models/decision_tree_model.py
import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, works in headless/server environments
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE


def run_decision_tree():
    print("\n===== DECISION TREE =====")

    # Clean 70k labeled dataset produced by create_labels.py
    df = pd.read_csv("Data/labeled_data.csv")

    # Derived features: value_ratio compares transaction value to execution cost
    # is_high_value flags transactions above the dataset average
    df["value_ratio"]    = df["Value"] / (df["GasCost"] + 1)
    df["gas_efficiency"] = df["GasEfficiency"]
    df["is_high_value"]  = (df["Value"] > df["Value"].mean()).astype(int)

    # Same 13 features as every other model so results are directly comparable
    # IF_Score, StatScore, TempScore are anomaly signals from MF-UFS
    # from_scam and to_scam are direct fraud signals from the original dataset
    features = [
        "Value_z", "GasCost_z", "GasEfficiency_z", "TimeGap_z", "BlockGap_z",
        "value_ratio", "gas_efficiency", "is_high_value",
        "from_scam", "to_scam"
    ]

    X = df[features].fillna(0)
    y = df["label"]

    # stratify=y preserves the 85/15 class ratio in both train and test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale BEFORE SMOTE so scaler fits on real data only
    # Decision Trees don't need scaling but we keep it consistent for fair comparison
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # SMOTE after scaling so synthetic fraud samples are created in normalized space
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # max_depth=8 prevents the tree from memorising SMOTE-augmented training data
    # min_samples controls ensure each split has enough data to be statistically meaningful
    # class_weight=balanced penalizes missing a fraud case more than a false alarm
    # No class_weight here because SMOTE already balanced the training set to 50/50
    # Using class_weight=balanced after SMOTE double-penalizes and hurts precision
    model = DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, preds):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    # Feature importances show which signals the tree relied on most
    print("\nFeature Importances:")
    for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True):
        print(f"  {feat}: {imp:.4f}")

    # Save trained model so app.py can load it for the Streamlit demo
    os.makedirs("Models", exist_ok=True)
    joblib.dump(model, "Models/decision_tree.pkl")

    # Save top 3 levels of the tree as PNG — deep enough to show logic, readable enough to interpret
    # plt.show() removed — it blocks execution in server and headless environments
    plt.figure(figsize=(20, 10))
    plot_tree(model, max_depth=3, feature_names=features, class_names=["Normal", "Fraud"], filled=True)
    plt.tight_layout()
    plt.savefig("Models/decision_tree_plot.png", dpi=150)
    plt.close()

    print("\nModel saved to Models/decision_tree.pkl")
    print("Tree plot saved to Models/decision_tree_plot.png")

    return model


if __name__ == "__main__":
    run_decision_tree()