def run_logistic():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import SMOTE

    print("\n===== LOGISTIC REGRESSION =====")

    df = pd.read_csv("Data/labeled_data.csv")

    # Derived features computed here because they are model-specific
    # value_ratio captures how large a transaction is relative to its execution cost
    # is_high_value flags transactions above the dataset average
    df["value_ratio"]    = df["Value"] / (df["GasCost"] + 1)
    df["gas_efficiency"] = df["GasEfficiency"]
    df["is_high_value"]  = (df["Value"] > df["Value"].mean()).astype(int)

    # We exclude IF_Score, StatScore, and TempScore as they are directly used to compute FinalScore which creates the label
    # FinalScore = 0.3*IF_Score + 0.4*StatScore + 0.3*TempScore, label = FinalScore > 85th percentile
    # Including them would mean the model is just learning to reconstruct that formula
    # not learning actual fraud patterns from the raw transaction features
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

    # Scale BEFORE SMOTE so the scaler fits on real data only
    # If we scaled after SMOTE, synthetic samples would corrupt the mean and std
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # SMOTE generates synthetic fraud samples by interpolating between existing ones
    # Applied after scaling so synthetic points are created in normalized space
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # No class_weight here because SMOTE already balanced the training set to 50/50
    # Using class_weight=balanced on top of SMOTE double-penalizes the majority class
    # which inflates recall but hurts precision — one correction is enough
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    run_logistic()