def run_knn():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import SMOTE

    print("\n===== KNN (k=5, distance-weighted) =====")

    # Clean 70k labeled dataset produced by create_labels.py
    df = pd.read_csv("Data/labeled_data.csv")

    # Derived features computed here because they are model-specific
    # value_ratio captures transaction value relative to its execution cost
    # is_high_value flags transactions above the dataset average
    df["value_ratio"]    = df["Value"] / (df["GasCost"] + 1)
    df["gas_efficiency"] = df["GasEfficiency"]
    df["is_high_value"]  = (df["Value"] > df["Value"].mean()).astype(int)

    # Scaling is mandatory for KNN — at prediction time it measures Euclidean distance
    # from each new point to every training point, so unscaled features break it entirely
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

    # Scale BEFORE SMOTE — SMOTE internally uses its own k-NN search to generate
    # synthetic points, so if data is unscaled, SMOTE's neighbour search is also broken
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # SMOTE after scaling so synthetic fraud samples are created in normalized space
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # weights=distance means closer neighbours vote more strongly than distant ones
    # this is better than uniform for fraud detection where very close matches are strong evidence
    model = KNeighborsClassifier(n_neighbors=5, weights="distance")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    run_knn()