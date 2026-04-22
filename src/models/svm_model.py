def run_svm():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import SMOTE

    print("\n===== SVM (RBF Kernel) =====")

    # Clean 70k labeled dataset produced by create_labels.py
    df = pd.read_csv("Data/labeled_data.csv")

    # Derived features computed here because they are model-specific
    # value_ratio captures transaction value relative to its execution cost
    # is_high_value flags transactions above the dataset average
    df["value_ratio"]    = df["Value"] / (df["GasCost"] + 1)
    df["gas_efficiency"] = df["GasEfficiency"]
    df["is_high_value"]  = (df["Value"] > df["Value"].mean()).astype(int)

    # Z-scored base features are mandatory for SVM
    # RBF kernel computes distances in feature space, so unscaled features
    # where Value is in millions completely drown out everything else
    # We deliberately exclude IF_Score, StatScore, and TempScore here
    # Those three scores directly compute FinalScore which creates the label
    # The supervised model must learn from raw features independently
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

    # Scale BEFORE SMOTE — this is the correct order
    # If SMOTE ran first, synthetic samples would be created in unscaled space
    # where Value dominates distances and all other features are effectively ignored
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # SMOTE after scaling so all 13 features contribute equally to interpolation
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # RBF kernel handles non-linear fraud patterns well
    # No class_weight here because SMOTE already balanced the training set to 50/50
    # Using class_weight=balanced after SMOTE double-penalizes and hurts precision
    model = SVC(C=1, kernel="rbf", gamma="scale", random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    run_svm()