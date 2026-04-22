def run_knn():
    import pandas as pd
    import joblib
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score
    )
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import SMOTE

    print("\n===== KNN (Clean Version) =====")

    # Load clean dataset
    df = pd.read_csv("Data/new dataset/labeled_data.csv")


    features = [
        "Value_z",
        "GasCost_z",
        "GasEfficiency_z",
        "TimeGap_z",
        "BlockGap_z"
    ]

    X = df[features].fillna(0)
    y = df["FraudFlag"]

    # -----------------------------
    # TRAIN-TEST SPLIT
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # SCALING (MANDATORY FOR KNN)
    # -----------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -----------------------------
    # SMOTE (IMPORTANT)
    # -----------------------------
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # -----------------------------
    # MODEL
    # -----------------------------
    model = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance"
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # -----------------------------
    # REPORT
    # -----------------------------
    print("\n Accuracy:", accuracy_score(y_test, preds))
    print("\n Precision:", precision_score(y_test, preds))
    print(" Recall   :", recall_score(y_test, preds))
    print(" F1 Score :", f1_score(y_test, preds))

    print("\n Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("\n Classification Report:")
    print(classification_report(y_test, preds))

    # -----------------------------
    # SAVE MODEL + SCALER
    # -----------------------------
    os.makedirs("Models", exist_ok=True)
    joblib.dump(model, "Models/knn.pkl")
    joblib.dump(scaler, "Models/knn_scaler.pkl")

    print("\n Model + Scaler saved")


if __name__ == "__main__":
    run_knn()