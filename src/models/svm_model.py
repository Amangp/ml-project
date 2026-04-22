def run_svm():
    import pandas as pd
    import joblib
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
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

    print("\n===== SVM (Final Version) =====")

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    df = pd.read_csv("Data/new dataset/labeled_data.csv")

    # -----------------------------
    # FEATURES (NO LEAKAGE)
    # -----------------------------
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
    # SCALING
    # -----------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -----------------------------
    # SMOTE (balanced but not overfit)
    # -----------------------------
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # -----------------------------
    # MODEL (TUNED SLIGHTLY)
    # -----------------------------
    model = SVC(
        C=2,              
        kernel="rbf",
        gamma="scale",
        probability=True,
        random_state=42
    )

    model.fit(X_train, y_train)

    # -----------------------------
    # PREDICTION (WITH THRESHOLD)
    # -----------------------------
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.45).astype(int)

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
    joblib.dump(model, "Models/svm.pkl")
    joblib.dump(scaler, "Models/svm_scaler.pkl")

    print("\n Model + Scaler saved")


if __name__ == "__main__":
    run_svm()