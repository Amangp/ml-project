def run_svm():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report , confusion_matrix
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import SMOTE
    
    df = pd.read_csv("Data/labeled_data.csv")

    #  Load data
    df = pd.read_csv("Data/labeled_data.csv")
    df["value_ratio"] = df["Net Value"] / (df["Total Value"] + 1)
    df["fee_to_value"] = df["TxnFee(ETH)"] / (df["Total Value"] + 1)
    df["is_high_fee"] = (df["Fee Ratio"] > df["Fee Ratio"].mean()).astype(int)

    features = [
        "Total Value_z",
        "Net Value_z",
        "Fee Ratio_z",
        "Time Gap_z",
        "Block Gap_z",
        "value_ratio",
        "fee_to_value",
        "is_high_fee"
    ]

    X = df[features]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    model = SVC(C=1, kernel='rbf', gamma='scale', class_weight='balanced')
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\n===== SVM =====")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))
if __name__ == "__main__":
    run_svm()