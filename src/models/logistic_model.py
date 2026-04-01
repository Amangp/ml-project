def run_logistic():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv("Data/labeled_data.csv")

    features = [
        "Total Value_z",
        "Net Value_z",
        "Fee Ratio_z",
        "Time Gap_z",
        "Block Gap_z"
    ]

    X = df[features]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,stratify=y
    )
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(class_weight='balanced',max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print(classification_report(y_test, preds))
if __name__ == "__main__":
    run_logistic()