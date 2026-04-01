import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def evaluate_models():
    print("\n===== MODEL EVALUATION STARTED =====\n")

    # 1️⃣ Load labeled data
    df = pd.read_csv("Data/labeled_data.csv")

    # 2️⃣ Select features
    features = [
        "Total Value",
        "Net Value",
        "Fee Ratio",
        "Time Gap",
        "Block Gap"
    ]

    X = df[features]
    y = df["label"]

    # 3️⃣ Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 4️⃣ Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 5️⃣ Initialize models
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
        "SVM": SVC(class_weight='balanced'),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    results = []

    # 6️⃣ Train and evaluate
    for name, model in models.items():
        print(f"\n===== {name} =====")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        results.append({
            "Model": name,
            "Accuracy": acc
        })

    # 7️⃣ Summary Table
    print("\n===== SUMMARY =====")
    results_df = pd.DataFrame(results)
    print(results_df)


if __name__ == "__main__":
    evaluate_models()