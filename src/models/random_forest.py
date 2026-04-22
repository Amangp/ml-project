import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE


def run_random_forest():
    print("\n===== RANDOM FOREST =====")

    # Clean 70k labeled dataset produced by create_labels.py
    df = pd.read_csv("Data/labeled_data.csv")

    # value_ratio compares transaction value to execution cost
    # is_high_value flags transactions above the dataset average as potentially suspicious
    df["value_ratio"]    = df["Value"] / (df["GasCost"] + 1)
    df["gas_efficiency"] = df["GasEfficiency"]
    df["is_high_value"]  = (df["Value"] > df["Value"].mean()).astype(int)

    # Same 10 features as all other models so comparisons are fair
    # We deliberately exclude IF_Score, StatScore, and TempScore here
    # Those three scores directly compute FinalScore which creates the label
    # The supervised model must learn from raw features independently
    # from_scam and to_scam are ground truth scam flags that came with the original dataset
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
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # SMOTE after scaling so synthetic fraud samples are created in normalized space
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Random Forest builds 200 independent decision trees using Bagging
    # Each tree is trained on a different random bootstrap sample of the training data
    # and at each split only a random subset of features is considered
    # The final prediction is a majority vote across all 200 trees
    # This is why Random Forest outperforms a single Decision Tree —
    # individual trees overfit but their average does not
    # max_depth=12 is deeper than our single tree because bagging naturally reduces overfitting
    # n_jobs=-1 trains all 200 trees in parallel using all available CPU cores
    # No class_weight here because SMOTE already balanced the training set to 50/50
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, preds):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    # Feature importances are averaged across all 200 trees
    # making them more reliable than a single Decision Tree's importances
    print("\nFeature Importances (averaged across 200 trees):")
    for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True):
        print(f"  {feat}: {imp:.4f}")


if __name__ == "__main__":
    run_random_forest()