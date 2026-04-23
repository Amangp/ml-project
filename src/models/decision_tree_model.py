def run_decision_tree():
    import os
    import joblib
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score
    )
    from imblearn.over_sampling import SMOTE

    print("\n===== DECISION TREE (Clean Version) =====")


    # LOAD DATA

    df = pd.read_csv("Data/new dataset/labeled_data.csv")


    # FEATURES

    features = [
        "Value_z",
        "GasCost_z",
        "GasEfficiency_z",
        "TimeGap_z",
        "BlockGap_z"
    ]

    X = df[features].fillna(0)
    y = df["FraudFlag"]


    # TRAIN-TEST SPLIT

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    # SMOTE (NO SCALING NEEDED)

    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)


    # MODEL

    model = DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)


    # REPORT

    print("\n Accuracy:", accuracy_score(y_test, preds))
    print("\n Precision:", precision_score(y_test, preds))
    print(" Recall   :", recall_score(y_test, preds))
    print(" F1 Score :", f1_score(y_test, preds))

    print("\n Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("\n Classification Report:")
    print(classification_report(y_test, preds))


    # FEATURE IMPORTANCE

    print("\nFeature Importances:")
    for feat, imp in sorted(
        zip(features, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  {feat}: {imp:.4f}")


    # SAVE MODEL

    os.makedirs("Models", exist_ok=True)
    joblib.dump(model, "Models/decision_tree.pkl")


    # TREE VISUALIZATION

    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        max_depth=3,
        feature_names=features,
        class_names=["Normal", "Fraud"],
        filled=True
    )
    plt.tight_layout()
    plt.savefig("Models/decision_tree_plot.png", dpi=150)
    plt.show()
    plt.close()

    print("\nModel saved")
    print("Tree plot saved")

    return model


if __name__ == "__main__":
    run_decision_tree()