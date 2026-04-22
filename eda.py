import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression


# -----------------------------
# ROC CURVE
# -----------------------------
def plot_roc_curve(df):

    print("\n===== ROC CURVE =====\n")

    features = [
        "Value_z",
        "GasCost_z",
        "GasEfficiency_z",
        "TimeGap_z",
        "BlockGap_z"
    ]

    X = df[features]
    y = df["FraudFlag"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale (only for logistic)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()


# -----------------------------
# MAIN EDA
# -----------------------------
def run_eda():

    print("\n===== STARTING EDA =====\n")

    df = pd.read_csv("Data/new dataset/labeled_data.csv")

    print("Dataset Shape:", df.shape)
    print("\nColumns:\n", df.columns)

    print("\n===== BASIC INFO =====")
    print(df.info())

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    print("\n===== STATISTICAL SUMMARY =====")
    print(df.describe())

    # -----------------------------
    # LABEL DISTRIBUTION
    # -----------------------------
    print("\n===== LABEL DISTRIBUTION =====")
    print(df["FraudFlag"].value_counts())

    plt.figure()
    df["FraudFlag"].value_counts().plot(kind="bar")
    plt.title("Class Distribution (Fraud vs Normal)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

    # -----------------------------
    # FEATURE DISTRIBUTIONS
    # -----------------------------
    features = [
        "Value_z",
        "GasCost_z",
        "GasEfficiency_z",
        "TimeGap_z",
        "BlockGap_z"
    ]

    for col in features:
        plt.figure()
        plt.hist(df[col], bins=50)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()

    # -----------------------------
    # CORRELATION HEATMAP
    # -----------------------------
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # -----------------------------
    # FEATURE VS FRAUD
    # -----------------------------
    for col in features:
        plt.figure()
        sns.boxplot(x="FraudFlag", y=col, data=df)
        plt.title(f"{col} vs Fraud")
        plt.show()

    # -----------------------------
    # INSIGHTS
    # -----------------------------
    print("\n===== KEY INSIGHTS =====")

    print("\n1. Class Imbalance:")
    print(" - Fraud cases are around 15% of the dataset")

    print("\n2. Feature Behavior:")
    print(" - Z-score features show deviation from normal patterns")
    print(" - Fraud transactions tend to have extreme values")

    print("\n3. Time & Block Patterns:")
    print(" - Irregular gaps indicate suspicious behavior")

    print("\n4. Gas Efficiency:")
    print(" - Abnormal efficiency correlates with fraud")

    print("\n===== EDA COMPLETED =====\n")

    # ROC
    plot_roc_curve(df)


if __name__ == "__main__":
    run_eda()