import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda():

    print("\n===== STARTING EDA =====\n")

    df = pd.read_csv("Data/labeled_data.csv")

    print("Dataset Shape:", df.shape)
    print("\nColumns:\n", df.columns)

    print("\n===== BASIC INFO =====")
    print(df.info())

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    print("\n===== STATISTICAL SUMMARY =====")
    print(df.describe())

    # Label Distribution
    print("\n===== LABEL DISTRIBUTION =====")
    print(df["label"].value_counts())

    plt.figure()
    df["label"].value_counts().plot(kind='bar')
    plt.title("Class Distribution (Fraud vs Normal)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

    # Feature Engineering for analysis
    df["value_ratio"] = df["Net Value"] / (df["Total Value"] + 1)
    df["fee_to_value"] = df["TxnFee(ETH)"] / (df["Total Value"] + 1)

    # Distributions
    plt.figure()
    plt.hist(df["Total Value"], bins=50)
    plt.title("Distribution of Total Transaction Value")
    plt.xlabel("Total Value")
    plt.ylabel("Frequency")
    plt.show()

    plt.figure()
    plt.boxplot(df["Fee Ratio"])
    plt.title("Fee Ratio Distribution (Outliers Detection)")
    plt.ylabel("Fee Ratio")
    plt.show()

    plt.figure()
    plt.hist(df["Time Gap"], bins=50)
    plt.title("Distribution of Time Gaps Between Transactions")
    plt.xlabel("Time Gap")
    plt.ylabel("Frequency")
    plt.show()

    plt.figure()
    plt.hist(df["Block Gap"], bins=50)
    plt.title("Distribution of Block Gaps")
    plt.xlabel("Block Gap")
    plt.ylabel("Frequency")
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['number'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # Feature vs Label Analysis
    plt.figure()
    sns.boxplot(x="label", y="Fee Ratio", data=df)
    plt.title("Fee Ratio vs Fraud")
    plt.show()

    plt.figure()
    sns.boxplot(x="label", y="Total Value", data=df)
    plt.title("Total Value vs Fraud")
    plt.show()

    plt.figure()
    sns.boxplot(x="label", y="value_ratio", data=df)
    plt.title("Value Ratio vs Fraud")
    plt.show()

    plt.figure()
    sns.boxplot(x="label", y="fee_to_value", data=df)
    plt.title("Fee to Value vs Fraud")
    plt.show()

    # Insights
    print("\n===== KEY INSIGHTS =====")

    print("\n1. Class Imbalance:")
    print(" - Dataset is highly imbalanced with very few fraud cases")

    print("\n2. Total Value:")
    print(" - Highly skewed distribution with a few large transactions")

    print("\n3. Fee Ratio:")
    print(" - Presence of extreme outliers indicating anomalies")

    print("\n4. Time Gap and Block Gap:")
    print(" - Irregular patterns suggest burst or unusual activity")

    print("\n5. Feature vs Fraud Behavior:")
    print(" - Fraud transactions show distinct distributions")
    print(" - Ratio-based features help capture anomalies more effectively")

    print("\n===== EDA COMPLETED =====\n")


if __name__ == "__main__":
    run_eda()