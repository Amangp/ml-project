import pandas as pd
import numpy as np

# Load dataset containing Isolation Forest scores
df = pd.read_csv("Data/if_scored_ethereum.csv")

epsilon = 1e-9

# Statistical anomaly score
df["StatScore"] = (
    abs(df["Total Value"]) +
    abs(df["Fee Ratio"]) +
    abs(df["Time Gap"])
)

df["StatScore"] = (
    df["StatScore"] - df["StatScore"].min()
) / (
    df["StatScore"].max() - df["StatScore"].min() + epsilon
)

# Temporal anomaly score
rolling_mean = df["Total Value"].rolling(window=20).mean()

df["TempScore"] = abs(df["Total Value"] - rolling_mean) / (rolling_mean + epsilon)
df["TempScore"] = df["TempScore"].fillna(0)

df["TempScore"] = (
    df["TempScore"] - df["TempScore"].min()
) / (
    df["TempScore"].max() - df["TempScore"].min() + epsilon
)

# MF-UFS composite score
w_if = 0.5
w_stat = 0.3
w_temp = 0.2

df["FinalScore"] = (
    w_if * df["IF_Score"] +
    w_stat * df["StatScore"] +
    w_temp * df["TempScore"]
)

# Fraud flag
threshold = df["FinalScore"].quantile(0.98)

df["FraudFlag"] = (df["FinalScore"] > threshold).astype(int)

# Save results
df.to_csv("Data/final_ethereum_fraud_analysis.csv", index=False)

print("MF-UFS scoring complete")
print("Suspicious transactions:", df["FraudFlag"].sum())