import pandas as pd
import numpy as np

df = pd.read_csv("Data/if_scored.csv")

# -----------------------------
# StatScore
# -----------------------------
df["StatScore"] = (
    abs(df["Value_z"]) +
    abs(df["GasEfficiency_z"]) +
    abs(df["TimeGap_z"]) +
    abs(df["GasCost_z"])
)

# -----------------------------
# TempScore
# -----------------------------
rolling_mean = df["Value"].rolling(20).mean()
epsilon = 1e-9

df["TempScore"] = abs(df["Value"] - rolling_mean) / (rolling_mean + epsilon)

# Normalize
df["StatScore"] = (df["StatScore"] - df["StatScore"].min()) / (df["StatScore"].max() - df["StatScore"].min())
df["TempScore"] = (df["TempScore"] - df["TempScore"].min()) / (df["TempScore"].max() - df["TempScore"].min())

# -----------------------------
# Final Score
# -----------------------------
df["FinalScore"] = (
    0.3 * df["IF_Score"] +
    0.4 * df["StatScore"] +
    0.3 * df["TempScore"]
)

# Threshold
threshold = df["FinalScore"].quantile(0.85)

df["FraudFlag"] = (df["FinalScore"] > threshold).astype(int)

df.to_csv("Data/final_output.csv", index=False)

print(" MF-UFS completed")

