import pandas as pd
import numpy as np

df = pd.read_csv("Data/Dataset.csv")
df.columns = df.columns.str.strip()

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------

df["Value"] = df["value"]
df["GasCost"] = df["gas"] * df["gas_price"]

epsilon = 1e-9
df["GasEfficiency"] = df["receipt_gas_used"] / (df["gas"] + epsilon)

# Time features
df["block_timestamp"] = pd.to_datetime(df["block_timestamp"], errors="coerce")
df = df.sort_values("block_timestamp")

df["TimeGap"] = df["block_timestamp"].diff().dt.total_seconds().fillna(0)
df["block_number"] = pd.to_numeric(df["block_number"], errors="coerce")
df["BlockGap"] = df["block_number"].diff().fillna(0)

# -----------------------------
# Z-SCORE NORMALIZATION
# -----------------------------

features = ["Value", "GasCost", "GasEfficiency", "TimeGap", "BlockGap"]

for col in features:
    df[col + "_z"] = (df[col] - df[col].mean()) / (df[col].std() + epsilon)

# -----------------------------
# FINAL CLEAN DATASET
# -----------------------------

final_df = df[[
    "Value_z",
    "GasCost_z",
    "GasEfficiency_z",
    "TimeGap_z",
    "BlockGap_z",
    "label"
]]

final_df.to_csv("Data/processed_data.csv", index=False)

print(" Clean preprocessing complete")