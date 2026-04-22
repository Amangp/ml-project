import pandas as pd

df = pd.read_csv("Data/final_output.csv", low_memory=False)

# Assign supervised label from MF-UFS FraudFlag
df["label"] = df["FraudFlag"]

# Keep only columns needed for model training
# Drops raw blockchain fields (hash, addresses, input data etc.)
keep_cols = [
    "Value",
    "GasCost",
    "GasEfficiency",
    "TimeGap",
    "BlockGap",
    "Value_z",
    "GasCost_z",
    "GasEfficiency_z",
    "TimeGap_z",
    "BlockGap_z",
    "IF_Score",
    "StatScore",
    "TempScore",
    "FinalScore",
    "FraudFlag",
    "label",
]

df = df[keep_cols]

df.to_csv("Data/labeled_data.csv", index=False)

print(" Labels created from MF-UFS")
print(f" Total rows    : {len(df)}")
print(f" Normal  (0)   : {(df['label'] == 0).sum()}")
print(f" Fraud   (1)   : {(df['label'] == 1).sum()}")
print(f" Fraud rate    : {df['label'].mean()*100:.2f}%")
