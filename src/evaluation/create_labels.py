import pandas as pd

df = pd.read_csv("Data/new dataset/final_output.csv", low_memory=False)


# -----------------------------
# FINAL CLEAN DATASET
# -----------------------------
df_final = df[[
    "Value_z",
    "GasCost_z",
    "GasEfficiency_z",
    "TimeGap_z",
    "BlockGap_z",
    "FraudFlag"
]]

df_final.to_csv("Data/new dataset/labeled_data.csv", index=False)

print(" Clean labeled dataset created")
print(f" Total rows    : {len(df_final)}")
print(f" Fraud rate    : {df_final['FraudFlag'].mean()*100:.2f}%")