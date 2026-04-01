import pandas as pd

df = pd.read_csv("Data/final_ethereum_fraud_analysis.csv")

# Just rename
df["label"] = df["FraudFlag"]

df.to_csv("Data/labeled_data.csv", index=False)

print("Labels created successfully")