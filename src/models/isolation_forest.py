import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.read_csv("Data/processed_data.csv")

features = [
    "Value_z",
    "GasCost_z",
    "GasEfficiency_z",
    "TimeGap_z",
    "BlockGap_z"
]

X = df[features]

model = IsolationForest(n_estimators=200, contamination=0.15, random_state=42)
model.fit(X)

scores = model.decision_function(X)

# Normalize
df["IF_Score"] = (scores.max() - scores) / (scores.max() - scores.min())

df.to_csv("Data/if_scored.csv", index=False)

print(" Isolation Forest done")