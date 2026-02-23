import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# 1️⃣ Load processed data
df = pd.read_csv(r"Data/processed_ethereum.csv")

print("Data Loaded:", df.shape)

# 2️⃣ Select features (same names as your feature engineering file)
features = [
    "Total Value",
    "Net Value",
    "Fee Ratio",
    "Time Gap",
    "Block Gap"
]

X = df[features]
X = X.fillna(0)

# 3️⃣ Train Isolation Forest
model = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42
)

model.fit(X)

print("Isolation Forest Trained")

# 4️⃣ Generate IF Score
raw_scores = model.decision_function(X)

inv_scores = raw_scores.max() - raw_scores

if_score = (inv_scores - inv_scores.min()) / (inv_scores.max() - inv_scores.min())

df["IF_Score"] = if_score

# 5️⃣ Save new file
df.to_csv(r"Data/if_scored_ethereum.csv", index=False)

print("IF Scores saved successfully")