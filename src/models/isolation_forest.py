import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# 1️⃣ Load processed data
df = pd.read_csv(r"Data/processed_ethereum.csv")

print("Data Loaded:", df.shape)

# 2️⃣ Select features
features = [
    "Total Value_z",
    "Net Value_z",
    "Fee Ratio_z",
    "Time Gap_z",
    "Block Gap_z"
]

X = df[features].fillna(0)

# 3️⃣ SCALE FEATURES
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ Train Isolation Forest
model = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42
)

model.fit(X_scaled)

print("Isolation Forest Trained")

# 5️⃣ Generate anomaly score
raw_scores = -model.decision_function(X_scaled)

# Normalize to 0–1
df["IF_Score"] = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())

# 6️⃣ Generate anomaly label (-1, 1)
df["anomaly_label"] = model.predict(X_scaled)

# Convert to 0/1
df["anomaly_label"] = df["anomaly_label"].apply(lambda x: 1 if x == -1 else 0)

# 7️⃣ Save file
df.to_csv(r"Data/if_scored_ethereum.csv", index=False)

print("IF Scores & labels saved successfully")