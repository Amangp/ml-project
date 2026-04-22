from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# -----------------------------
# LOAD MODELS
# -----------------------------
rf = joblib.load("Models/random_forest.pkl")
dt = joblib.load("Models/decision_tree.pkl")
lr = joblib.load("Models/logistic.pkl")

# -----------------------------
# LOAD SCALERS
# -----------------------------
lr_scaler  = joblib.load("Models/logistic_scaler.pkl")

# -----------------------------
# DATASET STATS
# -----------------------------
STATS = {
    "Value": (0.1823, 2.8741),
    "GasCost": (0.00412, 0.00831),
    "GasEfficiency": (0.8912, 0.1503),
    "TimeGap": (14.21, 38.92),
    "BlockGap": (1.14, 2.87),
}

def z(x, mean, std):
    return (x - mean) / (std + 1e-9)

# -----------------------------
# PREDICT API
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    print("POST REQUEST HIT")
    data = request.json

    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------
    Value = data["val_in"] + data["val_out"]
    GasCost = data["fee"]
    GasEfficiency = data["gas_used"] / (data["gas"] + 1e-9)

    Value_z = z(Value, *STATS["Value"])
    GasCost_z = z(GasCost, *STATS["GasCost"])
    GasEfficiency_z = z(GasEfficiency, *STATS["GasEfficiency"])
    TimeGap_z = z(data["time_gap"], *STATS["TimeGap"])
    BlockGap_z = z(data["block_gap"], *STATS["BlockGap"])

    X = np.array([[Value_z, GasCost_z, GasEfficiency_z, TimeGap_z, BlockGap_z]])

    # -----------------------------
    # SCALE FOR MODELS
    # -----------------------------
    X_scaled_lr  = lr_scaler.transform(X)

    # -----------------------------
    # MODEL PREDICTIONS
    # -----------------------------
    results = {
        "Decision Tree": float(dt.predict_proba(X)[0][1]),
        "Random Forest": float(rf.predict_proba(X)[0][1]),
        "Logistic": float(lr.predict_proba(X_scaled_lr)[0][1]),
    }

    # -----------------------------
    # EXTRA DATA FOR UI
    # -----------------------------
    zscores = {
        "Value": float(Value_z),
        "GasCost": float(GasCost_z),
        "GasEfficiency": float(GasEfficiency_z),
        "TimeGap": float(TimeGap_z),
        "BlockGap": float(BlockGap_z),
    }

    IF_Score = abs(Value_z)*0.2 + abs(GasCost_z)*0.2
    StatScore = abs(Value_z)+abs(GasCost_z)
    TempScore = abs(TimeGap_z)

    IF_Score = min(IF_Score, 1)
    StatScore = min(StatScore/5, 1)
    TempScore = min(TempScore/5, 1)

    FinalScore = 0.3*IF_Score + 0.4*StatScore + 0.3*TempScore

    derivedFeatures = {
        "Value": float(Value),
        "GasCost": float(GasCost),
        "GasEfficiency": float(GasEfficiency),
    }

    # -----------------------------
    # RESPONSE
    # -----------------------------
    return jsonify({
        "modelProbs": results,
        "FinalScore": float(np.mean(list(results.values()))),

        "IF_Score": IF_Score,
        "StatScore": StatScore,
        "TempScore": TempScore,
        "zscores": zscores,
        "derivedFeatures": derivedFeatures
    })


if __name__ == "__main__":
    app.run(debug=True)