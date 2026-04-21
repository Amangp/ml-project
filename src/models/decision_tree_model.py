import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

def run_decision_tree():
    print("\n===== DECISION TREE MODEL (on final_output.csv) =====")
    
    df = pd.read_csv("Data/final_output.csv")
    
    if all(col in df.columns for col in ["Value_z", "GasCost_z", "GasEfficiency_z", "TimeGap_z"]):
        features = ["Value_z", "GasCost_z", "GasEfficiency_z", "TimeGap_z"]
        print("Using z-scored features.")
    else:
        features = ["Value", "GasCost", "GasEfficiency", "TimeGap"]
        print("Using raw features.")
    
    X = df[features]
    y = df["FraudFlag"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # TODO: Add model training and evaluation

if _name_ == "_main_":
    run_decision_tree()