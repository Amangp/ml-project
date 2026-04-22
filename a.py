import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("Data/new dataset/labeled_data.csv")

# -----------------------------
# FEATURES (NO LEAKAGE)
# -----------------------------
features = [
    "Value_z",
    "GasCost_z",
    "GasEfficiency_z",
    "TimeGap_z",
    "BlockGap_z"
]

X = df[features].fillna(0)

# ✅ FIX: use correct target
y = df["FraudFlag"]

# -----------------------------
# SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# SCALING (for KNN, SVM, Logistic)
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# MODELS
# -----------------------------
models = {
    "Logistic": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42)
}

# -----------------------------
# PLOT ROC
# -----------------------------
plt.figure()

for name, model in models.items():

    # Train + Predict Probabilities
    if name in ["Logistic", "KNN", "SVM"]:
        model.fit(X_train_scaled, y_train)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

# -----------------------------
# FINAL GRAPH
# -----------------------------
plt.plot([0, 1], [0, 1], linestyle="--")  # baseline

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison (All 5 Models)")
plt.legend()
plt.grid()

plt.show()