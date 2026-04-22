import subprocess
import sys

print("\n===== STARTING FULL PIPELINE =====\n")

# We run each pipeline step as a subprocess instead of a bare import
# With bare imports, Python caches the module after the first run
# so if any step crashes and you re-run main.py, that step silently does nothing
# subprocess.run forces every step to execute fresh every time
# check=True means the pipeline stops immediately if any step fails with a clear error message

print("1. Running Feature Engineering...")
subprocess.run([sys.executable, "src/preprocessing/feature_engineering.py"], check=True)

# Isolation Forest scores each transaction for anomalousness — contamination=0.15
# because our dataset has roughly 15% fraud transactions
print("2. Running Isolation Forest...")
subprocess.run([sys.executable, "src/models/isolation_forest.py"], check=True)

# MF-UFS combines IF_Score + StatScore + TempScore into FinalScore
# and sets FraudFlag=1 for transactions in the top 15% of FinalScore
print("3. Running MF-UFS Algorithm...")
subprocess.run([sys.executable, "src/models/mfufs.py"], check=True)

# create_labels reads final_output.csv, drops raw blockchain columns,
# and writes labeled_data.csv which all supervised models below read from
print("4. Creating Labels...")
subprocess.run([sys.executable, "src/evaluation/create_labels.py"], check=True)

# All five supervised models below read from labeled_data.csv
# They use only raw transaction features — not the MF-UFS scores —
# so they learn independently from the unsupervised labeler
from src.models.logistic_model import run_logistic
from src.models.svm_model import run_svm
from src.models.knn_model import run_knn
from src.models.decision_tree_model import run_decision_tree
from src.models.random_forest import run_random_forest

print("5. Running Logistic Regression...")
run_logistic()

print("6. Running SVM...")
run_svm()

print("7. Running KNN...")
run_knn()

print("8. Running Decision Tree...")
run_decision_tree()

# Random Forest is a Bagging ensemble — 200 decision trees trained on bootstrap samples
# Added to cover Unit III ensemble methods in CS1138 syllabus
print("9. Running Random Forest...")
run_random_forest()

print("\n===== PIPELINE COMPLETED SUCCESSFULLY =====\n")