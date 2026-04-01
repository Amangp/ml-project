# main.py

print("\n===== STARTING FULL PIPELINE =====\n")

print("1. Running Feature Engineering...")
import src.preprocessing.feature_engineering

print("2. Running Isolation Forest...")
import src.models.isolation_forest

print("3. Running MF-UFS Algorithm...")
import src.models.mfufs

print("4. Creating Labels...")
import src.evaluation.create_labels

print("5. Running Logistic Regression...")
import src.models.logistic_model

print("6. Running SVM...")
import src.models.svm_model

print("7. Running KNN...")
import src.models.knn_model

print("\n===== PIPELINE COMPLETED SUCCESSFULLY =====\n")