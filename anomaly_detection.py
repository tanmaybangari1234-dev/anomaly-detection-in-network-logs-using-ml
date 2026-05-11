# ================================
# ANOMALY DETECTION
# ================================
import training_Set
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# ================================
# Load Data
# ================================
X_train = training_Set.X_train
X_test = training_Set.X_test
y_train = training_Set.y_train
y_test = training_Set.y_test
# ================================
# Scaling (IMPORTANT for SVM)
# ================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# ================================
# Use ONLY NORMAL data for training
# ================================
X_train_normal = X_train[y_train == 0]
print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)
# ================================
# Isolation Forest
# ================================
print("\n===============================")
print("Training Isolation Forest...")
iso_model = IsolationForest(
    n_estimators=100,
    contamination=0.1,   
    random_state=42)
iso_model.fit(X_train_normal)
y_pred_iso = iso_model.predict(X_test)
y_pred_iso = np.where(y_pred_iso == -1, 1, 0)
print("\nIsolation Forest Results")
print(confusion_matrix(y_test, y_pred_iso))
print(classification_report(y_test, y_pred_iso))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_iso))
# ================================
# One-Class SVM
# ================================
print("\n===============================")
print("Training One-Class SVM...")
ocsvm = OneClassSVM(kernel='rbf', gamma='auto')
ocsvm.fit(X_train_normal)
y_pred_svm = ocsvm.predict(X_test)
y_pred_svm = np.where(y_pred_svm == -1, 1, 0)
print("\nOne-Class SVM Results")
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_svm))
print("Week 5 Completed Successfully!")