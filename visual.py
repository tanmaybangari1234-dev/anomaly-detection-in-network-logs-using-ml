# ============================================
# FINAL MODEL 
# ============================================
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve)
from sklearn.utils import resample
print("\n===== FINAL MODEL STARTED =====")
# ============================================
# STEP 1: LOAD DATA
# ============================================
train_df = pd.read_excel("UNSW_NB15_training-set.xlsx")
test_df = pd.read_excel("UNSW_NB15_testing-set.xlsx")
# ============================================
# STEP 2: DROP UNUSED COLUMNS
# ============================================
drop_cols = ['id', 'attack_cat']
train_df.drop(columns=drop_cols, inplace=True, errors='ignore')
test_df.drop(columns=drop_cols, inplace=True, errors='ignore')
# ============================================
# STEP 3: ENCODE CATEGORICAL FEATURES
# ============================================
categorical_cols = train_df.select_dtypes(include=['object', 'string']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([
        train_df[col].astype(str),
        test_df[col].astype(str)    ])
    le.fit(combined)
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
    label_encoders[col] = le
print("Encoding completed")
# ============================================
# STEP 4: SPLIT FEATURES & LABEL
# ============================================
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']
# ============================================
# STEP 5: BALANCE DATA
# ============================================
train_data = pd.concat([X_train, y_train], axis=1)
normal = train_data[train_data.label == 0]
attack = train_data[train_data.label == 1]
attack_upsampled = resample(
    attack,
    replace=True,
    n_samples=len(normal),
    random_state=42)
train_balanced = pd.concat([normal, attack_upsampled])
train_balanced = train_balanced.sample(frac=1, random_state=42)
X_train = train_balanced.drop('label', axis=1)
y_train = train_balanced['label']
print("Training data balanced")
# ============================================
# STEP 6: SCALING
# ============================================
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns)
# ============================================
# STEP 7: TRAIN MODEL
# ============================================
model = RandomForestClassifier(
    n_estimators=150,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1)
model.fit(X_train_scaled, y_train)
print("Model trained successfully")
# ============================================
# STEP 8: SAVE FILES (FOR GUI / DASHBOARD)
# ============================================
joblib.dump(model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X_train.columns), "features.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
print("Model + scaler + features saved")
# ============================================
# STEP 9: PREDICTIONS
# ============================================
y_pred = model.predict(X_test_scaled)
# IMPORTANT FIX: USE PROBABILITIES
y_prob = model.predict_proba(X_test_scaled)[:, 1]
# ============================================
# STEP 10: EVALUATION
# ============================================
print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))
# CORRECT ROC-AUC
roc = roc_auc_score(y_test, y_prob)
print("\nROC-AUC Score:", roc)
# ============================================
# STEP 11: ROC CURVE PLOT (CORRECT)
# ============================================
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Random Forest)")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()
print("ROC curve saved as roc_curve.png")
# ============================================
# STEP 12: ANALYSIS
# ============================================
print("\n===== ANALYSIS =====")
if roc > 0.90:
    print("Excellent performance")
elif roc > 0.80:
    print("Good performance")
else:
    print("Needs improvement")
print("\n===== FINAL MODEL COMPLETED =====")