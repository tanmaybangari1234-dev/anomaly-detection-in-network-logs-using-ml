# =====================================
# 1. IMPORT LIBRARIES
# =====================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
# =====================================
# 2. LOAD DATASET
# =====================================
df = pd.read_excel("UNSW_NB15_training-set.xlsx")
print("Original Shape:", df.shape)
# =====================================
# 3. HANDLE MISSING VALUES
# =====================================
df.replace("-", np.nan, inplace=True)
df.replace("", np.nan, inplace=True)
#to find categorical values
categorical_cols = ['proto', 'service', 'state']
for col in categorical_cols:
    df[col] = df[col].fillna("unknown")
#to find numerical values
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
# =====================================
# 4. DROP UNNECESSARY COLUMNS
# =====================================
original_cols = df.columns.tolist()
if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)
if 'attack_cat' in df.columns:
    df.drop(columns=['attack_cat'], inplace=True)
# after drop
final_cols = df.columns.tolist()
# find removed columns
removed_cols = list(set(original_cols) - set(final_cols))
print("Removed Columns:", removed_cols)
# =====================================
# 5. ENCODE CATEGORICAL DATA
# =====================================
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le
# =====================================
# 6. SPLIT FEATURES AND LABEL
# =====================================
X = df.drop("label", axis=1)
y = df["label"]
# =====================================
# 7. FEATURE SCALING
# =====================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# =====================================
# 8. SAVE PREPROCESSING OBJECTS
# =====================================
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_dict, "label_encoders.pkl")
print("Preprocessing Completed Successfully!")
print("Final Shape:", X_scaled.shape)
# 9. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y)
print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)