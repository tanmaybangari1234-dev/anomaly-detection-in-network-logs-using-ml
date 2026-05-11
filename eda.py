# ==========================================
# EDA ANALYSIS 
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# -------------------------------
# 1. LOAD DATASET
# -------------------------------
df = pd.read_excel("UNSW_NB15_training-set.xlsx")
print("Original Shape:", df.shape)
print(df.head())
# -------------------------------
# 2. BASIC CLEANING
# -------------------------------
drop_cols = ['id', 'attack_cat']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])
print("After basic cleaning:", df.shape)
# -------------------------------
# 3. DOMAIN-BASED FEATURE REMOVAL
# -------------------------------
manual_drop = [
    'sbytes', 'dbytes',        # dependent
    'sload', 'dload',          # derived
    'sloss', 'dloss'           # redundant
]
df = df.drop(columns=[col for col in manual_drop if col in df.columns])
print("After domain removal:", df.shape)
# -------------------------------
# 4. ATTACK vs NORMAL DISTRIBUTION
# -------------------------------
plt.figure(figsize=(6, 4))
df['label'].value_counts().plot(kind='bar')
plt.title("Normal vs Attack Traffic")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks([0, 1], ["Normal", "Attack"])
plt.tight_layout()
plt.savefig("attack_distribution.png")
plt.show()
# -------------------------------
# 5. CONNECTION DURATION
# -------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df['dur'], bins=50)
plt.title("Connection Duration Distribution")
plt.xlabel("Duration")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("duration_distribution.png")
plt.show()
# -------------------------------
# 6. SOURCE PACKETS DISTRIBUTION
# -------------------------------
if 'spkts' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df['spkts'], bins=50)
    plt.title("Source Packets Distribution")
    plt.xlabel("Source Packets")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("source_packets_distribution.png")
    plt.show()
# -------------------------------
# 7. ATTACK vs NORMAL COMPARISON
# -------------------------------
if 'spkts' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='label', y='spkts', data=df)
    plt.title("Source Packets vs Traffic Type")
    plt.xlabel("Traffic Type")
    plt.ylabel("Source Packets")
    plt.xticks([0, 1], ["Normal", "Attack"])
    plt.tight_layout()
    plt.savefig("attack_vs_normal_packets.png")
    plt.show()
# -------------------------------
# 8. REMOVE HIGHLY CORRELATED FEATURES
# -------------------------------
numeric_df_initial = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df_initial.corr().abs()
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
threshold = 0.85
to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > threshold)]
print("\nHighly correlated columns dropped:")
print(to_drop)
df_reduced = df.drop(columns=to_drop)
print("Final Shape after correlation filtering:", df_reduced.shape)
# -------------------------------
# 9. CLEAN CORRELATION HEATMAP
# -------------------------------
numeric_df = df_reduced.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
corr = numeric_df.corr()
sns.heatmap(corr, cmap="coolwarm")
plt.title("Optimized Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("optimized_correlation_heatmap.png")
plt.show()
# -------------------------------
# 10. TOP FEATURES HEATMAP (CLIENT FRIENDLY)
# -------------------------------
if 'label' in numeric_df.columns:
    target_corr = numeric_df.corr()['label'].abs().sort_values(ascending=False)
    top_features = target_corr[1:15].index
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df[top_features].corr(), cmap="coolwarm")
    plt.title("Top Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("top_features_heatmap.png")
    plt.show()
# -------------------------------
# 11. SAVE FINAL DATASET
# -------------------------------
df_reduced.to_excel("UNSW_NB15_reduced_features.xlsx", index=False)
# -------------------------------
# 12. FINAL LOG
# -------------------------------
print("\nEDA Completed Successfully!")
print("Final Feature Count:", len(df_reduced.columns))
print("Final Features:\n", df_reduced.columns.tolist())