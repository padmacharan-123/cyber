# XGBoost on UNSW-NB15 Dataset + RandomizedSearchCV + Early Stopping + Metrics

import time
import zipfile
from pathlib import Path

# CSV files are inside this zip (train/test CSVs at root of zip).
ZIP_PATH = Path(r"C:\Users\pspad\Downloads\OneDrive_1_11-02-2026.zip")
TRAIN_CSV_IN_ZIP = "UNSW_NB15_training-set.csv"
TEST_CSV_IN_ZIP = "UNSW_NB15_testing-set.csv"
LABEL_COL = "label"
RANDOM_SEED = 42

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

print("="*70)
print("UNSW-NB15 Cybersecurity Dataset - XGBoost Training")
print("="*70)

# -----------------------------
# 1) Load Pre-Split Data
# -----------------------------
print("\n[*] Loading UNSW-NB15 dataset from zip...")
if not ZIP_PATH.exists():
    raise FileNotFoundError(f"Zip not found: {ZIP_PATH}")
with zipfile.ZipFile(ZIP_PATH, "r") as z:
    train_df = pd.read_csv(z.open(TRAIN_CSV_IN_ZIP))
    test_df = pd.read_csv(z.open(TEST_CSV_IN_ZIP))

print(f"[OK] Training set: {train_df.shape}")
print(f"[OK] Testing set: {test_df.shape}")

# -----------------------------
# 2) Prepare Features
# -----------------------------
# Drop columns not needed for prediction
drop_cols = [LABEL_COL, 'id', 'attack_cat']  # id is just index, attack_cat is multiclass (we use binary)

y_train_raw = train_df[LABEL_COL].copy()
X_train = train_df.drop(columns=drop_cols).copy()

y_test = test_df[LABEL_COL].copy()
X_test = test_df.drop(columns=drop_cols).copy()

print(f"\n[Label distribution]")
print(f"   Training: {np.bincount(y_train_raw)} (0=normal, 1=attack)")
print(f"   Testing:  {np.bincount(y_test)} (0=normal, 1=attack)")

# Handle categorical features (proto, service, state)
print(f"\n[*] Encoding categorical features...")
categorical_cols = X_train.select_dtypes(include=['object', 'string']).columns.tolist()
print(f"   Categorical columns: {categorical_cols}")

X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=False)
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=False)

# Align columns between train and test (in case some categories only in one set)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

print(f"[OK] Feature matrix after encoding: {X_train.shape[1]} features")

# Clean INF/NaN
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)
X_train = X_train.fillna(X_train.median(numeric_only=True))
X_test = X_test.fillna(X_test.median(numeric_only=True))
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

y_train = y_train_raw.values
y_test = y_test.values

# -----------------------------
# 3) Create Validation Set from Training
# -----------------------------
print(f"\n[*] Splitting training into train/val for early stopping...")
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=y_train
)

print(f"   Train: {X_train_final.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# Handle class imbalance
neg_train = np.sum(y_train == 0)
pos_train = np.sum(y_train == 1)
scale_pos_weight = neg_train / pos_train if pos_train > 0 else 1.0
print(f"[OK] scale_pos_weight (neg/pos): {scale_pos_weight:.4f}")

# -----------------------------
# 4) Base Model Parameters
# -----------------------------
base_params = dict(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbosity=1,
    scale_pos_weight=scale_pos_weight
)

base_model = XGBClassifier(**base_params)

# -----------------------------
# 5) Hyperparameter Search Space
# -----------------------------
param_dist = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 4, 5, 6],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "gamma": [0, 0.1, 0.3],
    "reg_alpha": [0, 1e-2, 1e-1],
    "reg_lambda": [1.0, 2.0, 5.0],
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=15,  # Reduced for faster training
    scoring="roc_auc",
    cv=cv,
    verbose=1,
    random_state=RANDOM_SEED,
    n_jobs=-1
)

# -----------------------------
# 6) Run Hyperparameter Tuning
# -----------------------------
print("\n[*] Starting hyperparameter tuning (this may take a few minutes)...\n")
t0 = time.perf_counter()
search.fit(X_train, y_train)
tune_time_s = time.perf_counter() - t0

print("\n[OK] Best params from search:")
print(search.best_params_)
print(f"Tuning time: {tune_time_s:.3f} s")
print(f"[OK] Best CV roc_auc: {search.best_score_:.6f}")

best_params = search.best_params_

# -----------------------------
# 7) Train Final Model with Early Stopping
# -----------------------------
print("\n[*] Training final model with early stopping...\n")
final_model = XGBClassifier(**base_params, **best_params, early_stopping_rounds=30)

t1 = time.perf_counter()
final_model.fit(
    X_train_final, y_train_final,
    eval_set=[(X_val, y_val)],
    verbose=True
)
train_time_s = time.perf_counter() - t1

print(f"\nFinal training time (with early stopping): {train_time_s:.6f} s")
print(f"[OK] Best iteration used: {final_model.best_iteration}")

# -----------------------------
# 8) Predictions
# -----------------------------
print("\n[*] Running predictions on test set...")
t2 = time.perf_counter()
proba = final_model.predict_proba(X_test)
infer_time_s = time.perf_counter() - t2

per_sample_ms = (infer_time_s / len(X_test)) * 1000
print(f"Total inference time (predict_proba): {infer_time_s:.6f} s")
print(f"Inference time per sample: {per_sample_ms:.6f} ms/sample")

y_pred = (proba[:, 1] >= 0.5).astype(int)

# -----------------------------
# 9) Metrics
# -----------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='binary', zero_division=0)
rec = recall_score(y_test, y_pred, average='binary', zero_division=0)
f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)

print("\n" + "="*70)
print("TEST SET METRICS (UNSW-NB15)")
print("="*70)
print(f"Accuracy : {acc:.6f}")
print(f"Precision: {prec:.6f} (binary)")
print(f"Recall   : {rec:.6f} (binary)")
print(f"F1-score : {f1:.6f} (binary)")

try:
    ll = log_loss(y_test, proba)
    print(f"Log Loss : {ll:.6f}")
except Exception as e:
    print("Log Loss : could not compute ->", e)

try:
    auc = roc_auc_score(y_test, proba[:, 1])
    print(f"ROC-AUC  : {auc:.6f}")
except Exception as e:
    print("ROC-AUC  : could not compute ->", e)

print("\n[Confusion Matrix]")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"   TN={cm[0,0]}, FP={cm[0,1]}")
print(f"   FN={cm[1,0]}, TP={cm[1,1]}")

print("\n[Classification Report]")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack'], zero_division=0))

# -----------------------------
# 10) Feature Importance (Top 10)
# -----------------------------
print("\n[Top 10 Most Important Features]")
feature_names = X_train.columns
importances = final_model.feature_importances_
top_features = sorted(zip(feature_names, importances), 
                     key=lambda x: x[1], reverse=True)[:10]
for i, (feat, imp) in enumerate(top_features, 1):
    print(f"   {i:2d}. {feat:30s}: {imp:.6f}")

print("\n" + "="*70)
print("[OK] UNSW-NB15 Training Complete!")
print("="*70)
