# CICIDS2017 Wednesday - DoS/DDoS Detection for Streaming Monitoring
# Features optimized for real-time threat detection

DATA_PATH = "Wednesday-workingHours.pcap_ISCX.csv"
LABEL_COL = "Label"
RANDOM_SEED = 42

print("="*70)
print("CICIDS2017 Wednesday - DoS/DDoS Attack Detection")
print("Attack Types: DoS Hulk, GoldenEye, Slowloris, Slowhttptest, Heartbleed")
print("="*70)

import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

# -----------------------------
# 1) Load & Initial Analysis
# -----------------------------
print("\n[*] Loading CICIDS2017 Wednesday dataset...")
df = pd.read_csv(DATA_PATH)

# ⚠️ CRITICAL: Strip whitespace from column names
df.columns = df.columns.str.strip()

if LABEL_COL not in df.columns:
    raise ValueError(f"Label column '{LABEL_COL}' not found. Columns: {list(df.columns)}")

print(f"[OK] Initial shape: {df.shape}")
print(f"\n[*] Attack Type Distribution:")
print(df[LABEL_COL].value_counts())

# ⚠️ CRITICAL: Remove NaN labels FIRST
nan_labels = df[LABEL_COL].isna().sum()
if nan_labels > 0:
    print(f"[!] Removing {nan_labels} rows with NaN labels")
    df = df.dropna(subset=[LABEL_COL])

# ⚠️ CRITICAL: Remove duplicates BEFORE splitting to prevent data leakage
duplicate_count = df.duplicated().sum()
if duplicate_count > 0:
    print(f"[!] Removing {duplicate_count} duplicate rows ({duplicate_count/len(df)*100:.2f}%)")
    df = df.drop_duplicates()
    print(f"[OK] Dataset after deduplication: {len(df)} rows")

# ⚠️ CRITICAL: Shuffle the dataframe to prevent label-sorting issues
print(f"\n[OK] Shuffling dataset to ensure random distribution...")
df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

# Convert to binary classification: BENIGN (0) vs ATTACK (1)
print(f"\n[*] Converting to binary classification (BENIGN vs ATTACK)...")
y_raw = (df[LABEL_COL] != 'BENIGN').astype(int)
attack_count = y_raw.sum()
benign_count = len(y_raw) - attack_count
print(f"   BENIGN: {benign_count:,} ({benign_count/len(y_raw)*100:.1f}%)")
print(f"   ATTACK: {attack_count:,} ({attack_count/len(y_raw)*100:.1f}%)")

X = df.drop(columns=[LABEL_COL]).copy()

# Select streaming-relevant features for DoS/DDoS detection
print(f"\n[*] Selecting features optimized for streaming DoS detection...")
streaming_features = [
    # Flow characteristics
    'Flow Duration', 'Flow Packets/s', 'Flow Bytes/s',
    # Packet counts (critical for DoS)
    'Total Fwd Packets', 'Total Backward Packets',
    # Byte counts
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    # Packet size analysis
    'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    'Packet Length Mean', 'Packet Length Std',
    # Timing features (key for DoS detection)
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
    'Fwd IAT Mean', 'Fwd IAT Std',
    'Bwd IAT Mean', 'Bwd IAT Std',
    # TCP flags (detect SYN floods)
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
    # Header info
    'Fwd Header Length', 'Bwd Header Length',
    # Packet rates
    'Fwd Packets/s', 'Bwd Packets/s',
    # Size metrics
    'Min Packet Length', 'Max Packet Length',
    'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
    # Subflow features
    'Subflow Fwd Packets', 'Subflow Bwd Packets',
    'Subflow Fwd Bytes', 'Subflow Bwd Bytes',
]

# Keep only features that exist in the dataset
available_features = [f for f in streaming_features if f in X.columns]
missing_features = [f for f in streaming_features if f not in X.columns]

print(f"   Available features: {len(available_features)}/{len(streaming_features)}")
if missing_features:
    print(f"   [!] Missing: {len(missing_features)} features")

X = X[available_features]

# Clean INF/NaN
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))
X = X.fillna(0)

print(f"\n[OK] Final feature matrix: {X.shape}")

# -----------------------------
# 2) Verify Binary Labels
# -----------------------------
y = y_raw.values
unique_labels = np.unique(y)
print(f"\n[OK] Binary labels verified: {unique_labels} (0=BENIGN, 1=ATTACK)")

if len(unique_labels) < 2:
    raise ValueError(
        f"\n[ERROR] Only one class present -> {unique_labels}.\n"
        "Training is not possible unless you have at least 2 classes."
    )

num_classes = len(unique_labels)
is_binary = (num_classes == 2)

# -----------------------------
# -----------------------------
# 2.5) Feature Importance Preview (Top correlations with DoS attacks)
# -----------------------------
print("\n[*] Top features correlated with DoS/DDoS attacks:")
correlations = []
for col in X.columns:
    try:
        corr = np.corrcoef(X[col].values, y)[0, 1]
        if not np.isnan(corr):
            correlations.append((col, abs(corr), corr))
    except:
        pass

correlations.sort(key=lambda x: x[1], reverse=True)
for i, (col, abs_corr, corr) in enumerate(correlations[:5], 1):
    direction = "higher in attacks" if corr > 0 else "lower in attacks"
    print(f"   {i}. {col:30s}: {abs_corr:.4f} {direction}")

# -----------------------------
# 3) 80:20 split (then split train into train/val for early stopping)
# -----------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size=0.20,
    train_size=0.80,
    random_state=RANDOM_SEED,
    stratify=y
)

# Validation split from training for early stopping
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.20,  # 20% of training used as validation
    random_state=RANDOM_SEED,
    stratify=y_train_full
)

print(f"\nTrain: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# Handle class imbalance (binary)
scale_pos_weight = None
if is_binary:
    neg = np.sum(y_train_full == 0)
    pos = np.sum(y_train_full == 1)
    if pos > 0:
        scale_pos_weight = neg / pos
        print(f"[OK] scale_pos_weight (neg/pos): {scale_pos_weight:.4f}")

# -----------------------------
# 4) Base model (for search)
# -----------------------------
base_params = dict(
    objective="binary:logistic" if is_binary else "multi:softprob",
    eval_metric="logloss" if is_binary else "mlogloss",
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbosity=0  # Quiet; script prints progress
)
if not is_binary:
    base_params["num_class"] = num_classes

if is_binary and scale_pos_weight is not None:
    base_params["scale_pos_weight"] = scale_pos_weight

base_model = XGBClassifier(**base_params)

# -----------------------------
# 5) Random search space (fine-tuning)
# -----------------------------
param_dist = {
    "n_estimators": [200, 400, 800],         # more trees; early stopping will pick best
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [3, 4, 5, 6, 8],
    "min_child_weight": [1, 3, 5, 7],
    "subsample": [0.6, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.8, 0.9, 1.0],
    "gamma": [0, 0.1, 0.3, 0.5, 1.0],
    "reg_alpha": [0, 1e-3, 1e-2, 1e-1],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

# Scoring: use ROC-AUC if binary, else accuracy
scoring = "roc_auc" if is_binary else "accuracy"

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=25,               # increase to 50 for deeper tuning
    scoring=scoring,
    cv=cv,
    verbose=0,
    random_state=RANDOM_SEED,
    n_jobs=-1
)

# -----------------------------
# 6) Run tuning (CV)
# -----------------------------
print("\n[*] Starting hyperparameter tuning (this may take a few minutes)...\n")
t0 = time.perf_counter()
search.fit(X_train_full, y_train_full)   # tuning on the full training set (80%)
tune_time_s = time.perf_counter() - t0

print("\n[OK] Best params from search:")
print(search.best_params_)
print(f"[*] Tuning time: {tune_time_s:.3f} s")
print(f"[OK] Best CV {scoring}: {search.best_score_:.6f}")

best_params = search.best_params_

# -----------------------------
# 7) Train final model with early stopping (train vs val)
# -----------------------------
print("\n[*] Training final model with early stopping...\n")
final_model = XGBClassifier(**base_params, **best_params, early_stopping_rounds=30)

t1 = time.perf_counter()
final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False  # Quiet; iteration count printed after fit
)
train_time_s = time.perf_counter() - t1

print(f"\n[*] Final training time (with early stopping): {train_time_s:.6f} s")
print(f"[OK] Best iteration used: {final_model.best_iteration}")

# -----------------------------
# 8) Inference time (ms/sample) + predictions
# -----------------------------
print("\n[*] Running predictions on test set...")
t2 = time.perf_counter()
proba = final_model.predict_proba(X_test)
infer_time_s = time.perf_counter() - t2

per_sample_ms = (infer_time_s / len(X_test)) * 1000
print(f"[*] Total inference time (predict_proba): {infer_time_s:.6f} s")
print(f"[*] Inference time per sample: {per_sample_ms:.6f} ms/sample")

y_pred = np.argmax(proba, axis=1) if not is_binary else (proba[:, 1] >= 0.5).astype(int)

# -----------------------------
# 9) Metrics & DoS Detection Analysis
# -----------------------------
avg = "binary" if is_binary else "weighted"

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)

print("\n" + "="*70)
print("TEST SET METRICS - DoS/DDoS Detection")
print("="*70)
print(f"Accuracy : {acc:.6f}")
print(f"Precision: {prec:.6f} ({avg})")
print(f"Recall   : {rec:.6f} ({avg})")
print(f"F1-score : {f1:.6f} ({avg})")

try:
    ll = log_loss(y_test, proba)
    print(f"Log Loss : {ll:.6f}")
except Exception as e:
    print("Log Loss : could not compute ->", e)

try:
    if is_binary:
        auc = roc_auc_score(y_test, proba[:, 1])
        print(f"ROC-AUC  : {auc:.6f}")
    else:
        auc = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")
        print(f"ROC-AUC  : {auc:.6f} (multiclass ovr, weighted)")
except Exception as e:
    print("ROC-AUC  : could not compute ->", e)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n[*] Confusion Matrix:")
print(cm)
print(f"   True Negatives (Benign correctly identified):  {tn:,}")
print(f"   False Positives (Benign flagged as attack):    {fp:,}")
print(f"   False Negatives (Attacks missed):              {fn:,}")
print(f"   True Positives (Attacks correctly detected):   {tp:,}")

# Calculate rates for streaming monitoring
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

print(f"\n[*] Streaming Monitoring Metrics:")
print(f"   False Positive Rate: {fpr:.4f} ({fpr*100:.2f}%)")
print(f"   False Negative Rate: {fnr:.4f} ({fnr*100:.2f}%)")
print(f"   -> Expected {fp:,} false alerts per {len(y_test):,} flows")
print(f"   -> Expected {fn:,} missed attacks per {len(y_test):,} flows")

print("\n[*] Classification Report:")
print(classification_report(y_test, y_pred, target_names=['BENIGN', 'ATTACK'], zero_division=0))

# Top features for DoS detection
print("\n[*] Top 10 Most Important Features for DoS Detection:")
feature_names = X.columns
importances = final_model.feature_importances_
top_features = sorted(zip(feature_names, importances), 
                     key=lambda x: x[1], reverse=True)[:10]
for i, (feat, imp) in enumerate(top_features, 1):
    print(f"   {i:2d}. {feat:35s}: {imp:.6f}")

print("\n" + "="*70)
print("[OK] CICIDS2017 Wednesday DoS/DDoS Detection Complete!")
print("="*70)
print(f"\n[*] Streaming Deployment Recommendations:")
print(f"   - Inference speed: {per_sample_ms:.4f} ms/flow -> ~{int(1000/per_sample_ms):,} flows/second")
print(f"   - Model optimized for: DoS Hulk, GoldenEye, Slowloris, Heartbleed")
print(f"   - Key indicators: Packet rates, flow timing, TCP flags")
print(f"   - Alert tuning: Consider threshold > 0.5 to reduce {fp:,} false positives")
print("="*70)