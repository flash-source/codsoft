"""
Train and evaluate Logistic Regression, Decision Tree, and Random Forest
classifiers for credit card fraud detection.
"""
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve
)
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import engineer_features, align_columns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'fraudTrain.csv')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'fraudTest.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUT_DIR = os.path.join(BASE_DIR, 'outputs')

t0 = time.time()

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading data...")
usecols = ['trans_date_trans_time', 'category', 'amt', 'gender', 'lat', 'long',
           'city_pop', 'dob', 'merch_lat', 'merch_long', 'is_fraud']
train_raw = pd.read_csv(TRAIN_PATH, usecols=usecols)
test_raw = pd.read_csv(TEST_PATH, usecols=usecols)
print(f"Train shape: {train_raw.shape}, Test shape: {test_raw.shape}")
print(f"[{time.time()-t0:.1f}s elapsed]")

# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
print("\nEngineering features...")
y_train = train_raw['is_fraud'].values
y_test = test_raw['is_fraud'].values

X_train = engineer_features(train_raw)
X_test = engineer_features(test_raw)

# align dummy columns (in case a category is missing in one split)
all_cols = sorted(set(X_train.columns) | set(X_test.columns))
X_train = align_columns(X_train, all_cols)
X_test = align_columns(X_test, all_cols)

print(f"Feature matrix shape: {X_train.shape}")
print(f"Features: {list(X_train.columns)}")
print(f"[{time.time()-t0:.1f}s elapsed]")

del train_raw, test_raw  # free memory

# ---------------------------------------------------------------------------
# 3. Scale features (for Logistic Regression)
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, f'{MODEL_DIR}/scaler.pkl')

# ---------------------------------------------------------------------------
# 4. Train models
# ---------------------------------------------------------------------------
results = {}
predictions = {}

def evaluate(name, model, X_te, y_te):
    t_start = time.time()
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    elapsed = time.time() - t_start

    report = classification_report(y_te, y_pred, target_names=['Legitimate', 'Fraud'], output_dict=True)
    cm = confusion_matrix(y_te, y_pred)
    roc_auc = roc_auc_score(y_te, y_proba)
    pr_auc = average_precision_score(y_te, y_proba)

    print(f"\n--- {name} ---")
    print(classification_report(y_te, y_pred, target_names=['Legitimate', 'Fraud']))
    print(f"Confusion matrix:\n{cm}")
    print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Inference time: {elapsed:.2f}s")

    results[name] = {
        'precision_fraud': report['Fraud']['precision'],
        'recall_fraud': report['Fraud']['recall'],
        'f1_fraud': report['Fraud']['f1-score'],
        'accuracy': report['accuracy'],
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm.tolist(),
    }
    predictions[name] = {'y_proba': y_proba, 'y_pred': y_pred}


# --- Logistic Regression ---
print("\n" + "=" * 60)
print("Training Logistic Regression...")
t1 = time.time()
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_scaled, y_train)
print(f"Trained in {time.time()-t1:.1f}s")
evaluate('Logistic Regression', lr, X_test_scaled, y_test)
joblib.dump(lr, f'{MODEL_DIR}/logistic_regression.pkl')

# --- Decision Tree ---
print("\n" + "=" * 60)
print("Training Decision Tree...")
t1 = time.time()
dt = DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, class_weight='balanced', random_state=42)
dt.fit(X_train, y_train)
print(f"Trained in {time.time()-t1:.1f}s")
evaluate('Decision Tree', dt, X_test, y_test)
joblib.dump(dt, f'{MODEL_DIR}/decision_tree.pkl')

# --- Random Forest ---
print("\n" + "=" * 60)
print("Training Random Forest...")
t1 = time.time()
rf = RandomForestClassifier(
    n_estimators=150, max_depth=16, min_samples_leaf=5,
    class_weight='balanced_subsample', n_jobs=-1, random_state=42
)
rf.fit(X_train, y_train)
print(f"Trained in {time.time()-t1:.1f}s")
evaluate('Random Forest', rf, X_test, y_test)
joblib.dump(rf, f'{MODEL_DIR}/random_forest.pkl')

# ---------------------------------------------------------------------------
# 5. Save results + feature list
# ---------------------------------------------------------------------------
with open(f'{OUT_DIR}/metrics_summary.json', 'w') as f:
    json.dump(results, f, indent=2)

with open(f'{MODEL_DIR}/feature_columns.json', 'w') as f:
    json.dump(all_cols, f, indent=2)

# save proba arrays for plotting (roc/pr curves) - compact via numpy savez
np.savez_compressed(
    f'{OUT_DIR}/predictions.npz',
    y_test=y_test,
    lr_proba=predictions['Logistic Regression']['y_proba'],
    dt_proba=predictions['Decision Tree']['y_proba'],
    rf_proba=predictions['Random Forest']['y_proba'],
)

print(f"\nTotal pipeline time: {time.time()-t0:.1f}s")
print("Models saved to models/, metrics saved to outputs/metrics_summary.json")
