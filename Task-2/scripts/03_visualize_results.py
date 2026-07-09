"""
Generate comparison plots: confusion matrices, ROC curves, PR curves,
metric comparison bars, and Random Forest feature importance.
"""
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve

sns.set_style('whitegrid')

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUT_DIR = os.path.join(BASE_DIR, 'outputs')

with open(f'{OUT_DIR}/metrics_summary.json') as f:
    results = json.load(f)

data = np.load(f'{OUT_DIR}/predictions.npz')
y_test = data['y_test']
proba = {
    'Logistic Regression': data['lr_proba'],
    'Decision Tree': data['dt_proba'],
    'Random Forest': data['rf_proba'],
}
colors = {'Logistic Regression': '#4C72B0', 'Decision Tree': '#DD8452', 'Random Forest': '#55A868'}

# ---------------------------------------------------------------------------
# 1. Confusion matrices (side by side)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, res) in zip(axes, results.items()):
    cm = np.array(res['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=ax,
                xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'], cbar=False)
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/05_confusion_matrices.png', dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# 2. ROC curves
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 6))
for name, p in proba.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})", color=colors[name], linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random baseline')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/06_roc_curves.png', dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# 3. Precision-Recall curves (more informative for imbalanced data)
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 6))
for name, p in proba.items():
    prec, rec, _ = precision_recall_curve(y_test, p)
    plt.plot(rec, prec, label=f"{name} (AP={results[name]['pr_auc']:.3f})", color=colors[name], linewidth=2)
baseline = y_test.mean()
plt.axhline(baseline, color='k', linestyle='--', alpha=0.4, label=f'Random baseline ({baseline:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/07_pr_curves.png', dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# 4. Metric comparison bar chart
# ---------------------------------------------------------------------------
metrics_to_plot = ['precision_fraud', 'recall_fraud', 'f1_fraud', 'roc_auc', 'pr_auc']
labels = ['Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC']
model_names = list(results.keys())

x = np.arange(len(metrics_to_plot))
width = 0.25
fig, ax = plt.subplots(figsize=(11, 6))
for i, name in enumerate(model_names):
    vals = [results[name][m] for m in metrics_to_plot]
    bars = ax.bar(x + i * width, vals, width, label=name, color=colors[name])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.2f}', ha='center', fontsize=8)
ax.set_xticks(x + width)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.08)
ax.set_ylabel('Score')
ax.set_title('Model Comparison — Fraud Class Metrics')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/08_metric_comparison.png', dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# 5. Random Forest feature importance
# ---------------------------------------------------------------------------
rf = joblib.load(f'{MODEL_DIR}/random_forest.pkl')
with open(f'{MODEL_DIR}/feature_columns.json') as f:
    feature_cols = json.load(f)

importances = rf.feature_importances_
idx = np.argsort(importances)[::-1][:15]
plt.figure(figsize=(9, 7))
plt.barh([feature_cols[i] for i in idx][::-1], importances[idx][::-1], color='#55A868')
plt.xlabel('Feature Importance')
plt.title('Random Forest — Top 15 Feature Importances')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/09_feature_importance.png', dpi=120)
plt.close()

print("All comparison plots saved to outputs/")
