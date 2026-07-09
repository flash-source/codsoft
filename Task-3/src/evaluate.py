"""
Evaluation: metrics table + saved plots.

Why not just accuracy: with ~80% of customers not churning, a model that
always predicts "stays" scores 80% accuracy while being useless. So every
model here is scored on accuracy, precision, recall, F1, and ROC-AUC, and
the model selection at the end optimizes for ROC-AUC, not accuracy.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)

from src import config


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return metrics, y_pred, y_proba


def plot_confusion_matrices(results, y_test):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]

    for ax, (name, r) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, r["y_pred"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name.replace("_", " ").title())
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Stayed", "Churned"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Stayed", "Churned"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.tight_layout()
    fig.savefig(os.path.join(config.PLOTS_DIR, "confusion_matrices.png"), dpi=150)
    plt.close(fig)


def plot_roc_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        ax.plot(fpr, tpr, label=f"{name.replace('_', ' ').title()} (AUC={r['metrics']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(config.PLOTS_DIR, "roc_curves.png"), dpi=150)
    plt.close(fig)


def plot_feature_importance(model, feature_names, model_name):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(
        [feature_names[i] for i in order][::-1],
        [importances[i] for i in order][::-1],
        color="#4C72B0",
    )
    ax.set_title(f"Feature Importance — {model_name.replace('_', ' ').title()}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(config.PLOTS_DIR, f"feature_importance_{model_name}.png"), dpi=150)
    plt.close(fig)


def build_metrics_table(results):
    rows = []
    for name, r in results.items():
        row = {"model": name, **r["metrics"]}
        rows.append(row)
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def save_metrics_json(results):
    out = {name: r["metrics"] for name, r in results.items()}
    with open(config.METRICS_PATH, "w") as f:
        json.dump(out, f, indent=2)
