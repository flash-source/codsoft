"""
These tests check evaluate.py's own math against known inputs, using a fake
model instead of a trained one -- so a bug in evaluate.py (e.g. swapped
precision/recall) fails here even if every real model happens to score fine.
"""
import numpy as np
import pandas as pd

from src.evaluate import evaluate_model, build_metrics_table


class DummyModel:
    def __init__(self, preds, probs):
        self._preds = np.array(preds)
        self._probs = np.array(probs)

    def predict(self, X):
        return self._preds

    def predict_proba(self, X):
        return np.column_stack([1 - self._probs, self._probs])


def test_evaluate_model_perfect_predictions():
    y_test = pd.Series([0, 1, 0, 1])
    model = DummyModel(preds=[0, 1, 0, 1], probs=[0.1, 0.9, 0.2, 0.8])

    metrics, y_pred, y_proba = evaluate_model("dummy", model, X_test=None, y_test=y_test)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_evaluate_model_catches_missed_positives():
    # Model predicts "no churn" for everyone -> recall must be 0, not silently high.
    y_test = pd.Series([0, 1, 0, 1, 1])
    model = DummyModel(preds=[0, 0, 0, 0, 0], probs=[0.1, 0.2, 0.1, 0.3, 0.2])

    metrics, _, _ = evaluate_model("dummy", model, X_test=None, y_test=y_test)

    assert metrics["recall"] == 0.0
    assert metrics["precision"] == 0.0  # no positive predictions at all


def test_build_metrics_table_sorted_by_auc_descending():
    results = {
        "low_auc": {"metrics": {"accuracy": 0.9, "precision": 0.9, "recall": 0.5, "f1_score": 0.6, "roc_auc": 0.60}},
        "high_auc": {"metrics": {"accuracy": 0.7, "precision": 0.5, "recall": 0.9, "f1_score": 0.6, "roc_auc": 0.90}},
    }

    table = build_metrics_table(results)

    assert table.iloc[0]["model"] == "high_auc"
    assert table.iloc[-1]["model"] == "low_auc"
