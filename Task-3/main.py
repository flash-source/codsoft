import os
import joblib

from src import config
from src.preprocess import get_processed_data
from src.train import train_all
from src.evaluate import (
    evaluate_model, plot_confusion_matrices, plot_roc_curves,
    plot_feature_importance, build_metrics_table, save_metrics_json,
)


def main():
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = get_processed_data()
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Churn rate — train: {y_train.mean():.3f}, test: {y_test.mean():.3f}\n")

    print("Training models: logistic_regression, random_forest, gradient_boosting...")
    models = train_all(X_train, y_train)

    results = {}
    for name, model in models.items():
        metrics, y_pred, y_proba = evaluate_model(name, model, X_test, y_test)
        results[name] = {"model": model, "metrics": metrics, "y_pred": y_pred, "y_proba": y_proba}
        print(f"  {name:20s} | acc={metrics['accuracy']:.3f}  prec={metrics['precision']:.3f}  "
              f"recall={metrics['recall']:.3f}  f1={metrics['f1_score']:.3f}  auc={metrics['roc_auc']:.3f}")

    print("\nGenerating plots...")
    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)
    for name, r in results.items():
        plot_feature_importance(r["model"], X_train.columns.tolist(), name)

    print("Saving metrics.json...")
    save_metrics_json(results)

    table = build_metrics_table(results)
    print("\n=== Ranked by ROC-AUC ===")
    print(table.to_string(index=False))

    best_name = table.iloc[0]["model"]
    best_model = results[best_name]["model"]
    best_path = os.path.join(config.MODELS_DIR, "best_model.joblib")
    joblib.dump({"model": best_model, "scaler": scaler, "feature_names": X_train.columns.tolist()}, best_path)
    print(f"\nBest model: {best_name} -> saved to {best_path}")


if __name__ == "__main__":
    main()
