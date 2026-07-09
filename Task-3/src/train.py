"""
Trains three classifiers on the same preprocessed split so results are comparable:
  - Logistic Regression (linear baseline, easy to interpret via coefficients)
  - Random Forest        (non-linear, handles feature interactions, robust to scale)
  - Gradient Boosting    (usually the strongest of the three on tabular data)

Churn is imbalanced (~80/20). Instead of ignoring that:
  - LogisticRegression / RandomForest use class_weight="balanced"
  - GradientBoostingClassifier has no class_weight param, so we pass
    per-sample weights computed the same way, at fit time.
Ignoring the imbalance would let a model get ~80% "accuracy" by always
predicting "not churned" -- which is why accuracy alone is reported
alongside precision/recall/F1/ROC-AUC in evaluate.py, not on its own.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src import config


def get_models():
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=config.RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=config.RANDOM_STATE,
        ),
    }


def train_all(X_train, y_train):
    models = get_models()
    fitted = {}

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    for name, model in models.items():
        if name == "gradient_boosting":
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)
        fitted[name] = model

    return fitted
