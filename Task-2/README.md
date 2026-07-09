# Task 2 — Credit Card Fraud Detection

## Objective
Build a model to detect fraudulent credit card transactions, experimenting with
Logistic Regression, Decision Tree, and Random Forest classifiers.

## Dataset
- **Source**: Simulated credit card transaction dataset (fraudTrain.csv / fraudTest.csv)
- **Train set**: 1,296,675 transactions
- **Test set**: 555,719 transactions
- **Class balance**: only **0.58%** of training transactions are fraudulent (7,506 of 1,296,675) — a severe class imbalance that shapes every modeling decision below.
- **Raw columns**: transaction time, merchant, category, amount, cardholder demographics/location, merchant location, and the `is_fraud` label.

## Approach

### 1. Exploratory Data Analysis (`scripts/01_eda.py`)
- Confirmed the extreme class imbalance (see `outputs/01_class_imbalance.png`).
- Fraudulent transactions average **$531** vs **$68** for legitimate ones — amount is highly discriminative.
- Fraud rate varies sharply by merchant `category` and by `hour` of day (late-night transactions are disproportionately fraudulent) — see `outputs/02_fraud_rate_by_category.png` and `outputs/03_fraud_rate_by_hour.png`.

### 2. Feature Engineering (`scripts/feature_engineering.py`)
Raw columns like credit card number, name, and street address carry no generalizable
signal and were dropped. Instead, engineered:
- **Temporal**: hour, day-of-week, month, plus cyclical sin/cos encoding of hour (so 23:00 and 00:00 are recognized as adjacent).
- **`age`**: computed from date of birth at time of transaction.
- **`distance_km`**: haversine distance between cardholder location and merchant location — fraud often involves geographically distant merchants.
- **`amt_log`**: log-transformed amount to tame the right-skewed distribution.
- **`category`**: one-hot encoded (14 merchant categories).
- **`gender_M`**: binary encoded.

Final feature matrix: 25 numeric columns, no missing values, no data leakage
(all features derivable from information available at transaction time).

### 3. Modeling (`scripts/02_train_models.py`)
All three models used `class_weight='balanced'` (or `'balanced_subsample'` for
Random Forest) to compensate for the 0.58% fraud rate, rather than naive resampling —
this keeps the full dataset intact and avoids synthetic-sample artifacts.

| Model | Key settings |
|---|---|
| Logistic Regression | `class_weight='balanced'`, features standardized |
| Decision Tree | `max_depth=12`, `min_samples_leaf=20`, `class_weight='balanced'` |
| Random Forest | `n_estimators=150`, `max_depth=16`, `min_samples_leaf=5`, `class_weight='balanced_subsample'` |

### 4. Evaluation (`scripts/03_visualize_results.py`)
Given the imbalance, **accuracy is a misleading metric** (predicting "legitimate"
for every transaction already scores 99.4%). Evaluation instead centers on
**precision, recall, F1, ROC-AUC, and PR-AUC** for the fraud class.

## Results

| Model | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.057 | 0.846 | 0.106 | 0.962 | 0.174 |
| Decision Tree | 0.109 | 0.964 | 0.196 | 0.982 | 0.605 |
| **Random Forest** | **0.356** | **0.927** | **0.514** | **0.996** | **0.858** |

**Random Forest is the best-performing model**, catching 93% of fraud
(1,989 / 2,145) in the held-out test set while cutting false positives by
~8x relative to the Decision Tree (3,605 vs 16,844) and ~8x relative to
Logistic Regression (3,605 vs 30,149). Its PR-AUC of 0.858 — the most
informative metric on this imbalanced a problem — is far ahead of the
other two models.

Logistic Regression's high recall comes at the cost of very low precision:
it flags too many legitimate transactions as fraud, because a linear
decision boundary can't capture the non-linear interactions (e.g., amount
× category × hour) that separate fraud from noise. The Decision Tree
improves on this by learning those interactions, but overfits to noisy
splits without the variance reduction that ensembling provides. Random
Forest's averaging over 150 trees gives the best bias-variance tradeoff.

The top predictive features (`outputs/09_feature_importance.png`) are
transaction amount (raw and log), a handful of merchant categories
(notably `grocery_pos`), geographic distance between cardholder and
merchant, and time-of-day — consistent with the EDA findings.

## Repository Structure
```
task2_credit_card_fraud_detection/
├── README.md
├── scripts/
│   ├── feature_engineering.py     # shared feature engineering module
│   ├── 01_eda.py                  # exploratory data analysis
│   ├── 02_train_models.py         # training + evaluation pipeline
│   └── 03_visualize_results.py    # comparison plots
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl          # best model
│   ├── scaler.pkl                 # StandardScaler (for Logistic Regression)
│   └── feature_columns.json       # exact feature order used at train time
└── outputs/
    ├── 01_class_imbalance.png
    ├── 02_fraud_rate_by_category.png
    ├── 03_fraud_rate_by_hour.png
    ├── 04_amount_distribution.png
    ├── 05_confusion_matrices.png
    ├── 06_roc_curves.png
    ├── 07_pr_curves.png
    ├── 08_metric_comparison.png
    ├── 09_feature_importance.png
    └── metrics_summary.json
```

## How to Reproduce
```bash
python scripts/01_eda.py
python scripts/02_train_models.py
python scripts/03_visualize_results.py
```

## Notes / Possible Extensions
- Could try SMOTE/undersampling as an alternative to `class_weight` balancing.
- Could add a threshold-tuning step: since Random Forest outputs probabilities,
  the precision/recall tradeoff can be adjusted post-hoc depending on whether
  the business prioritizes catching more fraud (favor recall) or minimizing
  false alarms (favor precision).
- Could add gradient boosting (XGBoost/LightGBM) as a fourth comparison point.
