# Customer Churn Prediction

Predicts whether a bank customer will churn (`Exited`) using the
`Churn_Modelling.csv` dataset (10,000 customers, 14 columns), comparing
Logistic Regression, Random Forest, and Gradient Boosting.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

This trains all three models on the same train/test split, evaluates them,
and writes everything to `outputs/`:

- `outputs/metrics.json` — accuracy, precision, recall, F1, ROC-AUC per model
- `outputs/plots/confusion_matrices.png`
- `outputs/plots/roc_curves.png`
- `outputs/plots/feature_importance_random_forest.png`
- `outputs/plots/feature_importance_gradient_boosting.png`
- `outputs/models/best_model.joblib` — best model by ROC-AUC, bundled with
  its scaler and feature order so it can be reused directly

## Project structure

```
churn-prediction/
├── data/Churn_Modelling.csv
├── src/
│   ├── config.py        # paths, column lists, random seed
│   ├── preprocess.py     # load, clean, encode, split, scale
│   ├── train.py           # model definitions + fitting
│   └── evaluate.py        # metrics + plots
├── main.py                 # orchestrates the full pipeline
├── requirements.txt
└── outputs/                # created by running main.py
```

## Preprocessing decisions

- Dropped `RowNumber`, `CustomerId`, `Surname` — identifiers with no
  predictive relationship to churn; keeping them risks the model
  memorizing noise instead of learning signal.
- `Geography` one-hot encoded (no ordinal relationship between countries),
  `Gender` binary-encoded.
- `StandardScaler` fit **only on the training split**, then applied to
  test — fitting on the full dataset before splitting would leak test-set
  statistics into training.
- Stratified train/test split (80/20) — churn is ~20% positive, so a plain
  random split risks skewing the class ratio between splits.

## Handling class imbalance

Churn is imbalanced (~20% positive). Logistic Regression and Random Forest
use `class_weight="balanced"`; Gradient Boosting doesn't support that
parameter, so equivalent per-sample weights are computed manually and
passed at `fit()` time. This is why **accuracy alone is not the metric to
trust here** — a model that always predicts "stays" would score ~80%
accuracy while catching zero churners. ROC-AUC and recall matter more for
this problem, since missing an actual churner (false negative) is usually
costlier than a false alarm.

## Results (this run, `random_state=42`)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Gradient Boosting | 0.802 | 0.508 | 0.754 | 0.607 | **0.867** |
| Random Forest | 0.820 | 0.545 | 0.705 | 0.615 | 0.865 |
| Logistic Regression | 0.714 | 0.388 | 0.700 | 0.499 | 0.777 |

**Gradient Boosting and Random Forest are effectively tied** (AUC within
0.002 of each other) — that gap is well within noise for a single
train/test split and shouldn't be read as "GB is better." Logistic
Regression is clearly behind both (AUC 0.777), which makes sense: it can
only draw a linear boundary, and churn here depends on interactions (e.g.
`Age` × `NumOfProducts` × `IsActiveMember`) that a linear model can't
represent no matter how the features are scaled.

Recall sits around 0.70–0.75 for all three balanced models, precision
around 0.39–0.55. In plain terms: the models catch roughly 3 out of 4
customers who will actually churn, but a little under half of the
customers flagged as "at risk" won't actually leave. That's a reasonable
trade for a retention-campaign use case (false alarms cost a discount
email; missed churners cost the customer), but it's a real trade-off, not
a solved problem — worth stating explicitly rather than just quoting the
accuracy number.

`Age` and `NumOfProducts` dominate feature importance in both tree models,
followed by `IsActiveMember` and `Balance` — consistent with the intuitive
story that older, less-engaged customers with fewer products are the ones
at risk.

## What I'd try next

- Cross-validation instead of a single split, to get a confidence interval
  on ROC-AUC rather than one point estimate.
- Hyperparameter search (`GridSearchCV`/`Optuna`) — none of the three
  models here were tuned beyond reasonable defaults.
- SMOTE or threshold tuning instead of `class_weight`, and picking the
  decision threshold based on an actual cost ratio between false
  positives and false negatives, rather than the default 0.5 cutoff.
