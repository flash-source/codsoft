# Task 4: Spam SMS Detection

Part of the CodSoft Machine Learning Internship.

## Problem Statement

Build an AI model that can classify SMS messages as spam or legitimate. Use techniques
like TF-IDF or word embeddings with classifiers like Naive Bayes, Logistic Regression, or
Support Vector Machines to identify spam messages.

## Dataset

[SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) —
5,572 SMS messages labeled `ham` (legitimate) or `spam`. A raw copy is kept in `archive/`;
the working copy used by the notebook is `spam.csv` at the repo root.

The raw file had 403 exact duplicate rows, which were dropped before splitting to avoid
train/test leakage — 5,169 unique messages remain (4,516 ham / 653 spam).

## Approach

1. **Cleaning** — lowercase, strip emails/URLs/punctuation/digits, remove stopwords
   (manual stopword list, no external downloads required).
2. **Features** — TF-IDF, unigrams + bigrams, capped at 3,000 features.
3. **Models** — Multinomial Naive Bayes, Logistic Regression, and a linear SVM, all
   trained on identical features so the comparison is apples-to-apples.
4. **Evaluation** — since ~87% of the dataset is `ham`, raw accuracy is misleading (a
   model that never predicts spam would still score ~87%). Models are compared on
   **precision, recall, and F1 for the spam class**.

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Logistic Regression** | 0.975 | 0.895 | 0.908 | **0.902** |
| SVM (Linear) | 0.974 | 0.906 | 0.885 | 0.896 |
| Naive Bayes | 0.968 | 0.980 | 0.763 | 0.858 |

**Logistic Regression** was selected as the final model — best F1, and a better
precision/recall balance than Naive Bayes, which has the highest precision but misses
about 1 in 4 spam messages (76% recall).

## Repo Structure

```
Task-4/
├── archive/                     # raw dataset backup (as downloaded)
├── spam.csv                     # working copy of the dataset
├── spam_sms_detection.ipynb     # main notebook: EDA, cleaning, training, evaluation
├── spam_classifier_model.pkl    # trained Logistic Regression model
├── tfidf_vectorizer.pkl         # fitted TF-IDF vectorizer
├── predictions.csv              # model predictions on the held-out test split
├── test_custom_input.py         # load the saved model and test custom messages
└── README.md
```

## How to Run

```bash
# install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn joblib

# open and run the notebook
jupyter notebook spam_sms_detection.ipynb

# or test the saved model directly on custom input
python test_custom_input.py
```

## Tech Stack

Python, pandas, scikit-learn (TF-IDF, Naive Bayes, Logistic Regression, SVM), matplotlib,
seaborn, joblib.
