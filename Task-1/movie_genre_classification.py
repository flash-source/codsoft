"""
Movie Genre Classification
---------------------------
Predicts a movie's genre from its plot summary using TF-IDF features
with Naive Bayes, Logistic Regression, and Linear SVM classifiers.

Dataset: Genre Classification Dataset (IMDb) - Kaggle
Expected files (place in the same folder as this script, or update the paths below):
    train_data.txt   -> id ::: title ::: genre ::: description
    test_data.txt    -> id ::: title ::: description
    test_data_solution.txt (optional) -> id ::: title ::: genre ::: description  (for scoring test predictions)

Usage:
    python movie_genre_classifier.py
"""

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

TRAIN_PATH = "train_data.txt"
TEST_PATH = "test_data.txt"
TEST_SOLUTION_PATH = "test_data_solution.txt"  # optional, set to None if you don't have it


# ---------- 1. Load data ----------
def load_train_data(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ::: ")
            if len(parts) == 4:
                rows.append(parts)
    return pd.DataFrame(rows, columns=["id", "title", "genre", "description"])


def load_test_data(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ::: ")
            if len(parts) == 3:
                rows.append(parts)
    return pd.DataFrame(rows, columns=["id", "title", "description"])


# ---------- 2. Clean text ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    train_df = load_train_data(TRAIN_PATH)
    test_df = load_test_data(TEST_PATH)

    train_df["clean_desc"] = train_df["description"].apply(clean_text)
    test_df["clean_desc"] = test_df["description"].apply(clean_text)

    print(f"Train rows: {len(train_df)}  |  Test rows: {len(test_df)}")
    print(f"Number of genres: {train_df['genre'].nunique()}")

    # ---------- 3. Train/validation split ----------
    X_train, X_val, y_train, y_val = train_test_split(
        train_df["clean_desc"], train_df["genre"],
        test_size=0.2, random_state=42, stratify=train_df["genre"]
    )

    # ---------- 4. TF-IDF vectorization ----------
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    # ---------- 5. Train and compare models ----------
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=5),
        "Linear SVM": LinearSVC(max_iter=2000),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_val_tfidf)
        acc = accuracy_score(y_val, preds)
        results[name] = acc
        print(f"\n=== {name} — Validation Accuracy: {acc:.4f} ===")
        print(classification_report(y_val, preds, zero_division=0))

    best_name = max(results, key=results.get)
    print(f"\nBest model: {best_name} ({results[best_name]:.4f})")

    # ---------- 6. Retrain best model on full training data ----------
    best_model = models[best_name]
    X_full_tfidf = vectorizer.fit_transform(train_df["clean_desc"])
    best_model.fit(X_full_tfidf, train_df["genre"])

    # ---------- 7. Predict on test set ----------
    X_test_tfidf = vectorizer.transform(test_df["clean_desc"])
    test_df["predicted_genre"] = best_model.predict(X_test_tfidf)
    test_df[["id", "title", "predicted_genre"]].to_csv("predictions.csv", index=False)
    print("\nPredictions saved to predictions.csv")

    # Optional: score against ground truth if provided
    if TEST_SOLUTION_PATH:
        try:
            sol_df = load_train_data(TEST_SOLUTION_PATH)  # same 4-column format
            merged = test_df.merge(sol_df[["id", "genre"]], on="id")
            test_acc = accuracy_score(merged["genre"], merged["predicted_genre"])
            print(f"Test set accuracy vs solution file: {test_acc:.4f}")
        except FileNotFoundError:
            pass

    # ---------- 8. Save model + vectorizer for reuse ----------
    joblib.dump(best_model, "genre_classifier_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("Model and vectorizer saved (genre_classifier_model.pkl, tfidf_vectorizer.pkl)")


if __name__ == "__main__":
    main()
