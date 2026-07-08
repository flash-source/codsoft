# Task 1: Movie Genre Classification

Predicts a movie's genre from its plot summary/description using TF-IDF text features and classical ML classifiers.

## Dataset
[Genre Classification Dataset (IMDb)](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)
- 54,214 training samples, 54,200 test samples, 27 genres

## Approach
1. **Text cleaning**: lowercasing, punctuation removal, whitespace normalization
2. **Feature extraction**: TF-IDF (unigrams + bigrams, 50,000 max features, English stopwords removed)
3. **Models trained & compared**:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Linear SVM
4. Best model (by validation accuracy) retrained on full training data and used for final test predictions

## Results

| Model | Validation Accuracy |
|---|---|
| Naive Bayes | 46.8% |
| **Logistic Regression (best)** | **58.8%** |
| Linear SVM | 57.9% |

**Final test set accuracy (Logistic Regression): 59.96%**

Best-performing genres: documentary, drama, comedy, western, horror (most frequent classes).
Weakest: biography, war, fantasy, mystery, news — genres with very few training samples (class imbalance).

## Files
- `movie_genre_classification.py` — full training/evaluation/prediction pipeline
- `predictions.csv` — predicted genres for the test set
- `genre_classifier_model.pkl` — trained Logistic Regression model
- `tfidf_vectorizer.pkl` — fitted TF-IDF vectorizer (needed to transform new text before prediction)

## Possible improvements
- Class weighting to help underrepresented genres
- Combine title + description as model input
- Word embeddings (e.g. sentence-transformers) instead of pure TF-IDF
