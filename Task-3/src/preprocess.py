"""
Data loading + preprocessing.

Design choices (and why):
- RowNumber / CustomerId / Surname are identifiers with no causal link to churn.
  Keeping them risks the model latching onto noise (e.g. memorizing CustomerId).
- Geography is one-hot encoded (no ordinal relationship between France/Germany/Spain).
- Gender is binary-encoded.
- Scaling is fit ONLY on the training split, then applied to test -> avoids leakage.
- Stratified split is used because churn is imbalanced (~20% positive class);
  a plain random split can skew the class ratio between train/test.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config


def load_data(path: str = config.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in config.DROP_COLS if c in df.columns])

    # Binary-encode Gender
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    # One-hot encode Geography (drop_first avoids the dummy-variable trap)
    df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

    return df


def split_and_scale(df: pd.DataFrame):
    X = df.drop(columns=[config.TARGET_COL])
    y = df[config.TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    numeric_cols_present = [c for c in config.NUMERIC_COLS if c in X_train.columns]
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[numeric_cols_present] = scaler.fit_transform(X_train[numeric_cols_present])
    X_test[numeric_cols_present] = scaler.transform(X_test[numeric_cols_present])

    return X_train, X_test, y_train, y_test, scaler


def get_processed_data():
    df = load_data()
    df = clean_and_encode(df)
    return split_and_scale(df)
