from src.preprocess import load_data, clean_and_encode
from src import config


def test_load_data_shape():
    df = load_data()
    assert df.shape == (10000, 14)
    assert config.TARGET_COL in df.columns


def test_no_nulls():
    df = load_data()
    assert df.isnull().sum().sum() == 0


def test_identifier_columns_dropped():
    df = clean_and_encode(load_data())
    for col in config.DROP_COLS:
        assert col not in df.columns


def test_geography_one_hot_encoded():
    df = clean_and_encode(load_data())
    assert "Geography" not in df.columns
    assert any(c.startswith("Geography_") for c in df.columns)


def test_gender_binary_encoded():
    df = clean_and_encode(load_data())
    assert set(df["Gender"].unique()) == {0, 1}


def test_split_preserves_row_count(processed_data):
    X_train, X_test, y_train, y_test, scaler = processed_data
    assert len(X_train) + len(X_test) == 10000


def test_split_is_stratified(processed_data):
    # churn rate should be near-identical between train and test (~20% either way)
    X_train, X_test, y_train, y_test, scaler = processed_data
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_scaler_fit_on_train_only(processed_data):
    # scaled numeric columns should average ~0 on the split the scaler was fit on
    X_train, X_test, y_train, y_test, scaler = processed_data
    numeric_cols = [c for c in config.NUMERIC_COLS if c in X_train.columns]
    train_means = X_train[numeric_cols].mean().abs()
    assert (train_means < 1e-6).all()
