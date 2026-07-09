"""
Shared fixtures. processed_data and trained_models are session-scoped so the
full pipeline runs once per test session, not once per test.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.preprocess import get_processed_data
from src.train import train_all


@pytest.fixture(scope="session")
def processed_data():
    return get_processed_data()


@pytest.fixture(scope="session")
def trained_models(processed_data):
    X_train, X_test, y_train, y_test, scaler = processed_data
    return train_all(X_train, y_train)
