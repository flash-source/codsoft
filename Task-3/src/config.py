"""
Central configuration: paths, column names, random seed.
Keeping this separate means every other module stays free of hardcoded strings.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Churn_Modelling.csv")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
METRICS_PATH = os.path.join(BASE_DIR, "outputs", "metrics.json")

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_COL = "Exited"

# Columns that are pure identifiers / free text -> no predictive value, drop them.
DROP_COLS = ["RowNumber", "CustomerId", "Surname"]

CATEGORICAL_COLS = ["Geography", "Gender"]
NUMERIC_COLS = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]
