"""Project configuration for the grocery demand forecasting pipeline.

This module centralizes paths, feature definitions, model settings, and
time-based split configuration used by training and inference scripts.
Keeping these values in one place makes the project easier to maintain
and reduces hardcoded values across files.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

DATA_PATH = DATA_DIR / "processed_sales_features_v2.csv"
MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"
REPORT_PATH = REPORTS_DIR / "item_sales_model_results_v2.csv"

# Dataset metadata
TARGET_COLUMN = "units_sold"
DATE_COLUMN = "date"

# Time split used in the assignment
TRAIN_START_DATE = "2025-09-06"
TRAIN_END_DATE = "2025-11-15"
TEST_START_DATE = "2025-11-16"
TEST_END_DATE = "2025-11-30"

# Feature lists
TRAIN_FEATURES = [
    "price",
    "promo_flag",
    "promo_depth",
    "stockout_flag",
    "lag_1",
    "lag_3",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_7",
    "rolling_14",
    "rolling_28",
    "weekday",
    "weekend",
    "stock_ratio",
    "promo_price_interaction",
    "lag1_stock_interaction",
]

PREDICTION_FEATURES = [
    "lag_1",
    "lag_3",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_7",
    "rolling_14",
    "rolling_28",
    "weekday",
    "weekend",
    "stock_ratio",
    "promo_price_interaction",
    "lag1_stock_interaction",
]

# Model configuration
RANDOM_STATE = 42
RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
