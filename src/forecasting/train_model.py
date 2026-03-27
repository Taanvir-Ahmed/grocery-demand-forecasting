"""Train the grocery demand forecasting model.

This script loads the engineered sales dataset, applies a time-based split,
trains a Random Forest regressor, evaluates it with forecasting metrics,
and saves both the trained model and a one-row metrics report.

Expected inputs:
- data/processed_sales_features_v2.csv

Generated outputs:
- models/random_forest_model.pkl
- reports/item_sales_model_results_v2.csv
"""

from __future__ import annotations

import logging
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import (
    DATA_PATH,
    DATE_COLUMN,
    MODEL_PATH,
    RANDOM_FOREST_PARAMS,
    REPORT_PATH,
    TARGET_COLUMN,
    TEST_END_DATE,
    TEST_START_DATE,
    TRAIN_END_DATE,
    TRAIN_FEATURES,
    TRAIN_START_DATE,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def wmape(y_true: pd.Series, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
    """Calculate weighted mean absolute percentage error.

    WMAPE is more stable than standard MAPE for demand forecasting because it
    does not explode on zero-sales days.

    Args:
        y_true: Actual target values.
        y_pred: Predicted target values.
        epsilon: Small value to avoid division by zero.

    Returns:
        Weighted mean absolute percentage error as a percentage.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    numerator = np.sum(np.abs(y_true_arr - y_pred_arr))
    denominator = max(np.sum(np.abs(y_true_arr)), epsilon)
    return float(numerator / denominator * 100)


def load_and_validate_data() -> pd.DataFrame:
    """Load the engineered dataset and validate required columns.

    Returns:
        Cleaned dataframe sorted by date.

    Raises:
        FileNotFoundError: If the input CSV does not exist.
        ValueError: If required columns are missing.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Input data file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).copy()

    required_columns = TRAIN_FEATURES + [TARGET_COLUMN, DATE_COLUMN]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    return df


def split_by_date(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into train and test periods using configured dates.

    Args:
        df: Input dataframe with a parsed date column.

    Returns:
        Tuple of (train_df, test_df).
    """
    train_df = df[
        (df[DATE_COLUMN] >= pd.Timestamp(TRAIN_START_DATE)) &
        (df[DATE_COLUMN] <= pd.Timestamp(TRAIN_END_DATE))
    ].copy()

    test_df = df[
        (df[DATE_COLUMN] >= pd.Timestamp(TEST_START_DATE)) &
        (df[DATE_COLUMN] <= pd.Timestamp(TEST_END_DATE))
    ].copy()

    return train_df, test_df


def main() -> None:
    """Run the full model training and evaluation pipeline."""
    try:
        logging.info("Loading engineered dataset...")
        df = load_and_validate_data()

        logging.info("Applying time-based split...")
        train_df, test_df = split_by_date(df)

        if train_df.empty or test_df.empty:
            raise ValueError(
                "Train or test split is empty. "
                "Please verify the configured date ranges and dataset coverage."
            )

        X_train = train_df[TRAIN_FEATURES]
        y_train = train_df[TARGET_COLUMN]
        X_test = test_df[TRAIN_FEATURES]
        y_test = test_df[TARGET_COLUMN]

        logging.info("Training Random Forest model...")
        model = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        demand_wmape = wmape(y_test, preds)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, MODEL_PATH)

        report = pd.DataFrame([
            {
                "model": "Random Forest",
                "mae": mae,
                "rmse": rmse,
                "wmape": demand_wmape,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_start_date": TRAIN_START_DATE,
                "train_end_date": TRAIN_END_DATE,
                "test_start_date": TEST_START_DATE,
                "test_end_date": TEST_END_DATE,
            }
        ])
        report.to_csv(REPORT_PATH, index=False)

        logging.info("Saved model to %s", MODEL_PATH)
        logging.info("Saved report to %s", REPORT_PATH)
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"WMAPE: {demand_wmape:.2f}%")

    except Exception as exc:
        logging.exception("Training pipeline failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
