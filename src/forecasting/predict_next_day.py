"""Generate a next-day demand prediction using the trained model.

This script loads the saved Random Forest model and predicts next-day
`units_sold` for a single feature row.

Expected input:
- a dictionary containing the required prediction features

Required model artifact:
- models/random_forest_model.pkl
"""

from __future__ import annotations

import logging
from typing import Dict

import joblib
import pandas as pd

from src.config import MODEL_PATH, TRAIN_FEATURES


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def predict_next_day(feature_row: Dict[str, float]) -> float:
    """Predict next-day units sold for a single observation.

    Args:
        feature_row: Dictionary containing one row of model input features.

    Returns:
        Predicted units sold as a float.

    Raises:
        FileNotFoundError: If the trained model file does not exist.
        ValueError: If required features are missing.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    missing = [feature for feature in TRAIN_FEATURES if
               feature not in feature_row]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    model = joblib.load(MODEL_PATH)
    X_new = pd.DataFrame([feature_row])[TRAIN_FEATURES]
    prediction = model.predict(X_new)[0]
    return float(prediction)


def main() -> None:
    """Run an example next-day prediction."""
    example = {
        "price": 3.49,
        "promo_flag": 1,
        "promo_depth": 0.20,
        "stockout_flag": 0,
        "lag_1": 20,
        "lag_3": 18,
        "lag_7": 22,
        "lag_14": 19,
        "lag_28": 17,
        "rolling_7": 20.1,
        "rolling_14": 19.4,
        "rolling_28": 18.8,
        "weekday": 2,
        "weekend": 0,
        "stock_ratio": 1.0,
        "promo_price_interaction": 3.49,
        "lag1_stock_interaction": 20.0,
    }

    try:
        prediction = predict_next_day(example)
        logging.info("Prediction completed successfully.")
        print(f"Predicted next-day units sold: {prediction:.2f}")
    except Exception as exc:
        logging.exception("Prediction failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
