from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_model.pkl"

FEATURES = [
    "lag_1", "lag_3", "lag_7", "lag_14", "lag_28",
    "rolling_7", "rolling_14", "rolling_28",
    "weekday", "weekend", "stock_ratio",
    "promo_price_interaction", "lag1_stock_interaction",
]

def predict_next_day(feature_row: dict) -> float:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    missing = [c for c in FEATURES if c not in feature_row]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X_new = pd.DataFrame([feature_row])[FEATURES]
    pred = model.predict(X_new)[0]
    return float(pred)

if __name__ == "__main__":
    example = {
        "lag_1": 12, "lag_3": 10, "lag_7": 14, "lag_14": 11, "lag_28": 13,
        "rolling_7": 12.1, "rolling_14": 11.8, "rolling_28": 12.4,
        "weekday": 2, "weekend": 0, "stock_ratio": 1.0,
        "promo_price_interaction": 0.0, "lag1_stock_interaction": 12.0,
    }
    print(f"Predicted next-day units sold: {predict_next_day(example):.2f}")
