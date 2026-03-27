from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed_sales_features_v2.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_model.pkl"
REPORT_PATH = PROJECT_ROOT / "reports" / "item_sales_model_results_v2.csv"

FEATURES = [
    "lag_1", "lag_3", "lag_7", "lag_14", "lag_28",
    "rolling_7", "rolling_14", "rolling_28",
    "weekday", "weekend", "stock_ratio",
    "promo_price_interaction", "lag1_stock_interaction",
]
TARGET = "units_sold"

def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Expected feature dataset at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    pd.DataFrame([{
        "model": "RandomForestRegressor",
        "mae": mae,
        "mape": mape
    }]).to_csv(REPORT_PATH, index=False)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved report to {REPORT_PATH}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}")

if __name__ == "__main__":
    main()
