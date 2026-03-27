from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


def wmape(y_true, y_pred, epsilon=1e-6):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(np.abs(y_true - y_pred)) / max(np.sum(np.abs(y_true)), epsilon) * 100


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed_sales_features_v2.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_model.pkl"
REPORT_PATH = PROJECT_ROOT / "reports" / "item_sales_model_results_v2.csv"


def main():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").copy()

    target = "units_sold"
    feature_cols = [
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

    missing = [c for c in feature_cols + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols]
    y_train = train_df[target]
    X_test = test_df[feature_cols]
    y_test = test_df[target]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
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
        }
    ])
    report.to_csv(REPORT_PATH, index=False)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved report to {REPORT_PATH}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"WMAPE: {demand_wmape:.2f}%")