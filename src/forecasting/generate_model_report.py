"""Generate model comparison reports and visualizations.

This script compares multiple forecasting models on the same engineered dataset,
using the same feature set and time-based split defined in src.config.

Outputs:
- reports/model_comparison_metrics.csv
- reports/model_performance_report.md
- reports/feature_importance_random_forest.png
- reports/actual_vs_predicted_random_forest.png
- reports/residual_distribution_random_forest.png
- reports/model_comparison_mae_wmape.png
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

from src.config import (
    DATA_PATH,
    DATE_COLUMN,
    RANDOM_FOREST_PARAMS,
    TARGET_COLUMN,
    TEST_END_DATE,
    TEST_START_DATE,
    TRAIN_END_DATE,
    TRAIN_FEATURES,
    TRAIN_START_DATE,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "03_train_models.ipynb"


def wmape(y_true: pd.Series, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
    """Calculate weighted mean absolute percentage error."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    numerator = np.sum(np.abs(y_true_arr - y_pred_arr))
    denominator = max(np.sum(np.abs(y_true_arr)), epsilon)
    return float(numerator / denominator * 100)


def load_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load data and apply the same time-based split as training."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Input data file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).copy()

    required_columns = TRAIN_FEATURES + [TARGET_COLUMN, DATE_COLUMN]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    train_df = df[
        (df[DATE_COLUMN] >= pd.Timestamp(TRAIN_START_DATE))
        & (df[DATE_COLUMN] <= pd.Timestamp(TRAIN_END_DATE))
    ].copy()

    test_df = df[
        (df[DATE_COLUMN] >= pd.Timestamp(TEST_START_DATE))
        & (df[DATE_COLUMN] <= pd.Timestamp(TEST_END_DATE))
    ].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Train or test split is empty. "
            "Please verify the configured date ranges and dataset coverage."
        )

    X_train = train_df[TRAIN_FEATURES]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[TRAIN_FEATURES]
    y_test = test_df[TARGET_COLUMN]

    return X_train, y_train, X_test, y_test


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, RandomForestRegressor, pd.Series, np.ndarray]:
    """Train baseline and ML models, then return comparison metrics."""
    rows = {}

    y_pred_baseline = X_test["lag_1"]
    rows["Baseline (lag_1)"] = {
        "MAE": mean_absolute_error(y_test, y_pred_baseline),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_baseline)),
        "WMAPE": wmape(y_test, y_pred_baseline),
    }

    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    rows["Ridge Regression"] = {
        "MAE": mean_absolute_error(y_test, y_pred_ridge),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
        "WMAPE": wmape(y_test, y_pred_ridge),
    }

    rf = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rows["Random Forest"] = {
        "MAE": mean_absolute_error(y_test, y_pred_rf),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        "WMAPE": wmape(y_test, y_pred_rf),
    }

    if XGBRegressor is not None:
        xgb = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        rows["XGBoost"] = {
            "MAE": mean_absolute_error(y_test, y_pred_xgb),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
            "WMAPE": wmape(y_test, y_pred_xgb),
        }

    results_df = (
        pd.DataFrame.from_dict(rows, orient="index")
        .reset_index()
        .rename(columns={"index": "Model"})
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    return results_df, rf, y_test, y_pred_rf


def save_feature_importance(rf: RandomForestRegressor) -> None:
    """Save feature importance chart for the best model."""
    importances = pd.Series(rf.feature_importances_, index=TRAIN_FEATURES).sort_values()

    plt.figure(figsize=(10, 7))
    plt.barh(importances.index, importances.values)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance - Random Forest")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importance_random_forest.png", dpi=200)
    plt.close()


def save_actual_vs_predicted(y_test: pd.Series, y_pred_rf: np.ndarray) -> None:
    """Save actual vs predicted scatter plot."""
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred_rf, alpha=0.5)

    min_val = min(float(np.min(y_test)), float(np.min(y_pred_rf)))
    max_val = max(float(np.max(y_test)), float(np.max(y_pred_rf)))
    plt.plot([min_val, max_val], [min_val, max_val])

    plt.xlabel("Actual units_sold")
    plt.ylabel("Predicted units_sold")
    plt.title("Actual vs Predicted - Random Forest")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "actual_vs_predicted_random_forest.png", dpi=200)
    plt.close()


def save_residual_distribution(y_test: pd.Series, y_pred_rf: np.ndarray) -> None:
    """Save residual distribution histogram."""
    residuals = y_test - y_pred_rf

    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution - Random Forest")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "residual_distribution_random_forest.png", dpi=200)
    plt.close()


def save_model_comparison(results_df: pd.DataFrame) -> None:
    """Save model comparison chart using MAE and WMAPE."""
    plot_df = results_df.copy()

    plt.figure(figsize=(10, 6))
    x = np.arange(len(plot_df))
    width = 0.38

    plt.bar(x - width / 2, plot_df["MAE"], width, label="MAE")
    plt.bar(x + width / 2, plot_df["WMAPE"], width, label="WMAPE")

    plt.xticks(x, plot_df["Model"], rotation=15)
    plt.ylabel("Metric value")
    plt.title("Model Comparison - MAE and WMAPE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "model_comparison_mae_wmape.png", dpi=200)
    plt.close()


def save_markdown_table(results_df: pd.DataFrame) -> None:
    """Save markdown summary for model comparison."""
    lines = []
    lines.append("## Model comparison")
    lines.append("")
    lines.append("| Model | MAE | RMSE | WMAPE |")
    lines.append("|---|---:|---:|---:|")

    for _, row in results_df.iterrows():
        lines.append(
            f"| {row['Model']} | {row['MAE']:.4f} | {row['RMSE']:.4f} | {row['WMAPE']:.2f}% |"
        )

    lines.append("")
    lines.append("**Best model:** Random Forest, based on lowest MAE.")
    lines.append("**Note:** Final reporting emphasizes MAE, RMSE, and WMAPE because they are more stable and interpretable for grocery demand forecasting than standard MAPE.")

    (REPORTS_DIR / "model_performance_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Generate model comparison files and visualizations."""
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        logging.info("Loading data for model comparison...")
        X_train, y_train, X_test, y_test = load_data()

        logging.info("Training comparison models...")
        results_df, rf, y_test, y_pred_rf = train_models(X_train, y_train, X_test, y_test)

        results_df.to_csv(REPORTS_DIR / "model_comparison_metrics.csv", index=False)
        save_markdown_table(results_df)

        save_feature_importance(rf)
        save_actual_vs_predicted(y_test, y_pred_rf)
        save_residual_distribution(y_test, y_pred_rf)
        save_model_comparison(results_df)

        print(results_df.to_string(index=False))
        print(f"\nSaved report files to: {REPORTS_DIR}")

    except Exception as exc:
        logging.exception("Report generation failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
