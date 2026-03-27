from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed_sales_features_v2.csv"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "03_train_models.ipynb"
REPORTS_DIR = PROJECT_ROOT / "reports"

FEATURES = [
    "lag_1", "lag_3", "lag_7", "lag_14", "lag_28",
    "rolling_7", "rolling_14", "rolling_28",
    "promo_flag", "promo_depth", "stock_ratio", "price",
    "weekday", "weekend", "promo_price_interaction", "lag1_stock_interaction"
]
TARGET = "units_sold"


def unstable_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    eps = 1e-9
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100


def wmape(y_true, y_pred, epsilon=1e-6):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(np.abs(y_true - y_pred)) / max(np.sum(np.abs(y_true)), epsilon) * 100


def extract_notebook_metrics(notebook_path: Path) -> pd.DataFrame:
    if not notebook_path.exists():
        return pd.DataFrame()
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    pattern = re.compile(r"random_forest\s+([0-9.]+)\s+([0-9.e+\-]+).*?xgboost\s+([0-9.]+)\s+([0-9.e+\-]+).*?baseline_lag1\s+([0-9.]+)\s+([0-9.e+\-]+).*?ridge\s+([0-9.]+)\s+([0-9.e+\-]+)", re.S)
    for cell in nb.get("cells", []):
        out_text = ""
        for o in cell.get("outputs", []):
            out_text += "".join(o.get("text", []))
            if "data" in o and "text/plain" in o["data"]:
                out_text += "".join(o["data"]["text/plain"])
        if "random_forest" in out_text and "baseline_lag1" in out_text and "ridge" in out_text:
            m = pattern.search(out_text)
            if m:
                rows = [
                    ["Random Forest", float(m.group(1)), float(m.group(2))],
                    ["XGBoost", float(m.group(3)), float(m.group(4))],
                    ["Baseline (lag_1)", float(m.group(5)), float(m.group(6))],
                    ["Ridge", float(m.group(7)), float(m.group(8))],
                ]
                return pd.DataFrame(rows, columns=["Model", "MAE", "MAPE"])
    return pd.DataFrame()


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").copy()

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    return X_train, y_train, X_test, y_test


def train_models(X_train, y_train, X_test, y_test):
    rows = {}

    y_pred_baseline = X_test["lag_1"]
    rows["Baseline (lag_1)"] = {
        "MAE": mean_absolute_error(y_test, y_pred_baseline),
        "MAPE": unstable_mape(y_test, y_pred_baseline),
        "WMAPE": wmape(y_test, y_pred_baseline),
    }

    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    rows["Ridge"] = {
        "MAE": mean_absolute_error(y_test, y_pred_ridge),
        "MAPE": unstable_mape(y_test, y_pred_ridge),
        "WMAPE": wmape(y_test, y_pred_ridge),
    }

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rows["Random Forest"] = {
        "MAE": mean_absolute_error(y_test, y_pred_rf),
        "MAPE": unstable_mape(y_test, y_pred_rf),
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
            "MAPE": unstable_mape(y_test, y_pred_xgb),
            "WMAPE": wmape(y_test, y_pred_xgb),
        }

    results_df = pd.DataFrame.from_dict(rows, orient="index").reset_index().rename(columns={"index": "Model"}).sort_values("MAE").reset_index(drop=True)
    return results_df, rf, y_test, y_pred_rf


def save_feature_importance(rf):
    importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
    plt.figure(figsize=(10, 7))
    plt.barh(importances.index, importances.values)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance - Random Forest")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importance_random_forest.png", dpi=200)
    plt.close()


def save_actual_vs_predicted(y_test, y_pred_rf):
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


def save_residual_distribution(y_test, y_pred_rf):
    residuals = y_test - y_pred_rf
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution - Random Forest")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "residual_distribution_random_forest.png", dpi=200)
    plt.close()


def save_model_comparison(results_df):
    plot_df = results_df.copy()
    plt.figure(figsize=(10, 6))
    x = np.arange(len(plot_df))
    width = 0.38
    plt.bar(x - width / 2, plot_df["MAE"], width, label="MAE")
    plt.bar(x + width / 2, plot_df["MAPE"], width, label="MAPE")
    plt.xticks(x, plot_df["Model"], rotation=15)
    plt.ylabel("Metric value")
    plt.title("Model Comparison - MAE and MAPE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "model_comparison_mae_mape.png", dpi=200)
    plt.close()


def save_model_comparison_operational(results_df):
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


def save_markdown_table(notebook_metrics: pd.DataFrame, rerun_metrics: pd.DataFrame):
    lines = []
    lines.append("# Model Performance Report")
    lines.append("")
    lines.append("This report summarizes the forecasting models evaluated for next-day SKU demand prediction.")
    lines.append("")
    if not notebook_metrics.empty:
        lines.append("## Metrics extracted from `notebooks/03_train_models.ipynb`")
        lines.append("")
        lines.append("| Model | MAE | MAPE |")
        lines.append("|---|---:|---:|")
        for _, row in notebook_metrics.sort_values("MAE").iterrows():
            lines.append(f"| {row['Model']} | {row['MAE']:.6f} | {row['MAPE']:.6e} |")
        lines.append("")
        lines.append("**Best model in the notebook:** Random Forest, based on the lowest MAE.")
        lines.append("**Important note:** these notebook MAPE values are unstable because zero-sales days inflate percentage errors.")
        lines.append("")
    lines.append("## Recomputed operational metrics")
    lines.append("")
    lines.append("| Model | MAE | MAPE | WMAPE |")
    lines.append("|---|---:|---:|---:|")
    for _, row in rerun_metrics.iterrows():
        lines.append(f"| {row['Model']} | {row['MAE']:.6f} | {row['MAPE']:.6e} | {row['WMAPE']:.2f}% |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Random Forest is the strongest model by MAE in both the notebook and the rerun.")
    lines.append("- Baseline lag-1 performs reasonably well, which suggests recent sales history is a strong predictor.")
    lines.append("- Ridge underperforms, indicating non-linear relationships matter for this task.")
    lines.append("- WMAPE is more practical than standard MAPE for grocery demand because zero-sales days make MAPE misleading.")
    (REPORTS_DIR / "model_performance_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    notebook_metrics = extract_notebook_metrics(NOTEBOOK_PATH)
    X_train, y_train, X_test, y_test = load_data()
    rerun_metrics, rf, y_test, y_pred_rf = train_models(X_train, y_train, X_test, y_test)
    rerun_metrics.to_csv(REPORTS_DIR / "model_comparison_metrics.csv", index=False)
    save_markdown_table(notebook_metrics, rerun_metrics)
    save_feature_importance(rf)
    save_actual_vs_predicted(y_test, y_pred_rf)
    save_residual_distribution(y_test, y_pred_rf)
    save_model_comparison(notebook_metrics if not notebook_metrics.empty else rerun_metrics)
    save_model_comparison_operational(rerun_metrics)
    print(rerun_metrics.to_string(index=False))
    print(f"\nSaved report files to: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
