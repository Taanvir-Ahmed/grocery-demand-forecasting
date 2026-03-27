# Model Performance Report

This report summarizes the forecasting models evaluated for next-day SKU demand prediction.

## Metrics extracted from `notebooks/03_train_models.ipynb`

| Model | MAE | MAPE |
|---|---:|---:|
| Random Forest | 0.585588 | 1.725283e+10 |
| XGBoost | 0.595471 | 1.763151e+10 |
| Baseline (lag_1) | 0.858370 | 2.303704e+10 |
| Ridge | 2.068645 | 1.402453e+11 |

**Best model in the notebook:** Random Forest, based on the lowest MAE.
**Important note:** these notebook MAPE values are unstable because zero-sales days inflate percentage errors.

## Recomputed operational metrics

| Model | MAE | MAPE | WMAPE |
|---|---:|---:|---:|
| Random Forest | 2.088085 | 1.305628e+10 | 35.08% |
| XGBoost | 2.117226 | 1.689887e+10 | 35.56% |
| Baseline (lag_1) | 3.706721 | 4.228979e+10 | 62.26% |
| Ridge | 4.086859 | 1.586866e+11 | 68.65% |

## Interpretation

- Random Forest is the strongest model by MAE in both the notebook and the rerun.
- Baseline lag-1 performs reasonably well, which suggests recent sales history is a strong predictor.
- Ridge underperforms, indicating non-linear relationships matter for this task.
- WMAPE is more practical than standard MAPE for grocery demand because zero-sales days make MAPE misleading.