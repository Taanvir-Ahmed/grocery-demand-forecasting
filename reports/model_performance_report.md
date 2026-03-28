## Model comparison

| Model | MAE | RMSE | WMAPE |
|---|---:|---:|---:|
| Random Forest | 2.0679 | 6.0986 | 35.04% |
| XGBoost | 2.0755 | 6.0717 | 35.17% |
| Baseline (lag_1) | 3.6806 | 10.5525 | 62.37% |
| Ridge Regression | 4.0652 | 8.3887 | 68.89% |

**Best model:** Random Forest, based on lowest MAE.
**Note:** Final reporting emphasizes MAE, RMSE, and WMAPE because they are more stable and interpretable for grocery demand forecasting than standard MAPE.