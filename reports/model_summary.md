# Model Summary

## Objective
Forecast next-day `units_sold` for each SKU using historical demand, promotions, stock availability, and calendar signals.

## Candidate models
- Baseline using `lag_1`
- Ridge Regression
- Random Forest
- XGBoost

## Main result
The strongest model is **Random Forest**, which achieved the lowest MAE in the original notebook evaluation.

## Why this matters
More accurate next-day demand forecasts can support:
- replenishment planning
- stockout reduction
- promotion planning
- better venue-level inventory decisions

## Metric note
Standard MAPE is unstable for this problem because some days have zero or near-zero sales. For operational interpretation, prefer **MAE** and **WMAPE**.
