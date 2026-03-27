# Grocery Demand Forecasting

A portfolio-ready machine learning project built on simulated daily grocery item sales data. The goal is to forecast next-day SKU-level demand so inventory and operations teams can make better replenishment and promotion decisions.

## Business context
Wolt-like grocery operations need short-term demand forecasts to reduce stockouts, improve item availability, and support smarter daily planning. This project frames the task as next-day item sales forecasting at the SKU and venue level.

## Dataset
This project uses `grocery_sales_autumn_2025.csv`, a simulated daily grocery sales dataset from venues in Finland. Key columns include:
- `date`
- `venue_id`, `sku_id`
- `phl1_id`, `phl2_id`, `phl3_id`
- `price`, `promo_flag`, `promo_depth`
- `operating_minutes`, `in_stock_minutes`, `stockout_flag`
- `units_sold`

## Project structure
```
.
├── data/
├── models/
├── notebooks/
├── reports/
├── sql/
└── src/forecasting/
```

## Workflow
1. Exploratory analysis in `notebooks/01_eda.ipynb`
2. Feature engineering in `notebooks/02_feature_engineering.ipynb`
3. Model training and evaluation in `notebooks/03_train_models.ipynb`
4. Reproducible training and prediction scripts in `src/forecasting/`

## Features used
- lag features: `lag_1`, `lag_3`, `lag_7`, `lag_14`, `lag_28`
- rolling means: `rolling_7`, `rolling_14`, `rolling_28`
- calendar signals: `weekday`, `weekend`
- stock signal: `stock_ratio`
- interaction features: `promo_price_interaction`, `lag1_stock_interaction`

## Models evaluated
- Baseline using lag-1
- Ridge Regression
- Random Forest
- XGBoost

## Suggested report artifacts
Add these to `reports/` for a stronger portfolio presentation:
- model comparison table
- feature importance plot
- actual vs predicted plot
- short summary of business implications

## Running the scripts
Train model:
```bash
python src/forecasting/train_model.py
```

Run prediction example:
```bash
python src/forecasting/predict_next_day.py
```
