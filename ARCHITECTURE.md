# Architecture

This document explains the end-to-end machine learning pipeline used in this project.

## Problem definition

This project focuses on next-day grocery demand forecasting.

Business question:
- Can we predict the number of units sold tomorrow for a given SKU at a given venue?

Why it matters:
- better inventory planning
- improved replenishment decisions
- fewer stockouts
- reduced waste from over-ordering

## Prediction target

- `units_sold`

The model predicts daily item demand at SKU and venue level.

## End-to-end pipeline

```text
Raw grocery sales CSV
    ->
Exploratory data analysis
    ->
Feature engineering
    ->
Time-based train/test split
    ->
Model training
    ->
Model evaluation
    ->
Best model selection
    ->
Saved model for inference
    ->
Next-day prediction script
```

## Data flow

### 1. Raw data
Input file:
- `data/grocery_sales_autumn_2025.csv`

This dataset contains daily sales records with pricing, promotions, stock availability, and product hierarchy information.

### 2. Exploratory data analysis
Notebook:
- `notebooks/01_eda.ipynb`

Main goals:
- understand sales distributions
- inspect promotion effects
- inspect stockout effects
- observe daily and weekday sales patterns

### 3. Feature engineering
Notebook:
- `notebooks/02_feature_engineering.ipynb`

Main outputs:
- lag features
- rolling averages
- calendar features
- stock and promotion context features

Saved engineered dataset:
- `data/processed_sales_features_v2.csv`

### 4. Model training
Notebook:
- `notebooks/03_train_models.ipynb`

Script:
- `src/forecasting/train_model.py`

Models compared:
- Baseline (`lag_1`)
- Ridge Regression
- Random Forest
- XGBoost

### 5. Evaluation
The project evaluates models on a time-based holdout set.

Primary metrics:
- MAE
- RMSE
- WMAPE

Why these metrics:
- **MAE** is easy to interpret in units sold
- **RMSE** penalizes larger errors more strongly
- **WMAPE** gives a demand-weighted percentage error and is more stable than standard MAPE

Note:
- standard MAPE was explored earlier, but it is unstable when actual sales are zero or near zero

### 6. Best model
Best practical model:
- Random Forest

Reason:
- strong predictive performance
- robust performance on engineered tabular features
- easy to interpret with feature importance

### 7. Inference
Script:
- `src/forecasting/predict_next_day.py`

Purpose:
- load the saved model
- prepare one feature row
- generate next-day demand prediction

## Modeling workflow

### Training
1. Load engineered data
2. Parse and sort by date
3. Select training features and target
4. Apply time-based split
5. Train Random Forest model
6. Evaluate on holdout set
7. Save model and metrics report

### Prediction
1. Load trained model
2. Build feature row with required columns
3. Match feature order with training schema
4. Predict next-day `units_sold`

## Current results

Current script-based evaluation:
- MAE: 2.0679
- RMSE: 6.0986
- WMAPE: 35.04%

These results show that the model can provide practical next-day forecasts, although there is still room for improvement.

## Limitations

Known limitations:
- only about three months of data
- simulated dataset rather than real production data
- limited external signals such as holidays, weather, or local events
- likely weaker performance for cold-start SKUs or changing demand patterns
- no probabilistic prediction intervals

## Production considerations

If this project were deployed in production, the next steps would be:
- retrain the model regularly
- monitor forecast accuracy by SKU and venue
- add holiday and seasonal signals
- improve handling of new SKUs
- compare against stronger forecasting baselines
- expose predictions through a batch pipeline or API

## Project structure relevance

This project combines:
- notebooks for exploration and experimentation
- Python scripts for reproducible training and prediction
- SQL for data preparation and validation
- markdown documentation for communication and maintainability

This structure makes the project easier to understand and more suitable for portfolio presentation.