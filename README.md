# Delivery ETA and Demand Forecasting

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A machine learning project for next-day grocery demand forecasting in Wolt-like retail operations, using historical sales, promotions, pricing, and stock availability to support better replenishment and inventory decisions.

Repository: [delivery-eta-and-demand-forecasting](https://github.com/Taanvir-Ahmed/delivery-eta-and-demand-forecasting)

---

## Overview

This project addresses a practical retail forecasting problem:

**Can we predict next-day item demand for each SKU at each venue?**

Reliable demand forecasts help operations teams:
- plan replenishment more effectively
- reduce stockouts
- avoid overstocking
- support promotion-aware inventory decisions

The project combines:
- exploratory data analysis
- feature engineering
- model training and comparison
- time-based evaluation
- reproducible Python scripts for training and inference
- reporting visuals and documentation

---

## Business Context

### Problem
Forecast next-day `units_sold` for grocery items at SKU and venue level.

### Who uses the output
- inventory planning teams
- replenishment planners
- operations analysts
- category managers
- demand planning teams

### What decisions it supports
- how much stock to prepare for tomorrow
- which SKUs may need replenishment
- where promotions may increase demand
- where stock risk may lead to lost sales

### Why it matters
Retail and quick-commerce operations depend on balancing product availability with efficient inventory use. Demand forecasting helps make these decisions more data-driven.

---

## Dataset

This project uses the item sales dataset:

- `data/grocery_sales_autumn_2025.csv`

### Dataset summary
- approximately **40,950 rows**
- daily grocery sales records
- time period: **September 1, 2025 to November 30, 2025**
- target variable: **`units_sold`**

### Main columns
- `date`
- `venue_id`
- `sku_id`
- `phl1_id`, `phl2_id`, `phl3_id`
- `country_id`
- `price`
- `promo_flag`
- `promo_depth`
- `operating_minutes`
- `in_stock_minutes`
- `stockout_flag`
- `units_sold`

For full column documentation, see:
- [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md)

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Taanvir-Ahmed/delivery-eta-and-demand-forecasting.git
cd delivery-eta-and-demand-forecasting
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

Run from the project root:

```bash
python -m src.forecasting.train_model
```

Expected output:
- trained model saved to `models/random_forest_model.pkl`
- evaluation report saved to `reports/item_sales_model_results_v2.csv`

### 4. Run a next-day prediction example

```bash
python -m src.forecasting.predict_next_day
```

Expected output:
- a predicted next-day `units_sold` value printed in the terminal

---

## Key Results

Final Random Forest model performance on a time-based holdout set:

- **MAE:** 2.0679
- **RMSE:** 6.0986
- **WMAPE:** 35.04%

These results indicate that the model can provide useful next-day demand estimates while preserving a realistic time-based evaluation setup.

---

## Project Highlights

- practical retail demand forecasting use case
- business-oriented framing for grocery operations
- combines EDA, feature engineering, modeling, and reporting
- includes both notebooks and reproducible Python scripts
- uses time-based validation to avoid leakage
- compares multiple approaches:
  - Baseline (`lag_1`)
  - Ridge Regression
  - Random Forest
  - XGBoost
- includes documentation for:
  - raw data
  - engineered features
  - project architecture
- includes reporting visuals in the `reports/` folder

---

## Project Structure

```text
.
├── ARCHITECTURE.md
├── DATA_DICTIONARY.md
├── FEATURES.md
├── README.md
├── data/
│   ├── grocery_sales_autumn_2025.csv
│   ├── processed_sales_features.csv
│   └── processed_sales_features_v2.csv
├── models/
│   └── random_forest_model.pkl
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_train_models.ipynb
├── reports/
│   ├── item_sales_model_results_v2.csv
│   ├── feature_importance_random_forest.png
│   ├── actual_vs_predicted_random_forest.png
│   ├── residual_distribution_random_forest.png
│   ├── model_comparison_mae_mape.png
│   └── model_comparison_mae_wmape.png
├── sql/
│   ├── 00_create_item_sales_table.sql
│   ├── 01_validation.sql
│   └── 02_business_analysis.sql
└── src/
    ├── config.py
    └── forecasting/
        ├── __init__.py
        ├── train_model.py
        └── predict_next_day.py
```

---

## Methodology

### 1. Exploratory Data Analysis
The EDA notebook examines:
- sales distribution
- top-selling SKUs
- weekday demand patterns
- promotion effects
- stockout effects
- daily sales trends over time

Notebook:
- `notebooks/01_eda.ipynb`

### 2. Feature Engineering
To improve forecasting performance, the project creates features that capture recent demand, trend, calendar effects, and business context.

Engineered features include:
- lag features: `lag_1`, `lag_3`, `lag_7`, `lag_14`, `lag_28`
- rolling averages: `rolling_7`, `rolling_14`, `rolling_28`
- calendar features: `weekday`, `weekend`
- context features:
  - `stock_ratio`
  - `promo_price_interaction`
  - `lag1_stock_interaction`

Notebook:
- `notebooks/02_feature_engineering.ipynb`

For full details, see:
- [`FEATURES.md`](FEATURES.md)

### 3. Model Training
The project compares multiple forecasting approaches:
- Baseline using `lag_1`
- Ridge Regression
- Random Forest
- XGBoost

Notebook:
- `notebooks/03_train_models.ipynb`

Reproducible script:
- `src/forecasting/train_model.py`

### 4. Evaluation Strategy
A time-based split is used instead of a random split.

- Train period: **September 6, 2025 to November 15, 2025**
- Test period: **November 16, 2025 to November 30, 2025**

This setup is important because it better reflects real forecasting, where future demand must be predicted using only past information.

### 5. Evaluation Metrics
The final project focuses on:
- **MAE** for directly interpretable unit error
- **RMSE** to penalize larger misses
- **WMAPE** as a stable demand-weighted percentage error

Standard MAPE was explored earlier but is not emphasized in the final project because zero-sales days make it unstable and less useful for this type of demand forecasting task.

---

## Results

### Best model
**Random Forest** was selected as the final practical model.

### Why it was chosen
- strong performance on engineered tabular features
- better practical error than simpler baselines
- works well with mixed business and time-based signals
- supports feature importance analysis for interpretability

### Reporting visuals
The `reports/` folder includes:
- feature importance chart
- actual vs predicted scatter plot
- residual distribution histogram
- model comparison charts

---

## How to Use

### Retrain the model

```bash
python -m src.forecasting.train_model
```

### Run an example prediction

```bash
python -m src.forecasting.predict_next_day
```

### Explore the notebooks
- `notebooks/01_eda.ipynb`
- `notebooks/02_feature_engineering.ipynb`
- `notebooks/03_train_models.ipynb`

### Review the reports
Check the `reports/` folder for:
- saved model evaluation outputs
- feature importance visualization
- actual vs predicted plot
- residual analysis
- model comparison charts

---

## Documentation

Additional project documentation is included in:

- [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md)
- [`FEATURES.md`](FEATURES.md)
- [`ARCHITECTURE.md`](ARCHITECTURE.md)

These files explain:
- the dataset columns
- the engineered features
- the overall ML pipeline and project design

---

## Limitations

This project has several known limitations:

- only around three months of data
- simulated dataset rather than real production data
- limited external signals such as holidays, weather, or local events
- possible weaker performance for cold-start SKUs
- no probabilistic prediction intervals yet
- feature engineering is useful, but not yet a full production feature pipeline

These limitations are important because real-world demand forecasting systems usually require broader historical coverage and more external context.

---

## Next Steps

If this project were extended further, useful next improvements would include:

- adding holiday and seasonal signals
- improving cold-start handling for new SKUs
- monitoring performance by SKU and venue
- adding automated tests
- adding lightweight CI checks
- improving reporting automation
- experimenting with more advanced forecasting approaches

---

## License

MIT License
