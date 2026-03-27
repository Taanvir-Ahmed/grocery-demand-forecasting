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