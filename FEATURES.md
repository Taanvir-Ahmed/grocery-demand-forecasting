# Features

This document explains the feature engineering pipeline used for next-day grocery demand forecasting.

## Goal

The goal of feature engineering is to transform raw daily sales data into useful predictive signals for forecasting `units_sold`.

## Target variable

- `units_sold`

## Engineered features

### 1. Lag features

Lag features capture historical demand from previous days.

Created features:
- `lag_1`
- `lag_3`
- `lag_7`
- `lag_14`
- `lag_28`

Why they were created:
- demand forecasting is strongly influenced by recent sales history
- short lags capture immediate momentum
- weekly and multi-week lags help capture recurring patterns

Example:
- `lag_1` = units sold yesterday
- `lag_7` = units sold one week ago

## 2. Rolling average features

Rolling averages smooth recent demand and reduce noise.

Created features:
- `rolling_7`
- `rolling_14`
- `rolling_28`

Why they were created:
- daily sales can be volatile
- rolling means help summarize recent trends
- wider windows provide more stable demand signals

## 3. Calendar features

Calendar features capture weekly patterns.

Created features:
- `weekday`
- `weekend`

Why they were created:
- grocery demand often varies by day of week
- weekends can have different shopping behavior than weekdays

## 4. Context and business features

These features add operational and pricing context.

### `stock_ratio`
Defined as:

`in_stock_minutes / operating_minutes`

Why it was created:
- measures how available the item was during the day
- helps distinguish low sales caused by low demand from low sales caused by lack of stock

### `promo_price_interaction`
Combines promotion activity and price.

Why it was created:
- promotion effects often depend on the actual item price
- interaction terms help model non-linear business effects

### `lag1_stock_interaction`
Interaction between yesterday's sales and stock availability.

Why it was created:
- combines recent demand with operational availability
- useful when recent demand should be interpreted together with stock conditions

## Features used for model training

The final training feature set includes:

- `price`
- `promo_flag`
- `promo_depth`
- `stockout_flag`
- `lag_1`
- `lag_3`
- `lag_7`
- `lag_14`
- `lag_28`
- `rolling_7`
- `rolling_14`
- `rolling_28`
- `weekday`
- `weekend`
- `stock_ratio`
- `promo_price_interaction`
- `lag1_stock_interaction`

## Data handling decisions

### Dropping the first 28 days
Because `lag_28` and `rolling_28` require historical observations, the first 28 days do not have complete values.

Decision:
- rows with missing lag and rolling features were dropped

Why this is reasonable:
- models need complete inputs
- the removed rows are a natural consequence of time-series feature engineering

## Train/test split strategy

A time-based split was used instead of a random split.

Train period:
- September 6, 2025 to November 15, 2025

Test period:
- November 16, 2025 to November 30, 2025

Why this was done:
- prevents data leakage
- better simulates real forecasting, where future data is not available during training

## Why these features make sense

This feature set combines:
- recent demand history
- smoothed trends
- calendar effects
- price and promotion signals
- stock and availability signals

Together, these features create a practical and business-relevant representation for next-day demand forecasting.