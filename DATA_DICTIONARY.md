# Data Dictionary

This document describes the columns used in the `grocery_sales_autumn_2025.csv` dataset.

## Dataset overview

- Dataset: `grocery_sales_autumn_2025.csv`
- Granularity: daily sales per SKU and venue
- Time period: September 1, 2025 to November 30, 2025
- Rows: 40,950
- Target variable: `units_sold`

## Column definitions

| Column | Type | Description | Example | Notes |
|---|---|---|---|---|
| `date` | date | Calendar date of the sales record | `2025-09-01` | Used for time-based analysis and train/test split |
| `venue_id` | integer | Unique identifier for the venue or store | `1012` | One venue can sell many SKUs |
| `sku_id` | integer | Unique identifier for a product SKU | `450321` | Main item-level identifier |
| `phl1_id` | integer | Product hierarchy level 1 | `10` | Broad product category |
| `phl2_id` | integer | Product hierarchy level 2 | `104` | Mid-level product category |
| `phl3_id` | integer | Product hierarchy level 3 | `1047` | Most detailed category level in the dataset |
| `country_id` | integer | Country identifier for the venue | `1` | In this case, grocery venues are in Finland |
| `price` | float | Unit price in EUR | `3.49` | Used as a demand signal |
| `promo_flag` | integer / binary | Whether a promotion is active | `0` or `1` | 1 means promoted |
| `promo_depth` | float | Promotion depth as a percentage | `0.20` | Example means 20% discount |
| `operating_minutes` | integer | Total operating minutes of the venue on that day | `900` | Used to normalize stock availability |
| `in_stock_minutes` | integer | Minutes the SKU was available in stock | `900` | Can be lower if stockouts occur |
| `stockout_flag` | integer / binary | Whether the SKU stocked out during the day | `0` or `1` | 1 means a stockout occurred |
| `units_sold` | integer | Number of units sold on that day | `24` | Target variable for forecasting |

## Target variable

### `units_sold`
- Type: integer
- Meaning: daily number of sold units for a specific SKU at a specific venue
- Role in project: prediction target for next-day demand forecasting

## Important modeling notes

- `date` is critical for preserving time order in training and evaluation.
- `promo_flag`, `promo_depth`, and `price` provide commercial context.
- `operating_minutes`, `in_stock_minutes`, and `stockout_flag` provide inventory and availability context.
- `units_sold` can sometimes be zero, which is why standard MAPE is unstable for this dataset.

## Example row

| date | venue_id | sku_id | phl1_id | phl2_id | phl3_id | country_id | price | promo_flag | promo_depth | operating_minutes | in_stock_minutes | stockout_flag | units_sold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025-09-01 | 1012 | 450321 | 10 | 104 | 1047 | 1 | 3.49 | 1 | 0.20 | 900 | 900 | 0 | 24 |