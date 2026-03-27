USE order_delivery_project;

-- 1. Missing value check
SELECT
  SUM(date IS NULL) AS missing_date,
  SUM(price IS NULL) AS missing_price,
  SUM(units_sold IS NULL) AS missing_units,
  SUM(in_stock_minutes IS NULL) AS missing_stock
FROM item_sales;

-- 2. Range sanity checks
SELECT
  MIN(price) AS min_price,
  MAX(price) AS max_price,
  MIN(units_sold) AS min_units,
  MAX(units_sold) AS max_units
FROM item_sales;

-- 3. Stockout logic check
SELECT COUNT(*) AS inconsistent_stock
FROM item_sales
WHERE stockout_flag = 1 AND in_stock_minutes = operating_minutes;