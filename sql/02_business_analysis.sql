USE order_delivery_project;

-- 1. Total sales by SKU
SELECT
  sku_id,
  SUM(units_sold) AS total_units
FROM item_sales
GROUP BY sku_id
ORDER BY total_units DESC
LIMIT 10;

-- 2. Promotion impact
SELECT
  promo_flag,
  AVG(units_sold) AS avg_units
FROM item_sales
GROUP BY promo_flag;

-- 3. Revenue by category (PHL1)
SELECT
  phl1_id,
  SUM(units_sold * price) AS revenue
FROM item_sales
GROUP BY phl1_id
ORDER BY revenue DESC;

-- 4. Stockout impact
SELECT
  stockout_flag,
  AVG(units_sold) AS avg_units
FROM item_sales
GROUP BY stockout_flag;