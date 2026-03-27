USE order_delivery_project;

DROP TABLE IF EXISTS item_sales;

CREATE TABLE item_sales (
  date DATE,
  venue_id VARCHAR(20),
  sku_id VARCHAR(20),
  phl1_id VARCHAR(20),
  phl2_id VARCHAR(20),
  phl3_id VARCHAR(20),
  operating_minutes INT,
  country_id VARCHAR(10),
  price DOUBLE,
  promo_flag INT,
  promo_depth DOUBLE,
  in_stock_minutes INT,
  stockout_flag INT,
  units_sold INT
);