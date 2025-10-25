CREATE OR REPLACE TABLE my_dataset.regression_table AS
WITH base AS (
  SELECT
    id,
    ROUND(RAND() * 10, 2) AS feature1,           -- e.g., temperature
    ROUND(50 + RAND() * 150, 2) AS feature2      -- e.g., advertising spend
  FROM UNNEST(GENERATE_ARRAY(1, 500)) AS id
)
SELECT
  id,
  feature1,
  feature2,
  ROUND(5 + 2 * feature1 + 0.3 * feature2 + RAND() * 5, 2) AS label  -- depends on features + noise
FROM base;


-- Train Regression (Linear) — quick SQL
CREATE OR REPLACE MODEL my_dataset.regression_model
OPTIONS(model_type = 'linear_reg') AS
SELECT feature1, feature2, label
FROM my_dataset.regression_table;

SELECT * 
FROM my_dataset.regression_table
LIMIT 5;


-- Evaluate:
SELECT *
FROM ML.EVALUATE(MODEL my_dataset.regression_model,
  (SELECT feature1, feature2, label FROM my_dataset.regression_table));



-- Predict:
SELECT
  t.id,
  t.label AS actual_label,
  p.predicted_label,
  p.predicted_label - t.label AS error,
  ABS(p.predicted_label - t.label) AS abs_error
FROM
  my_dataset.regression_table AS t
JOIN
  ML.PREDICT(MODEL my_dataset.regression_model,
    (SELECT id, feature1, feature2 FROM my_dataset.regression_table)
  ) AS p
USING (id)
