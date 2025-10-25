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


CREATE OR REPLACE TABLE my_dataset.classification_table AS
SELECT
  id,
  CAST(20 + RAND() * 40 AS INT64) AS age,
  ROUND(20000 + RAND() * 80000, 2) AS income,
  IF(RAND() < 0.3, 1, 0) AS churn
FROM UNNEST(GENERATE_ARRAY(1, 500)) AS id;

CREATE OR REPLACE TABLE my_dataset.forecast_table AS
SELECT
  DATE_ADD('2023-01-01', INTERVAL day DAY) AS ds,
  ROUND(100 + 10 * SIN(day / 10.0) + RAND() * 5, 2) AS y,
  country
FROM UNNEST(GENERATE_ARRAY(0, 364)) AS day,
     UNNEST(['IN', 'US', 'UK']) AS country;

CREATE OR REPLACE TABLE my_dataset.reco_ratings AS
SELECT
  user_id,
  item_id,
  ROUND(1 + RAND() * 4, 1) AS rating
FROM UNNEST(GENERATE_ARRAY(1, 50)) AS user_id,
     UNNEST(GENERATE_ARRAY(1, 30)) AS item_id
WHERE RAND() < 0.3;  -- 30% sparsity

CREATE OR REPLACE TABLE my_dataset.autoenc_table AS
SELECT
  id,
  ROUND(RAND() * 10, 2) AS num1,
  ROUND(RAND() * 100, 2) AS num2,
  ROUND(RAND() * 50, 2) AS num3,
  ['A', 'B', 'C'][OFFSET(CAST(FLOOR(RAND() * 3) AS INT64))] AS cat1
FROM UNNEST(GENERATE_ARRAY(1, 300)) AS id;

SELECT 'classification' AS table, COUNT(*) FROM my_dataset.classification_table
UNION ALL
SELECT 'forecast', COUNT(*) FROM my_dataset.forecast_table
UNION ALL
SELECT 'reco', COUNT(*) FROM my_dataset.reco_ratings
UNION ALL
SELECT 'autoenc', COUNT(*) FROM my_dataset.autoenc_table;

-- create normalized, binned, and one-hot features for classification:
CREATE OR REPLACE TABLE my_dataset.classification_features AS
SELECT
  id,
  age,
  income,
  CASE WHEN age < 30 THEN 'young'
       WHEN age < 60 THEN 'mid'
       ELSE 'senior' END AS age_bucket,
  SAFE_DIVIDE(income, 1000) AS income_k,
  IF(income IS NULL, 0, 1) AS income_present,
  churn AS label
FROM my_dataset.classification_table;

-- One-hot (manual) example (or use ML.FEATURE_CROSS/ML.PREPARE if desired):
SELECT
  id,
  age,
  income_k,
  IF(age_bucket='young', 1, 0) AS is_young,
  IF(age_bucket='mid', 1, 0) AS is_mid,
  IF(age_bucket='senior', 1, 0) AS is_senior,
  label
FROM my_dataset.classification_features;


SELECT * 
FROM my_dataset.regression_table
LIMIT 5;

-- Train Regression (Linear) — quick SQL
CREATE OR REPLACE MODEL my_dataset.regression_model
OPTIONS(model_type = 'linear_reg') AS
SELECT feature1, feature2, label
FROM my_dataset.regression_table;


-- Evaluate:
SELECT *
FROM ML.EVALUATE(MODEL my_dataset.regression_model,
  (SELECT feature1, feature2, label FROM my_dataset.regression_table));

-- Predict:
SELECT id, predicted_label
FROM ML.PREDICT(MODEL my_dataset.regression_model,
  (SELECT id, feature1, feature2 FROM my_dataset.regression_table));


-- Step 2a: create base features
CREATE OR REPLACE TABLE my_dataset.classification_features AS
SELECT
  id,
  age,
  income,
  CASE 
    WHEN age < 30 THEN 'young'
    WHEN age < 60 THEN 'mid'
    ELSE 'senior'
  END AS age_bucket,
  SAFE_DIVIDE(income, 1000) AS income_k,
  IF(income IS NULL, 0, 1) AS income_present,
  churn AS label
FROM my_dataset.classification_table;

-- Step 2b: one-hot encode the categorical "age_bucket"
CREATE OR REPLACE TABLE my_dataset.classification_features_onehot AS
SELECT
  id,
  age,
  income_k,
  IF(age_bucket = 'young', 1, 0) AS is_young,
  IF(age_bucket = 'mid', 1, 0) AS is_mid,
  IF(age_bucket = 'senior', 1, 0) AS is_senior,
  label
FROM my_dataset.classification_features;


-- Train Classification (Logistic)
CREATE OR REPLACE MODEL my_dataset.logistic_model
OPTIONS(
  model_type = 'logistic_reg',
  input_label_cols = ['label']
) AS
SELECT
  age,
  income_k,
  is_young, is_mid, is_senior,
  label
FROM my_dataset.classification_features_onehot;

-- evaluate
SELECT *
FROM ML.EVALUATE(MODEL `gcp-nuvepro.my_dataset.logistic_model`,
  (SELECT age, income_k, is_young, is_mid, is_senior, label
   FROM `gcp-nuvepro.my_dataset.classification_features_onehot`));

--Predict
SELECT id, predicted_label, predicted_label_probs
FROM ML.PREDICT(MODEL `gcp-nuvepro.my_dataset.logistic_model`,
  (SELECT id, age, income_k, is_young, is_mid, is_senior
   FROM `gcp-nuvepro.my_dataset.classification_features_onehot`));

-- Boosted Trees (fast, often better off the shelf)
-- Regression example (boosted tree regressor):
CREATE OR REPLACE MODEL my_dataset.boosted_reg
OPTIONS(
  model_type = 'boosted_tree_regressor',
  input_label_cols = ['label'],
  max_iterations = 50   -- ✅ correct parameter name
) AS
SELECT
  feature1,
  feature2,
  label
FROM my_dataset.regression_table;

--Classification example (classifier):
CREATE OR REPLACE MODEL my_dataset.boosted_clf
OPTIONS(
  model_type = 'boosted_tree_classifier',
  input_label_cols = ['label'],
  max_iterations = 50
) AS
SELECT age, income_k, is_young, is_mid, is_senior, label FROM my_dataset.classification_features_onehot;

-- Explainability: ML.FEATURE_IMPORTANCE(MODEL ...) gives feature importances for tree models:
SELECT * FROM ML.FEATURE_IMPORTANCE(MODEL my_dataset.boosted_clf);

-- Time-series forecasting (ARIMA / ARIMA_PLUS)
CREATE OR REPLACE MODEL my_dataset.ts_model
OPTIONS(
  model_type='ARIMA_PLUS',
  time_series_timestamp_col='ds',
  time_series_data_col='y',
  time_series_id_col='country',   -- optional for multiple series
  horizon=7                      -- forecast 7 steps ahead
) AS
SELECT ds, y, country FROM my_dataset.forecast_table;

-- Forecast:
SELECT * FROM ML.FORECAST(MODEL my_dataset.ts_model, STRUCT(7 AS horizon));

-- Matrix Factorization (recommendation)
CREATE OR REPLACE MODEL my_dataset.reco_mf
OPTIONS(
  model_type='matrix_factorization',
  user_col='user_id',
  item_col='item_id',
  rating_col='rating',
  feedback_type='explicit'     -- explicit ratings example
) AS
SELECT user_id, item_id, rating FROM my_dataset.reco_ratings;

-- Get recommendations (top-k for a user):
SELECT *
FROM ML.RECOMMEND(MODEL my_dataset.reco_mf,
  (SELECT 'user_123' AS user_id), STRUCT(10 AS max_recommendations));

-- Autoencoder (anomaly detection / dimensionality reduction)
CREATE OR REPLACE MODEL my_dataset.autoenc_model
OPTIONS(
  model_type='auto_encoder',
  input_label_cols=[],        -- none; unsupervised
  hidden_units=[64,32,8]     -- example architecture
) AS
SELECT
  SAFE_CAST(num1 AS FLOAT64) AS num1,
  SAFE_CAST(num2 AS FLOAT64) AS num2,
  SAFE_CAST(num3 AS FLOAT64) AS num3
FROM my_dataset.autoenc_table;


