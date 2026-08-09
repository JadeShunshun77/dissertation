-- DOCUMENTED EQUIVALENT QUERY, NOT THE EXACT HISTORICAL SQL.
-- The original BigQuery SQL and Job ID were not retained. This query records
-- the known method: successful top-level ETH transactions in the 90-day
-- window, retaining the 100 highest-transaction-count counterparties in each
-- direction for every seed address. The frozen raw_edges.csv is authoritative.
-- Expected seed table: `dissertation-500320.ethereum_data.final_seed_addresses`

WITH seeds AS (
  SELECT LOWER(address) AS address
  FROM `dissertation-500320.ethereum_data.final_seed_addresses`
),
pair_aggregates AS (
  SELECT
    LOWER(from_address) AS source,
    LOWER(to_address) AS target,
    COUNT(*) AS tx_count,
    SUM(value) / 1e18 AS total_eth_value,
    AVG(
      SAFE_CAST(receipt_gas_used AS FLOAT64)
      * SAFE_CAST(COALESCE(receipt_effective_gas_price, gas_price) AS FLOAT64)
    ) / 1e18 AS avg_gas_fee
  FROM `bigquery-public-data.crypto_ethereum.transactions`
  WHERE block_timestamp BETWEEN TIMESTAMP("2026-03-25") AND TIMESTAMP("2026-06-23")
    AND to_address IS NOT NULL
    AND from_address != to_address
    AND receipt_status = 1
    AND (
      LOWER(from_address) IN (SELECT address FROM seeds)
      OR LOWER(to_address) IN (SELECT address FROM seeds)
    )
  GROUP BY source, target
),
ranked_outgoing AS (
  SELECT
    pair_aggregates.*,
    ROW_NUMBER() OVER (
      PARTITION BY source ORDER BY tx_count DESC, target
    ) AS counterparty_rank
  FROM pair_aggregates
  WHERE source IN (SELECT address FROM seeds)
),
ranked_incoming AS (
  SELECT
    pair_aggregates.*,
    ROW_NUMBER() OVER (
      PARTITION BY target ORDER BY tx_count DESC, source
    ) AS counterparty_rank
  FROM pair_aggregates
  WHERE target IN (SELECT address FROM seeds)
),
selected_pairs AS (
  SELECT source, target FROM ranked_outgoing WHERE counterparty_rank <= 100
  UNION DISTINCT
  SELECT source, target FROM ranked_incoming WHERE counterparty_rank <= 100
)
SELECT
  aggregates.source,
  aggregates.target,
  aggregates.tx_count,
  aggregates.total_eth_value,
  aggregates.avg_gas_fee
FROM selected_pairs
JOIN pair_aggregates AS aggregates USING (source, target);
