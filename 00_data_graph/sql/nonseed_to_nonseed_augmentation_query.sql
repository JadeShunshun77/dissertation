-- RECOVERED NONSEED-TO-NONSEED AUGMENTATION QUERY LOGIC.
-- This is the read-only fixed-node augmentation recorded in the historical
-- project log. SAFE_CAST is used for the gas-fee multiplication to avoid
-- overflow. The frozen nonseed_to_nonseed_augmentation_edges.csv remains the
-- authoritative export.

WITH non_seed_nodes AS (
  SELECT DISTINCT address
  FROM (
    SELECT LOWER(source) AS address
    FROM `dissertation-500320.ethereum_data.raw_edges`
    UNION ALL
    SELECT LOWER(target) AS address
    FROM `dissertation-500320.ethereum_data.raw_edges`
  )
  WHERE address NOT IN (
    SELECT LOWER(address)
    FROM `dissertation-500320.ethereum_data.final_seed_addresses`
  )
)
SELECT
  LOWER(from_address) AS source,
  LOWER(to_address) AS target,
  COUNT(*) AS tx_count,
  SUM(value) / 1e18 AS total_eth_value,
  AVG(
    SAFE_CAST(receipt_gas_used AS FLOAT64)
    * SAFE_CAST(COALESCE(receipt_effective_gas_price, gas_price) AS FLOAT64)
  ) / 1e18 AS avg_gas_fee
FROM `bigquery-public-data.crypto_ethereum.transactions` AS transactions
JOIN non_seed_nodes AS senders
  ON LOWER(transactions.from_address) = senders.address
JOIN non_seed_nodes AS receivers
  ON LOWER(transactions.to_address) = receivers.address
WHERE block_timestamp BETWEEN TIMESTAMP("2026-03-25") AND TIMESTAMP("2026-06-23")
  AND to_address IS NOT NULL
  AND from_address != to_address
  AND receipt_status = 1
GROUP BY source, target;
