from google.cloud import bigquery
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUERY_DIR = PROJECT_ROOT / "02_feature_engineering" / "nonseed_transaction_query"
ADDRESS_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else QUERY_DIR / "non_seed_addresses.txt"
OUT_FILE = Path(sys.argv[2]) if len(sys.argv) > 2 else QUERY_DIR / "nonseed_tx_behaviour_raw.csv"

with open(ADDRESS_FILE) as f:
    addresses = [line.strip() for line in f if line.strip()]
print('n addresses:', len(addresses))

client = bigquery.Client(project='dissertation-500320')

query = '''
WITH out_pairs AS (
  SELECT from_address, to_address, COUNT(*) AS tx_count, SUM(value)/1e18 AS total_eth_value,
         AVG(SAFE_CAST(receipt_gas_used AS FLOAT64) * SAFE_CAST(COALESCE(receipt_effective_gas_price, gas_price) AS FLOAT64))/1e18 AS avg_gas_fee
  FROM `bigquery-public-data.crypto_ethereum.transactions`
  WHERE from_address IN UNNEST(@addresses)
    AND block_timestamp BETWEEN TIMESTAMP("2026-03-25") AND TIMESTAMP("2026-06-23")
    AND to_address IS NOT NULL AND receipt_status = 1
  GROUP BY from_address, to_address
),
out_agg AS (
  SELECT from_address AS address,
    SUM(tx_count) AS out_tx_count,
    SUM(total_eth_value) AS out_eth_value,
    AVG(avg_gas_fee) AS out_gas_fee_mean,
    SUM(CASE WHEN total_eth_value = 0 THEN 1 ELSE 0 END) AS out_zero_value_edges,
    COUNT(*) AS out_edge_count
  FROM out_pairs
  GROUP BY from_address
),
in_pairs AS (
  SELECT from_address, to_address, COUNT(*) AS tx_count, SUM(value)/1e18 AS total_eth_value,
         AVG(SAFE_CAST(receipt_gas_used AS FLOAT64) * SAFE_CAST(COALESCE(receipt_effective_gas_price, gas_price) AS FLOAT64))/1e18 AS avg_gas_fee
  FROM `bigquery-public-data.crypto_ethereum.transactions`
  WHERE to_address IN UNNEST(@addresses)
    AND block_timestamp BETWEEN TIMESTAMP("2026-03-25") AND TIMESTAMP("2026-06-23")
    AND to_address IS NOT NULL AND receipt_status = 1
  GROUP BY from_address, to_address
),
in_agg AS (
  SELECT to_address AS address,
    SUM(tx_count) AS in_tx_count,
    SUM(total_eth_value) AS in_eth_value,
    AVG(avg_gas_fee) AS in_gas_fee_mean,
    SUM(CASE WHEN total_eth_value = 0 THEN 1 ELSE 0 END) AS in_zero_value_edges,
    COUNT(*) AS in_edge_count
  FROM in_pairs
  GROUP BY to_address
)
SELECT addr AS address,
  out_tx_count, out_eth_value, out_gas_fee_mean, out_zero_value_edges, out_edge_count,
  in_tx_count, in_eth_value, in_gas_fee_mean, in_zero_value_edges, in_edge_count
FROM UNNEST(@addresses) AS addr
LEFT JOIN out_agg ON addr = out_agg.address
LEFT JOIN in_agg ON addr = in_agg.address
'''

job_config = bigquery.QueryJobConfig(
    query_parameters=[bigquery.ArrayQueryParameter('addresses', 'STRING', addresses)],
)
job = client.query(query, job_config=job_config)
df = job.result().to_dataframe()
print('rows:', len(df))
print('bytes billed:', job.total_bytes_billed, '=', job.total_bytes_billed/1e9, 'GB')
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_FILE, index=False)
print(df.head())
print(df.describe())
