import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# =========================
# 0. Setup
# =========================
# contract_confound_check.py's eth_getCode queries (Sec 10.2) used "latest"
# without recording the block number or query timestamp -- a reproducibility
# gap flagged in review (the is_contract=True/False determination cannot be
# exactly re-derived without knowing which chain state was queried). This
# script re-queries eth_getCode for all 500 seeds at one recorded block,
# records the UTC timestamp, and cross-checks against the
# original seed_is_contract.csv. Contract deployment is permanent (bytecode
# at a deployed address does not revert to empty), so if the original query
# was correct, is_contract=True addresses must still show non-empty bytecode
# today; this also serves as an independent re-verification of the original
# 150/150 DeFi-is-contract finding.

RPC_URL = "https://ethereum-rpc.publicnode.com"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATUS_DIR = PROJECT_ROOT / "06_robustness_audit" / "02_contract_confound" / "contract_status"
parser = argparse.ArgumentParser()
parser.add_argument("--input-cache", type=Path, default=STATUS_DIR / "seed_is_contract.csv")
parser.add_argument(
    "--output-dir",
    type=Path,
    default=PROJECT_ROOT
    / "06_robustness_audit"
    / "02_contract_confound"
    / "rerun_validation"
    / "contract_reproducibility",
)
parser.add_argument("--force", action="store_true")
args = parser.parse_args()
OUT_DIR = args.output_dir
result_path = OUT_DIR / "contract_reproducibility_check.csv"
metadata_path = OUT_DIR / "contract_reproducibility_metadata.json"
if (result_path.exists() or metadata_path.exists()) and not args.force:
    raise FileExistsError("Recheck outputs already exist; use --force to replace them")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rpc_call(method, params, retries=3):
    payload = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
    for attempt in range(retries):
        try:
            r = requests.post(RPC_URL, json=payload, timeout=15)
            data = r.json()
            if "result" in data:
                return data["result"]
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def get_code(address, block_tag):
    return rpc_call("eth_getCode", [address, block_tag])


# =========================
# 1. Record block number + timestamp at query time
# =========================

query_started_utc = datetime.now(timezone.utc).isoformat()
block_hex = rpc_call("eth_blockNumber", [])
if block_hex is None:
    raise RuntimeError("Unable to resolve a chain-head block; no address queries were run")
block_number = int(block_hex, 16)
print(f"Query started at {query_started_utc} UTC, chain head block = {block_number}")

# =========================
# 2. Re-query eth_getCode for all 500 seeds
# =========================

original = pd.read_csv(args.input_cache)
rows = []
for i, addr in enumerate(original["address"]):
    code = get_code(addr, block_hex)
    ok = code is not None
    bytecode_len = (len(code) - 2) // 2 if ok else None
    is_contract = bytecode_len > 0 if ok else None
    rows.append({"address": addr, "bytecode_length_recheck": bytecode_len, "is_contract_recheck": is_contract, "rpc_ok_recheck": ok})
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(original)} re-queried")

recheck = pd.DataFrame(rows)
failed = int((~recheck["rpc_ok_recheck"]).sum())
if failed:
    raise RuntimeError(
        f"Reproducibility query failed for {failed} addresses; incomplete results "
        "were not written."
    )
merged = original.merge(recheck, on="address", how="left")
merged["match"] = merged["is_contract"] == merged["is_contract_recheck"]

query_finished_utc = datetime.now(timezone.utc).isoformat()

# =========================
# 3. Save results + metadata
# =========================

merged.to_csv(result_path, index=False)

n_match = int(merged["match"].sum())
n_total = len(merged)
mismatches = merged[~merged["match"]][["address", "is_contract", "is_contract_recheck", "rpc_ok_recheck"]]

metadata = {
    "rpc_url": RPC_URL,
    "query_started_utc": query_started_utc,
    "query_finished_utc": query_finished_utc,
    "block_parameter_used": block_hex,
    "chain_head_block_number_at_query_start": block_number,
    "n_addresses_rechecked": n_total,
    "n_matches_vs_original_is_contract_csv": n_match,
    "match_rate": n_match / n_total,
    "n_mismatches": int(len(mismatches)),
}
with metadata_path.open("w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"\n=== Reproducibility check: {n_match}/{n_total} addresses match original is_contract ({100 * n_match / n_total:.1f}%) ===")
if len(mismatches) > 0:
    print(f"Mismatches:\n{mismatches.to_string(index=False)}")
print(f"Pinned block used for every eth_getCode call: {block_number} ({block_hex})")
print(f"Saved {result_path} and {metadata_path}")
print("\n=== Contract reproducibility check complete ===")
