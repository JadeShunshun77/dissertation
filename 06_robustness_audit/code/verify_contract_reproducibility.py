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
# script re-queries eth_getCode for all 500 seeds, this time recording the
# block number and UTC timestamp at query time, and cross-checks against the
# original seed_is_contract.csv. Contract deployment is permanent (bytecode
# at a deployed address does not revert to empty), so if the original query
# was correct, is_contract=True addresses must still show non-empty bytecode
# today; this also serves as an independent re-verification of the original
# 150/150 DeFi-is-contract finding.

RPC_URL = "https://ethereum-rpc.publicnode.com"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "06_robustness_audit" / "outputs" / "contract_confound"
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


def get_code(address):
    return rpc_call("eth_getCode", [address, "latest"])


# =========================
# 1. Record block number + timestamp at query time
# =========================

query_started_utc = datetime.now(timezone.utc).isoformat()
block_hex = rpc_call("eth_blockNumber", [])
block_number = int(block_hex, 16) if block_hex else None
print(f"Query started at {query_started_utc} UTC, chain head block = {block_number}")

# =========================
# 2. Re-query eth_getCode for all 500 seeds
# =========================

original = pd.read_csv(OUT_DIR / "seed_is_contract.csv")
rows = []
for i, addr in enumerate(original["address"]):
    code = get_code(addr)
    ok = code is not None
    bytecode_len = len(code) - 2 if ok else None  # strip "0x"
    is_contract = ok and bytecode_len > 0
    rows.append({"address": addr, "bytecode_length_recheck": bytecode_len, "is_contract_recheck": is_contract, "rpc_ok_recheck": ok})
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(original)} re-queried")

recheck = pd.DataFrame(rows)
merged = original.merge(recheck, on="address", how="left")
merged["match"] = merged["is_contract"] == merged["is_contract_recheck"]

query_finished_utc = datetime.now(timezone.utc).isoformat()

# =========================
# 3. Save results + metadata
# =========================

merged.to_csv(OUT_DIR / "contract_reproducibility_check.csv", index=False)

n_match = int(merged["match"].sum())
n_total = len(merged)
mismatches = merged[~merged["match"]][["address", "is_contract", "is_contract_recheck", "rpc_ok_recheck"]]

metadata = {
    "rpc_url": RPC_URL,
    "query_started_utc": query_started_utc,
    "query_finished_utc": query_finished_utc,
    "block_parameter_used": "latest",
    "chain_head_block_number_at_query_start": block_number,
    "n_addresses_rechecked": n_total,
    "n_matches_vs_original_is_contract_csv": n_match,
    "match_rate": n_match / n_total,
    "n_mismatches": int(len(mismatches)),
}
with open(OUT_DIR / "contract_reproducibility_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n=== Reproducibility check: {n_match}/{n_total} addresses match original is_contract ({100 * n_match / n_total:.1f}%) ===")
if len(mismatches) > 0:
    print(f"Mismatches:\n{mismatches.to_string(index=False)}")
print(f"Chain head block at query time: {block_number} (block parameter used: 'latest')")
print(f"Saved {OUT_DIR / 'contract_reproducibility_check.csv'} and contract_reproducibility_metadata.json")
print("\n=== Contract reproducibility check complete ===")
