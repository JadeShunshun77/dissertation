"""Query Ethereum contract status at one pinned block.

The script writes a complete cache only when every ``eth_getCode`` request
succeeds. Existing frozen evidence is protected unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


RPC_URL = "https://ethereum-rpc.publicnode.com"


def rpc_call(method: str, params: list[object], retries: int = 3) -> str | None:
    payload = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
    for attempt in range(retries):
        try:
            response = requests.post(RPC_URL, json=payload, timeout=15)
            response.raise_for_status()
            data = response.json()
            if "result" in data:
                return data["result"]
        except (requests.RequestException, ValueError):
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=project_root
        / "02_feature_engineering"
        / "feature_engineering"
        / "seed_feature_matrix_reconstructed.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "contract_status"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cache_path = args.output_dir / "seed_is_contract.csv"
    log_path = args.output_dir / "contract_query_log.json"
    if (cache_path.exists() or log_path.exists()) and not args.force:
        raise FileExistsError(
            "Frozen contract evidence already exists. Use --force only for an "
            "intentional refresh."
        )

    features = pd.read_csv(args.features, usecols=["address"])
    features["address"] = features["address"].str.strip().str.lower()
    if features["address"].isna().any() or features["address"].duplicated().any():
        raise ValueError("Feature matrix contains missing or duplicate addresses")

    started_at = datetime.now(timezone.utc).isoformat()
    block_hex = rpc_call("eth_blockNumber", [])
    if block_hex is None:
        raise RuntimeError("Unable to resolve a chain-head block")
    block_number = int(block_hex, 16)

    rows: list[dict[str, object]] = []
    for index, address in enumerate(features["address"], start=1):
        code = rpc_call("eth_getCode", [address, block_hex])
        if code is None:
            raise RuntimeError(
                f"RPC failed for {address} at row {index}; no partial cache was written"
            )
        bytecode_length = (len(code) - 2) // 2
        rows.append(
            {
                "address": address,
                "bytecode_length": bytecode_length,
                "is_contract": bytecode_length > 0,
                "rpc_ok": True,
            }
        )
        if index % 50 == 0:
            print(f"Queried {index}/{len(features)} addresses")

    finished_at = datetime.now(timezone.utc).isoformat()
    query_log = {
        "rpc_endpoint": RPC_URL,
        "method": "eth_getCode",
        "query_started_utc": started_at,
        "query_finished_utc": finished_at,
        "block_parameter_used": block_hex,
        "chain_head_block_number": block_number,
        "empty_bytecode_rule": 'result == "0x" => EOA; otherwise => contract',
        "n_addresses_queried": len(rows),
        "n_rpc_success": len(rows),
        "n_rpc_failed_after_retries": 0,
        "retry_policy": "up to 3 attempts, 1.5s * attempt_number backoff",
        "coverage_pct": 100.0,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(cache_path, index=False)
    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(query_log, handle, indent=2)
    print(f"Saved {cache_path}")
    print(f"Saved {log_path}")


if __name__ == "__main__":
    main()
