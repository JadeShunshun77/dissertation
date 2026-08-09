"""Rebuild and audit the frozen reconstructed edge table without querying a database."""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_EDGES = PROJECT_ROOT / "00_data_graph" / "data" / "reconstruction_reference" / "raw_edges.csv"
AUGMENTATION_EDGES = (
    PROJECT_ROOT
    / "00_data_graph"
    / "data"
    / "reconstruction_reference"
    / "nonseed_to_nonseed_augmentation_edges.csv"
)
ACTIVE_EDGES = PROJECT_ROOT / "00_data_graph" / "data" / "main" / "raw_edges_reconstructed.csv"
ACTIVE_LABELS = PROJECT_ROOT / "00_data_graph" / "data" / "main" / "final_seed_address500.csv"
EDGE_COLUMNS = ["source", "target", "tx_count", "total_eth_value", "avg_gas_fee"]
EXPECTED_ROWS = {"raw": 12861, "augmentation": 34336, "reconstructed": 47197}
EXPECTED_NODES = 8029
EXPECTED_ACTIVE_SHA256 = "59CECEDA94AF4792C1B3397A9EA5F5E44A3B15C2A59BB14FA27AE01B05BB1DAB"


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest().upper()


def load_edges(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != EDGE_COLUMNS:
        raise ValueError(f"Unexpected columns in {path}: {list(frame.columns)}")
    if frame[EDGE_COLUMNS].isna().any().any():
        raise ValueError(f"Missing edge values in {path}")
    return frame


def pair_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(frame["source"].str.lower(), frame["target"].str.lower()))


def serialize_edges(frame: pd.DataFrame) -> bytes:
    # CRLF is required to reproduce the frozen Windows CSV byte-for-byte.
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, lineterminator="\r\n")
    return buffer.getvalue().encode("utf-8")


def audit_reconstruction() -> tuple[bytes, dict[str, object]]:
    raw = load_edges(RAW_EDGES)
    augmentation = load_edges(AUGMENTATION_EDGES)
    seeds = set(pd.read_csv(ACTIVE_LABELS)["address"].astype(str).str.lower())

    if len(raw) != EXPECTED_ROWS["raw"] or len(augmentation) != EXPECTED_ROWS["augmentation"]:
        raise ValueError("Frozen component edge counts do not match the recorded design")

    raw_pairs = pair_set(raw)
    augmentation_pairs = pair_set(augmentation)
    if len(raw_pairs) != len(raw) or len(augmentation_pairs) != len(augmentation):
        raise ValueError("A component contains duplicate directed source-target pairs")
    if raw_pairs & augmentation_pairs:
        raise ValueError("Raw and nonseed-to-nonseed augmentation components overlap")

    raw_nodes = {node for pair in raw_pairs for node in pair}
    augmentation_nodes = {node for pair in augmentation_pairs for node in pair}
    if not augmentation_nodes <= raw_nodes:
        raise ValueError("The augmentation introduces nodes outside the original fixed node set")
    if augmentation_nodes & seeds:
        raise ValueError("The augmentation contains an edge touching a seed address")
    if any(source == target for source, target in raw_pairs | augmentation_pairs):
        raise ValueError("A component contains a self-loop")

    reconstructed = pd.concat([raw, augmentation], ignore_index=True)
    rebuilt_bytes = serialize_edges(reconstructed)
    active_bytes = ACTIVE_EDGES.read_bytes()
    active_nodes = raw_nodes | augmentation_nodes
    if len(reconstructed) != EXPECTED_ROWS["reconstructed"]:
        raise ValueError("Unexpected reconstructed edge count")
    if len(active_nodes) != EXPECTED_NODES:
        raise ValueError("Unexpected reconstructed node count")
    if rebuilt_bytes != active_bytes:
        raise ValueError("Offline reconstruction is not byte-identical to the active graph")
    if sha256_bytes(active_bytes) != EXPECTED_ACTIVE_SHA256:
        raise ValueError("The active graph hash differs from the frozen reference")

    raw_seed_touching = sum(source in seeds or target in seeds for source, target in raw_pairs)
    report = {
        "raw_rows": len(raw),
        "augmentation_rows": len(augmentation),
        "reconstructed_rows": len(reconstructed),
        "reconstructed_nodes": len(active_nodes),
        "raw_seed_touching_rows": raw_seed_touching,
        "raw_nonseed_to_nonseed_rows": len(raw) - raw_seed_touching,
        "augmentation_seed_touching_rows": 0,
        "augmentation_nonseed_to_nonseed_rows": len(augmentation),
        "component_pair_overlap": 0,
        "self_loops": 0,
        "augmentation_new_nodes": 0,
        "byte_exact_match": True,
        "active_sha256": sha256_bytes(active_bytes),
    }
    return rebuilt_bytes, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the rebuilt graph to a new path; omitted by default for a read-only audit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rebuilt_bytes, report = audit_reconstruction()
    if args.output:
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(rebuilt_bytes)

    for key, value in report.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
