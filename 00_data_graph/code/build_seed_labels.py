"""Audit and reconstruct the frozen seed-label set from source snapshots."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = PROJECT_ROOT / "00_data_graph" / "provenance" / "label_snapshots"
ACTIVE_LABELS = PROJECT_ROOT / "00_data_graph" / "data" / "main" / "final_seed_address500.csv"
DEFAULT_PROVENANCE = (
    PROJECT_ROOT / "00_data_graph" / "provenance" / "seed_label_provenance.csv"
)
SOURCE_FILES = {
    "CEX": "seed_cex_addresses_active.csv",
    "DeFi": "dune_defi_addresses.csv",
    "Malicious_MEV": "seed_mev_addresses.csv",
}
EXPECTED_COUNTS = {"CEX": 51, "DeFi": 150, "Malicious_MEV": 299}
REQUIRED_COLUMNS = ["address", "label", "sub_label"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_snapshot(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != REQUIRED_COLUMNS:
        raise ValueError(f"Unexpected columns in {path}: {list(frame.columns)}")
    frame = frame.copy()
    frame["address"] = frame["address"].astype(str).str.strip().str.lower()
    return frame


def reconstruct_labels() -> tuple[pd.DataFrame, int]:
    parts = []
    for expected_label, filename in SOURCE_FILES.items():
        frame = load_snapshot(SNAPSHOT_DIR / filename)
        if set(frame["label"]) != {expected_label}:
            raise ValueError(f"Unexpected labels in {filename}: {sorted(frame['label'].unique())}")
        parts.append(frame)

    combined = pd.concat(parts, ignore_index=True)
    before_deduplication = len(combined)
    labels_per_address = combined.groupby("address")["label"].nunique()
    if (labels_per_address > 1).any():
        raise ValueError("A source address has conflicting top-level labels")

    def merge_sub_labels(values: pd.Series) -> str:
        unique = sorted({str(value).strip() for value in values if pd.notna(value)})
        return "; ".join(unique)

    combined = (
        combined.groupby("address", sort=False, as_index=False)
        .agg(label=("label", "first"), sub_label=("sub_label", merge_sub_labels))
        .loc[:, REQUIRED_COLUMNS]
    )
    return combined, before_deduplication - len(combined)


def audit_active_labels() -> tuple[pd.DataFrame, dict[str, object]]:
    reconstructed, duplicate_rows_removed = reconstruct_labels()
    active = load_snapshot(ACTIVE_LABELS)

    if active["address"].duplicated().any():
        raise ValueError("The active label file contains duplicate addresses")

    counts = active["label"].value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected active label counts: {counts}")

    rebuilt_by_address = reconstructed.set_index("address").sort_index()
    active_by_address = active.set_index("address").sort_index()
    membership_matches = rebuilt_by_address.index.equals(active_by_address.index)
    labels_match = membership_matches and rebuilt_by_address.equals(active_by_address)
    if not labels_match:
        raise ValueError("Source snapshots do not reproduce the active address-label mapping")

    source_by_label = {label: filename for label, filename in SOURCE_FILES.items()}
    platform_by_label = {
        "CEX": "Dune-derived active selection",
        "DeFi": "Dune",
        "Malicious_MEV": "not_recorded",
    }
    provenance = active.copy()
    provenance["source_snapshot"] = provenance["label"].map(source_by_label)
    provenance["source_platform"] = provenance["label"].map(platform_by_label)
    provenance["source_query_reference"] = "not_recorded"
    provenance["provenance_status"] = provenance["label"].map(
        {
            "CEX": "row matched to frozen active-selection snapshot",
            "DeFi": "row matched to frozen Dune export snapshot",
            "Malicious_MEV": "row matched to frozen export; upstream source not recorded",
        }
    )

    report = {
        "active_file": ACTIVE_LABELS.relative_to(PROJECT_ROOT).as_posix(),
        "active_sha256": sha256(ACTIVE_LABELS),
        "active_rows": len(active),
        "label_counts": counts,
        "source_rows_before_deduplication": len(reconstructed) + duplicate_rows_removed,
        "duplicate_source_rows_removed": duplicate_rows_removed,
        "address_membership_matches": membership_matches,
        "address_label_mapping_matches": labels_match,
        "historical_row_order_reproduced": reconstructed.equals(active),
        "row_order_note": (
            "The frozen active file is authoritative. Its address-label mapping is fully "
            "auditable, but the historical post-merge row ordering step was not retained."
        ),
    }
    return provenance, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provenance-output",
        type=Path,
        help="Write row-level provenance CSV; omitted by default for a read-only audit.",
    )
    parser.add_argument(
        "--canonical-output",
        type=Path,
        help="Write a canonical source-order reconstruction to a new path.",
    )
    return parser.parse_args()


def write_new_file(path: Path, content: bytes) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def main() -> int:
    args = parse_args()
    provenance, report = audit_active_labels()

    if args.provenance_output:
        csv_bytes = provenance.to_csv(index=False, lineterminator="\r\n").encode("utf-8")
        write_new_file(args.provenance_output, csv_bytes)

    if args.canonical_output:
        reconstructed, _ = reconstruct_labels()
        csv_bytes = reconstructed.to_csv(index=False, lineterminator="\r\n").encode("utf-8")
        write_new_file(args.canonical_output, csv_bytes)

    for key, value in report.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
