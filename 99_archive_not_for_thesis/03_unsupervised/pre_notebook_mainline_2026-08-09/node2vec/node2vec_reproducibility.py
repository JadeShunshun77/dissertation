"""Reproducibility and provenance helpers shared by Node2Vec experiments."""

from __future__ import annotations

import hashlib
import json
import random
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
PACKAGE_NAMES = (
    "gensim",
    "networkx",
    "node2vec",
    "numpy",
    "pandas",
    "scikit-learn",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def stable_hash(value: object) -> int:
    """Return a process-independent hash for gensim's vocabulary seeding."""
    payload = str(value).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def seed_everything(seed: int) -> None:
    """Seed both RNGs used by node2vec 0.3.2's walk implementation."""
    random.seed(seed)
    np.random.seed(seed)


def package_versions() -> dict[str, str]:
    versions = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def build_compatibility_metadata(
    *,
    graph_file: Path,
    feature_file: Path,
    parameters: dict,
    code_files: list[Path],
) -> dict:
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "input_hashes": {
            "graph_sha256": sha256_file(graph_file),
            "feature_sha256": sha256_file(feature_file),
        },
        "parameters": parameters,
        "package_versions": package_versions(),
        "code_hashes": {
            Path(path).name: sha256_file(path) for path in code_files
        },
        "randomness_control": {
            "python_random_seeded": True,
            "numpy_random_seeded": True,
            "word2vec_seeded": True,
            "word2vec_hashfxn": "sha256-first-64-bit-little-endian",
            "workers": 1,
        },
    }
    return json.loads(json.dumps(metadata, sort_keys=True))


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
