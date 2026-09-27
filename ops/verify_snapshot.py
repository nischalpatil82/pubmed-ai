"""Verify a copied PubMed release against its published file inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def verify(dataset_path: Path, inventory_path: Path) -> tuple[int, int]:
    dataset_path = dataset_path.resolve()
    inventory_path = inventory_path.resolve()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    published = inventory["dataset"]
    if dataset.get("status") != "ready" or dataset.get("pilot"):
        raise ValueError("Dataset is not a complete, ready snapshot")
    if dataset.get("snapshot") != published.get("snapshot"):
        raise ValueError("Dataset snapshot does not match the release inventory")
    roots = {}
    for source, key in (("store", "store"), ("index", "index")):
        location = Path(dataset[key])
        roots[source] = (location if location.is_absolute() else dataset_path.parent / location).resolve()
        if not roots[source].is_dir():
            raise FileNotFoundError(f"Missing {source} directory: {roots[source]}")

    files = inventory["files"]
    if not files:
        raise ValueError("Release inventory has no files")
    total_bytes = 0
    for number, (relative, expected) in enumerate(files.items(), 1):
        source, separator, filename = relative.partition("/")
        if not separator or source not in roots or not filename:
            raise ValueError(f"Invalid inventory path: {relative}")
        path = (roots[source] / filename).resolve()
        if not path.is_relative_to(roots[source]) or not path.is_file():
            raise FileNotFoundError(f"Missing or unsafe release file: {relative}")
        size = path.stat().st_size
        if size != expected["bytes"]:
            raise ValueError(f"Size mismatch: {relative}")
        with path.open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != expected["sha256"]:
                raise ValueError(f"Checksum mismatch: {relative}")
        total_bytes += size
        if number % 100 == 0:
            print(f"Verified {number}/{len(files)} files", flush=True)
    return len(files), total_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "covid-files" / "dataset.json")
    parser.add_argument("--release", type=Path, default=ROOT / "covid-files" / "release.json")
    args = parser.parse_args()
    count, size = verify(args.dataset, args.release)
    print(f"Verified {count} files ({size / 1e9:.2f} GB) against the release inventory")


if __name__ == "__main__":
    main()
