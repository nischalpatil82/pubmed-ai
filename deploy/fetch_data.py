#!/usr/bin/env python3
"""
Pull the prebuilt store/ and index/ into the container at startup.

The vectors were computed once, on a laptop, over about 15 hours. They are
ordinary files. Downloading them is a few minutes; recomputing them would be
impossible on a free CPU Space. So the data lives in a Hugging Face dataset
repo and is fetched on boot.

Skips the download entirely if the data is already present, so a warm restart
costs nothing.
"""
from __future__ import annotations

import os
import sys
import time

DATASET = os.environ.get("PUBMED_DATA_REPO", "Nischalpatil/pubmed-ai-data")
TARGET = os.environ.get("PUBMED_DATA_DIR", "/home/user/data")
STORE = os.path.join(TARGET, "store")
INDEX = os.path.join(TARGET, "index")


def already_there() -> bool:
    """A marker file from each half - enough to tell a warm start from a cold one."""
    return (os.path.exists(os.path.join(STORE, "articles.parquet"))
            and os.path.exists(os.path.join(INDEX, "vector_model.json")))


def main() -> None:
    if already_there():
        print(f"data already present in {TARGET} - skipping download", flush=True)
        return

    from huggingface_hub import snapshot_download

    os.makedirs(TARGET, exist_ok=True)
    print(f"fetching {DATASET} -> {TARGET}", flush=True)
    t0 = time.time()
    snapshot_download(
        repo_id=DATASET,
        repo_type="dataset",
        local_dir=TARGET,
        # A private dataset needs a token; a public one does not. HF_TOKEN is
        # injected by the Space when it is set as a secret.
        token=os.environ.get("HF_TOKEN") or None,
        max_workers=4,
    )
    print(f"downloaded in {time.time() - t0:.0f}s", flush=True)

    if not already_there():
        sys.exit(f"ERROR: {DATASET} downloaded but store/ or index/ is missing. "
                 f"Check the dataset repo has store/ and index/ at its root.")

    for label, path in (("store", STORE), ("index", INDEX)):
        n = sum(len(f) for _, _, f in os.walk(path))
        mb = sum(os.path.getsize(os.path.join(r, f))
                 for r, _, fs in os.walk(path) for f in fs) / 1e6
        print(f"  {label:6s} {mb:8.0f} MB  {n} files", flush=True)


if __name__ == "__main__":
    main()
