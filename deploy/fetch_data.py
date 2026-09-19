"""Download a pinned release and verify every published file before activation."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from dataset import atomic_json


def verify(root, inventory):
    root = Path(root).resolve()
    for relative, expected in inventory["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not relative.startswith(("store/", "index/")):
            raise ValueError("Unsafe release path")
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Size mismatch: {relative}")
        with path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != expected["sha256"]:
                raise ValueError(f"Checksum mismatch: {relative}")


def main():
    from huggingface_hub import snapshot_download
    repo = os.environ["PUBMED_DATA_REPO"]
    revision = os.environ["PUBMED_DATA_REVISION"]
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("PUBMED_DATA_REVISION must be an immutable commit hash")
    target = Path(os.environ.get("PUBMED_DATA_DIR", "/home/user/data"))
    release = target / "releases" / revision
    snapshot_download(repo_id=repo, repo_type="dataset", revision=revision,
                      local_dir=str(release), token=os.environ.get("HF_TOKEN"), max_workers=2)
    inventory = json.loads((release / "release.json").read_text(encoding="utf-8"))
    verify(release, inventory)
    cfg = inventory["dataset"]
    atomic_json(target / "dataset.json", {**cfg, "store": str(release / "store"), "index": str(release / "index")})
    print("Verified dataset snapshot: " + cfg["snapshot"], flush=True)


if __name__ == "__main__":
    main()
