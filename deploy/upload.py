"""Publish an explicitly selected release or application to explicitly named repos."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

# The Rust Xet client can deadlock while finalizing multi-file uploads on
# Windows. Traditional LFS streams files with bounded memory and leaves
# uploaded blobs reusable when a later commit must be retried.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from dataset import paths, read_json


def configure_space(api, repo_id, data_repo=None, data_revision=None):
    """Apply non-secret runtime settings and optional secrets to a Space."""
    variables = {
        "PUBMED_DATA_REPO": data_repo or os.environ.get("PUBMED_DATA_REPO"),
        "PUBMED_DATA_REVISION": data_revision or os.environ.get("PUBMED_DATA_REVISION"),
        "PUBMED_LLM": os.environ.get("PUBMED_LLM"),
        "PUBMED_ALLOW_CLOUD": os.environ.get("PUBMED_ALLOW_CLOUD"),
        "PUBMED_API_BASE": os.environ.get("PUBMED_API_BASE"),
        "PUBMED_CLOUD_MODEL": os.environ.get("PUBMED_CLOUD_MODEL"),
        # A fresh Space has no model cache. Permit the first startup to fetch
        # the exact embedding-model revision recorded by the vector index.
        "PUBMED_ALLOW_MODEL_DOWNLOAD": os.environ.get(
            "PUBMED_ALLOW_MODEL_DOWNLOAD", "1"
        ),
    }
    for key, value in variables.items():
        if value:
            api.add_space_variable(repo_id, key, value)
    if os.environ.get("PUBMED_API_KEY"):
        api.add_space_secret(repo_id, "PUBMED_API_KEY", os.environ["PUBMED_API_KEY"])
    if os.environ.get("SPACE_HF_TOKEN"):
        api.add_space_secret(repo_id, "HF_TOKEN", os.environ["SPACE_HF_TOKEN"])
    return [key for key, value in variables.items() if value]


def publish_release(api, repo_id, inventory, store, index, snapshot):
    """Upload a large release in small commits and resume from matching files."""
    from huggingface_hub import CommitOperationAdd, RepoFile

    existing = {
        item.path: item.size
        for item in api.list_repo_tree(repo_id, repo_type="dataset", recursive=True,
                                       expand=True)
        if isinstance(item, RepoFile)
    }
    pending = []
    for relative, metadata in inventory["files"].items():
        if existing.get(relative) == metadata["bytes"]:
            continue
        label, rest = relative.split("/", 1)
        pending.append((relative, metadata["bytes"],
                        (store if label == "store" else index) / rest))

    uploaded = len(inventory["files"]) - len(pending)
    total = len(inventory["files"])
    while pending:
        batch = []
        # Use one release commit for the remaining files. Hugging Face limits
        # free repositories to 128 commits per hour; LFS blob uploads remain
        # resumable even if the final commit must be retried.
        while pending:
            relative, size, source = pending[0]
            pending.pop(0)
            batch.append(CommitOperationAdd(path_in_repo=relative,
                                            path_or_fileobj=str(source)))
        api.create_commit(repo_id, repo_type="dataset", operations=batch,
                          num_threads=4,
                          commit_message=(f"Upload {snapshot}: files "
                                          f"{uploaded + 1}-{uploaded + len(batch)} of {total}"))
        uploaded += len(batch)
        print(f"Dataset upload: {uploaded}/{total} files committed", flush=True)

    # Publish the inventory only after every referenced object is committed.
    release_path = Path(os.environ["PUBMED_DATASET"]).resolve().parent / "release.json"
    commit = api.create_commit(
        repo_id, repo_type="dataset",
        operations=[CommitOperationAdd(path_in_repo="release.json",
                                       path_or_fileobj=str(release_path))],
        commit_message="Publish verified snapshot " + snapshot,
    )
    return commit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--data-repo")
    ap.add_argument("--data-revision")
    ap.add_argument("--space-repo")
    ap.add_argument("--public", action="store_true")
    a = ap.parse_args()
    if not a.data_repo and not a.space_repo:
        ap.error("Name the destination with --data-repo or --space-repo")
    if a.dataset and not a.data_repo:
        ap.error("Data publication requires --data-repo")
    if a.data_repo and not a.dataset and not a.space_repo:
        ap.error("Data publication requires --dataset")
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    published_revision = None
    if a.data_repo and a.dataset:
        os.environ["PUBMED_DATASET"] = str(Path(a.dataset).resolve())
        # Keep the heavy build-time dependency out of Space-only deployments.
        from manage import release_manifest
        store, index, cfg = paths()
        release_path = Path(a.dataset).resolve().parent / "release.json"
        inventory = release_manifest(release_path)
        api.create_repo(a.data_repo, repo_type="dataset", private=not a.public, exist_ok=True)
        commit = publish_release(api, a.data_repo, inventory, store, index, cfg["snapshot"])
        published_revision = commit.oid
        print(f"Dataset commit: {commit.oid}; configure PUBMED_DATA_REVISION to this exact commit.")
    if a.space_repo:
        from huggingface_hub.errors import RepositoryNotFoundError

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            shutil.copytree(ROOT / "pipeline", target / "pipeline", ignore=shutil.ignore_patterns("__pycache__", "_unused_*"))
            shutil.copy2(ROOT / "requirements.txt", target / "requirements.txt")
            shutil.copy2(ROOT / "deploy" / "Dockerfile", target / "Dockerfile")
            (target / "deploy").mkdir()
            shutil.copy2(ROOT / "deploy" / "fetch_data.py", target / "deploy" / "fetch_data.py")
            shutil.copy2(ROOT / "deploy" / "README.md", target / "README.md")
            # Existing free Docker Spaces can still be updated even when the
            # account cannot create a new Docker Space. Avoid calling the
            # create endpoint for an existing deployment because it now
            # returns HTTP 402 for non-Pro accounts.
            try:
                api.repo_info(a.space_repo, repo_type="space")
            except RepositoryNotFoundError:
                api.create_repo(a.space_repo, repo_type="space", space_sdk="docker",
                                private=not a.public)
            api.upload_folder(repo_id=a.space_repo, repo_type="space", folder_path=str(target),
                              commit_message="Deploy dataset-aware PubMed application")
            configured = configure_space(api, a.space_repo, a.data_repo,
                                         published_revision or a.data_revision)
            print("Space variables configured: " + (", ".join(configured) or "none"))
            print("Space secrets configured: " +
                  ("PUBMED_API_KEY" if os.environ.get("PUBMED_API_KEY") else "none"))


if __name__ == "__main__":
    main()
