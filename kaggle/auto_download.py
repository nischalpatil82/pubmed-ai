"""Wait for a saved Kaggle run, then restore its embedding bundles locally.

The Kaggle notebook writes a small number of stored ZIP bundles so this script
can download, verify, extract, and delete one bundle at a time.  This avoids
needing space for both the complete download and the extracted result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time
import zipfile


FORMAT = "pubmed-embedding-bundle-v1"


def sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def kaggle_executable() -> list[str]:
    executable = shutil.which("kaggle")
    if executable:
        return [executable]
    try:
        result = subprocess.run(
            [sys.executable, "-m", "kaggle", "--version"],
            text=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        result = None
    if result and result.returncode == 0:
        return [sys.executable, "-m", "kaggle"]
    raise RuntimeError(
        "Kaggle CLI is missing. Install it with: python -m pip install --user kaggle"
    )


def run_cli(base: list[str], arguments: list[str], check: bool = True) -> str:
    child_env = os.environ.copy()
    # Kaggle's progress output contains Unicode block characters. Force UTF-8 in
    # the child CLI so Windows code pages cannot abort an otherwise valid download.
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"
    result = subprocess.run(
        [*base, *arguments],
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=child_env,
    )
    output = "\n".join(item for item in (result.stdout.strip(), result.stderr.strip()) if item)
    if check and result.returncode != 0:
        raise RuntimeError(output or f"Kaggle command failed with exit code {result.returncode}")
    return output


def wait_for_saved_output(base: list[str], kernel: str, poll_seconds: int) -> None:
    print(f"Watching Kaggle notebook {kernel}", flush=True)
    observed_new_run = False
    while True:
        status = run_cli(base, ["kernels", "status", kernel]).casefold()
        print(status, flush=True)
        if any(word in status for word in ("error", "failed", "cancelled", "canceled")):
            if observed_new_run:
                raise RuntimeError("The Kaggle run failed; output download was not started")
            print("Waiting for the new web-started T4 x2 run", flush=True)
            time.sleep(poll_seconds)
            continue
        if "running" in status or "queued" in status:
            observed_new_run = True
        if "complete" in status:
            files = run_cli(
                base,
                ["kernels", "files", kernel, "--page-size", "200"],
            )
            if "DOWNLOAD_READY.json" in files and "OUTPUT_MANIFEST.json" in files:
                print("Verified saved output marker on Kaggle", flush=True)
                return
            print(
                "Latest completed version has no embedding-ready marker; waiting for the new Save & Run All version",
                flush=True,
            )
        time.sleep(poll_seconds)


def download_matching(base: list[str], kernel: str, root: Path, pattern: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    output = run_cli(
        base,
        [
            "kernels",
            "output",
            kernel,
            "-p",
            str(root),
            "-o",
            "--page-size",
            "200",
            "--file-pattern",
            pattern,
        ],
    )
    if output:
        print(output, flush=True)


def find_unique(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob(name) if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one downloaded {name}; found {matches}")
    return matches[0]


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            relative = PurePosixPath(item.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe archive member: {item.filename}")
            target = (destination / Path(*relative.parts)).resolve()
            if not target.is_relative_to(destination):
                raise RuntimeError(f"Unsafe archive target: {item.filename}")
        source.extractall(destination)


def validate_restored_index(index: Path, manifest: dict) -> dict:
    part_manifests = sorted((index / "embedding-checkpoints").glob("w*-*/part-*.json"))
    if len(part_manifests) != manifest["parts"]:
        raise RuntimeError(
            f"Expected {manifest['parts']} part manifests; found {len(part_manifests)}"
        )
    articles = passages = 0
    for path in part_manifests:
        item = json.loads(path.read_text(encoding="utf-8"))
        for key in ("passage_file", "vector_file"):
            target = (index / item[key]).resolve()
            if not target.is_relative_to(index.resolve()) or not target.is_file():
                raise RuntimeError(f"Missing restored file: {target}")
            if target.stat().st_size != item[key + "_bytes"]:
                raise RuntimeError(f"Size mismatch: {target}")
            if sha256_file(target) != item[key + "_sha256"]:
                raise RuntimeError(f"Checksum mismatch: {target}")
        articles += item["articles"]
        passages += item["passages"]
    if articles != manifest["articles"] or passages != manifest["passages"]:
        raise RuntimeError(
            f"Restored totals differ: {articles:,} articles and {passages:,} passages"
        )
    return {"articles": articles, "passages": passages, "parts": len(part_manifests)}


def restore(base: list[str], kernel: str, download_root: Path, dataset_root: Path) -> Path:
    download_matching(
        base,
        kernel,
        download_root,
        r"(^|/)(OUTPUT_MANIFEST|DOWNLOAD_READY|embedding-plan|source-dataset)\.json$",
    )
    output_manifest_path = find_unique(download_root, "OUTPUT_MANIFEST.json")
    ready_path = find_unique(download_root, "DOWNLOAD_READY.json")
    plan_path = find_unique(download_root, "embedding-plan.json")
    source_dataset_path = find_unique(download_root, "source-dataset.json")
    manifest = json.loads(output_manifest_path.read_text(encoding="utf-8"))
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    source_dataset = json.loads(source_dataset_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT or ready.get("manifest_sha256") != sha256_file(output_manifest_path):
        raise RuntimeError("Saved output marker or manifest checksum is invalid")
    if (plan.get("snapshot") != manifest["snapshot"]
            or plan.get("descriptor_hash") != manifest["descriptor_hash"]
            or source_dataset.get("snapshot") != manifest["snapshot"]
            or source_dataset.get("counts", {}).get("articles") != manifest["articles"]):
        raise RuntimeError("Embedding plan or source dataset metadata does not match the saved output")

    required = (
        manifest["vector_bytes"]
        + manifest["passage_bytes"]
        + max(item["bytes"] for item in manifest["archives"])
        + 512 * 1024**2
    )
    while shutil.disk_usage(dataset_root.parent).free < required:
        free = shutil.disk_usage(dataset_root.parent).free
        print(
            f"Saved output is ready, but the safe restore needs {required / 1e9:.2f} GB "
            f"and only {free / 1e9:.2f} GB is free. Waiting for disk space.",
            flush=True,
        )
        time.sleep(300)

    snapshot = manifest["snapshot"]
    final_index = dataset_root / "indexes" / snapshot
    staging = dataset_root / "indexes" / f".{snapshot}.restoring"
    state_path = staging / "restore-state.json"
    staging.mkdir(parents=True, exist_ok=True)
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {"format": FORMAT, "snapshot": snapshot, "archives": {}}
    )
    if state.get("snapshot") != snapshot:
        raise RuntimeError(f"Restore staging belongs to another snapshot: {staging}")
    local_plan = staging / "embedding-checkpoints" / "w1-0" / "plan.json"
    local_plan.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(plan_path, local_plan)
    shutil.copy2(source_dataset_path, staging / "source-dataset.json")

    for number, item in enumerate(manifest["archives"], start=1):
        name = item["name"]
        expected_hash = item["sha256"]
        if state["archives"].get(name) == expected_hash:
            print(f"Archive {number}/{len(manifest['archives'])} already restored: {name}", flush=True)
            continue
        pattern = r"(^|/)" + re.escape(name) + r"$"
        download_matching(base, kernel, download_root, pattern)
        archive = find_unique(download_root, Path(name).name)
        if archive.stat().st_size != item["bytes"] or sha256_file(archive) != expected_hash:
            raise RuntimeError(f"Downloaded archive failed verification: {archive}")
        safe_extract(archive, staging)
        state["archives"][name] = expected_hash
        atomic_json(state_path, state)
        archive.unlink()
        print(f"Restored archive {number}/{len(manifest['archives'])}: {name}", flush=True)

    result = validate_restored_index(staging, manifest)
    state_path.unlink(missing_ok=True)
    atomic_json(staging / "embedding-parts.json", {**manifest, "restored": result})
    if final_index.exists():
        raise RuntimeError(f"Final index already exists; refusing to overwrite: {final_index}")
    os.replace(staging, final_index)
    atomic_json(
        dataset_root / "DOWNLOAD_COMPLETE.json",
        {"format": FORMAT, "snapshot": snapshot, **result, "index": str(final_index)},
    )
    print(f"Download and verification complete: {final_index}", flush=True)
    return final_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="nischalg/notebook1b4a41db61")
    parser.add_argument("--download-root", default="kaggle/downloads/notebook1b4a41db61")
    parser.add_argument("--dataset-root", default="covid-files")
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()
    if args.poll_seconds < 10:
        raise ValueError("poll-seconds must be at least 10")
    base = kaggle_executable()
    wait_for_saved_output(base, args.kernel, args.poll_seconds)
    restore(base, args.kernel, Path(args.download_root).resolve(), Path(args.dataset_root).resolve())


if __name__ == "__main__":
    main()
