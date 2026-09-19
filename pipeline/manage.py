"""One dataset-aware entry point for building, serving and release preparation."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
from build_store import SCHEMAS
from dataset import ROOT, DEFAULT_MANIFEST, paths, read_json, atomic_json, space_guard


def status():
    store, index, cfg = paths()
    out = {"dataset": cfg, "store": str(store), "index": str(index),
           "free_gb": round(shutil.disk_usage(store).free / 1e9, 2),
           "source": read_json(store / "manifest.json")}
    for name in ("lexical", "vector_model"):
        out[name] = read_json(index / f"{name}.json") if (index / f"{name}.json").exists() else None
    return out


def release_manifest(output):
    """Hash the selected completed snapshot in place, without duplicating gigabytes."""
    store, index, cfg = paths()
    required_tables = set(SCHEMAS) | {"vocabulary"}
    missing_tables = sorted(name for name in required_tables if not (store / f"{name}.parquet").is_file())
    if missing_tables:
        raise RuntimeError("Release is missing required tables: " + ", ".join(missing_tables))
    source = read_json(store / "manifest.json")
    if source.get("snapshot") != cfg["snapshot"]:
        raise RuntimeError("Store manifest does not match the selected dataset snapshot")
    lexical = read_json(index / "lexical.json")
    if not lexical.get("complete") or lexical.get("snapshot") != cfg["snapshot"]:
        raise RuntimeError("Finish the matching lexical index before preparing a release")
    missing_shards = [name for name in lexical.get("shards", [])
                      if not (index / name).is_dir() or not (index / name / "ready.json").is_file()]
    if missing_shards:
        raise RuntimeError("Release is missing lexical index shards: " + ", ".join(missing_shards))
    vector = read_json(index / "vector_model.json") if (index / "vector_model.json").exists() else None
    if vector and vector.get("complete") and vector.get("snapshot") == cfg["snapshot"]:
        table = index / (vector["table"] + ".lance")
        if not table.is_dir():
            raise RuntimeError("Release is missing the completed vector table")
        if vector.get("passage_glob") and not any(index.glob(vector["passage_glob"])):
            raise RuntimeError("Release is missing completed vector passage parts")
    files = {}
    # Publish only active lexical shards and the committed complete vector table.
    selected = list(store.glob("*.parquet")) + [store / "manifest.json", index / "lexical.json"]
    for shard in lexical["shards"]:
        selected.extend(p for p in (index / shard).rglob("*") if p.is_file())
    if vector and vector.get("complete") and vector.get("snapshot") == cfg["snapshot"]:
        selected.append(index / "vector_model.json")
        selected.extend(p for p in (index / (vector["table"] + ".lance")).rglob("*") if p.is_file())
        if vector.get("passage_glob"):
            selected.extend(p for p in index.glob(vector["passage_glob"]) if p.is_file())
    for path in selected:
        relative = "store/" + path.relative_to(store).as_posix() if path.is_relative_to(store) else "index/" + path.relative_to(index).as_posix()
        with path.open("rb") as source:
            files[relative] = {"sha256": hashlib.file_digest(source, "sha256").hexdigest(), "bytes": path.stat().st_size}
    result = {"dataset": cfg, "files": files, "retrieval": "hybrid" if vector and vector.get("complete") else "lexical",
              "rollback": "Keep the previous dataset.json and immutable stores/indexes; stop the server before selecting another snapshot."}
    atomic_json(output, result)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["serve", "status", "bm25", "vectors", "release"])
    ap.add_argument("--dataset", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--output", default="covid-files/release.json")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    os.environ["PUBMED_DATASET"] = str(Path(a.dataset).resolve())
    if a.command == "serve":
        from retrieval import BM25Search
        BM25Search()  # Fail clearly before opening the port if the snapshot is unready.
        import uvicorn
        from app import app
        uvicorn.run(app, host="127.0.0.1", port=a.port)
    elif a.command == "status":
        import json
        print(json.dumps(status(), indent=2))
    elif a.command == "release":
        release_manifest(a.output)
        print(f"Release inventory written to {a.output}")
    else:
        from retrieval import build_bm25, build_vectors
        build_bm25(a.limit) if a.command == "bm25" else build_vectors(a.model, a.limit)
