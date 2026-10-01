"""Optional citation summaries derived from a committed dataset snapshot."""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import uuid

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from dataset import BuildLock, atomic_json, paths, read_json, space_guard

FORMAT = "citation-counts-v1"
MARKER = "citation-counts.json"


def _signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _footer_fingerprint(path):
    # Footer metadata is small and unchanged by copying to another installation.
    # Snapshot identity covers source content; this also checks physical layout.
    metadata = pq.read_metadata(path)
    payload = json.dumps(metadata.to_dict(), sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest(), metadata.num_rows


@functools.lru_cache(maxsize=8)
def _validated_path(store, index, snapshot, source_signature, marker_signature):
    source = Path(store) / "citations.parquet"
    directory = Path(index) / "analytics"
    try:
        manifest = read_json(directory / MARKER)
        if (not isinstance(manifest, dict) or manifest.get("format") != FORMAT
                or manifest.get("snapshot") != snapshot):
            return None
        filename = manifest.get("file", "")
        if not filename or Path(filename).name != filename:
            return None
        artifact = (directory / filename).resolve()
        if artifact.parent != directory.resolve() or not artifact.is_file():
            return None
        fingerprint, source_rows = _footer_fingerprint(source)
        if (fingerprint != manifest.get("source_footer_sha256")
                or source_rows != manifest.get("source_rows")):
            return None
        metadata = pq.read_metadata(artifact)
        schema = pl.read_parquet_schema(artifact)
        if (metadata.num_rows != manifest.get("rows")
                or artifact.stat().st_size != manifest.get("bytes")
                or schema != {"cited_pmid": pl.String, "times_cited": pl.UInt64}):
            return None
        with artifact.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != manifest.get("sha256"):
                return None
        return artifact, _signature(artifact)
    except (OSError, ValueError, TypeError, KeyError, pl.exceptions.PolarsError, pa.ArrowInvalid):
        return None


def citation_counts_path(store, index, snapshot):
    """Ignore missing/stale summaries; all endpoints retain their raw-data path."""
    source = Path(store) / "citations.parquet"
    marker = Path(index) / "analytics" / MARKER
    if not marker.is_file():
        return None
    arguments = (str(Path(store).resolve()), str(Path(index).resolve()), snapshot,
                 _signature(source), _signature(marker))
    found = _validated_path(*arguments)
    if found is None:
        return None
    artifact, signature = found
    try:
        if _signature(artifact) != signature:
            _validated_path.cache_clear()
            found = _validated_path(*arguments)
        return found[0] if found else None
    except OSError:
        return None


def build_citation_counts(reserve_gb=2):
    """Build once, then publish an atomic pointer; never modify source tables."""
    store, index, cfg = paths()
    directory = index / "analytics"
    directory.mkdir(parents=True, exist_ok=True)
    with BuildLock(directory / "build.lock"):
        existing = citation_counts_path(store, index, cfg["snapshot"])
        if existing is not None:
            return {**read_json(directory / MARKER), "reused": True}
        space_guard(directory, reserve_gb)
        source = store / "citations.parquet"
        before = _signature(source)
        fingerprint, source_rows = _footer_fingerprint(source)
        artifact = directory / f"citation-counts-{uuid.uuid4().hex}.parquet"
        temporary = artifact.with_suffix(".tmp.parquet")
        try:
            (pl.scan_parquet(source).group_by("cited_pmid")
             .agg(pl.len().cast(pl.UInt64).alias("times_cited"))
             .sink_parquet(temporary, engine="streaming"))
            totals = (pl.scan_parquet(temporary)
                      .select(pl.len().alias("rows"), pl.col("times_cited").sum().alias("citations"))
                      .collect(engine="streaming").row(0, named=True))
            if totals["citations"] != source_rows or _signature(source) != before:
                raise RuntimeError("Citation source changed or summary count does not match")
            os.replace(temporary, artifact)
            with artifact.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            manifest = {"format": FORMAT, "snapshot": cfg["snapshot"], "sha256": checksum,
                        "source_footer_sha256": fingerprint, "source_rows": source_rows,
                        "file": artifact.name, "rows": totals["rows"], "bytes": artifact.stat().st_size}
            atomic_json(directory / MARKER, manifest)
            _validated_path.cache_clear()
            return {**manifest, "reused": False}
        finally:
            # This unique temporary file belongs to this build only.
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    args = parser.parse_args()
    if args.dataset:
        os.environ["PUBMED_DATASET"] = str(Path(args.dataset).resolve())
    print(json.dumps(build_citation_counts(), indent=2))
