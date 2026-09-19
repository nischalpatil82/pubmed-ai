"""XML -> bounded Parquet event batches -> reconciled immutable fact snapshot.

Each event owns all child rows. Last event in ordered sources wins, including
deletion and subsequent reintroduction. Checkpoints publish only fully read,
CRC-validated members. A failed member is replayed, never partially committed.
"""
import argparse
import json
from pathlib import Path
import zipfile

import polars as pl
import pyarrow.parquet as pq
from lxml import etree

from build_store import SCHEMAS, INT_COLS, BOOL_COLS, DEDUP_KEYS, country_from_affiliation, build_vocabulary
from dataset import atomic_json, read_json, digest, space_guard, BuildLock
from pubmed_ingest import source_jobs, open_source, handle_article, text

VERSION = "events-v1"

# These tables are partitioned by their owning PMID during reconciliation. Any
# duplicate key is therefore confined to one bucket, so each bucket can be
# deduplicated and appended independently instead of materializing the entire
# corpus in memory.
OWNER_PARTITIONED = {
    "articles", "authorships", "mesh_headings", "substances", "keywords",
    "citations", "publication_types", "grants", "databank_links", "deleted_pmids",
}


class Rows:
    def __init__(self):
        self.rows = []

    def writerow(self, row):
        self.rows.append(row)


def events(path, member=None):
    with open_source(str(path), member) as source:
        context = etree.iterparse(source, events=("end",),
                                  tag=("PubmedArticle", "DeleteCitation", "PubmedBookArticle"),
                                  load_dtd=False, no_network=True, resolve_entities=False)
        for _, node in context:
            if node.tag == "PubmedBookArticle":
                raise ValueError("PubmedBookArticle requires a separate schema adapter")
            if node.tag == "DeleteCitation":
                for item in node.findall("PMID"):
                    pmid = (item.text or "").strip()
                    if not pmid.isdigit():
                        raise ValueError("Invalid deletion PMID")
                    yield pmid, True, {}
            else:
                writers = {table: Rows() for table in SCHEMAS}
                handle_article(node, writers, set(), set())
                if not writers["articles"].rows:
                    raise ValueError("Article missing PMID or Article element")
                payload = {t: w.rows for t, w in writers.items() if w.rows}
                pmid = payload["articles"][0][0]
                if not pmid.isdigit():
                    raise ValueError("Invalid article PMID")
                payload["revision"] = "-".join(text(node, f"MedlineCitation/DateRevised/{part}", "")
                                                for part in ("Year", "Month", "Day"))
                yield pmid, False, payload
            node.clear()
            while node.getprevious() is not None:
                del node.getparent()[0]
        # Force EOF for gzip/ZIP trailer validation even if parser behavior changes.
        while source.read(1024 * 1024):
            pass


def source_identity(src, jobs):
    if Path(src).suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as z:
            return [{"name": m, "crc": z.getinfo(m).CRC, "bytes": z.getinfo(m).file_size}
                    for _, m in jobs]
    import hashlib
    out = []
    for path, _ in jobs:
        with open(path, "rb") as f:
            checksum = hashlib.file_digest(f, "sha256").hexdigest()
        out.append({"name": Path(path).name, "sha256": checksum})
    return out


def table_frame(table, rows):
    schema = {c: pl.String for c in SCHEMAS[table]}
    df = pl.DataFrame([[None if v is None else str(v) for v in row] for row in rows],
                      schema=schema, orient="row")
    casts = [pl.col(c).cast(pl.Int32, strict=True) for c in schema if c in INT_COLS]
    casts += [(pl.col(c) == "1").alias(c) for c in schema if c in BOOL_COLS]
    df = df.with_columns(casts)
    if table == "authorships":
        df = df.with_columns(country_from_affiliation(pl.col("affiliation")).alias("country"))
    return df


def output_columns(table):
    columns = list(SCHEMAS[table])
    if table == "articles":
        columns.extend(("source_member", "source_order", "source_ordinal",
                        "date_revised_full", "content_hash"))
    elif table == "authorships":
        columns.append("country")
    return columns


def valid_parquet(path, columns):
    path = Path(path)
    if not path.is_file():
        return False
    try:
        return pq.ParquetFile(path).schema_arrow.names == list(columns)
    except (OSError, ValueError):
        return False


def finalize_owner_partitioned(table, parts, destination):
    """Write one corpus table while keeping memory bounded to one PMID bucket."""
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    writer = None
    rows = 0
    try:
        for part in parts:
            frame = pl.read_parquet(part)
            keys = DEDUP_KEYS.get(table)
            if keys and frame.height:
                frame = frame.unique(subset=keys, keep="last", maintain_order=False)
            arrow = frame.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(temporary, arrow.schema, compression="zstd")
            if arrow.num_rows:
                writer.write_table(arrow, row_group_size=100_000)
                rows += arrow.num_rows
        if writer is None:
            raise RuntimeError(f"No reconciled parts found for {table}")
    finally:
        if writer is not None:
            writer.close()
    if not valid_parquet(temporary, output_columns(table)):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Invalid finalized table: {temporary}")
    temporary.replace(destination)
    return rows


def build(src, root, max_files=None, batch=1000, buckets=128, reserve_gb=2, sample=False, mesh=None):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if batch < 1 or buckets < 1 or (max_files is not None and max_files < 1):
        raise ValueError("Batch, bucket and file limits must be positive")
    all_jobs = source_jobs(str(src))
    if not all_jobs:
        raise ValueError("No XML members found")
    if sample and max_files and max_files < len(all_jobs):
        selected = sorted({round(i * (len(all_jobs) - 1) / max(1, max_files - 1)) for i in range(max_files)})
        jobs = [all_jobs[i] for i in selected]
    else:
        jobs = all_jobs[:max_files] if max_files else all_jobs
    identity = source_identity(src, jobs)
    event_snapshot = digest({"sources": identity, "parser": VERSION, "buckets": buckets})[:20]
    mesh_hash = None
    if mesh:
        import hashlib
        with open(mesh, "rb") as source:
            mesh_hash = hashlib.file_digest(source, "sha256").hexdigest()
    snapshot = digest({"events": event_snapshot, "store_version": "facts-v2", "mesh_sha256": mesh_hash})[:20]
    stage = root / "staging" / event_snapshot
    store = root / "stores" / snapshot
    index = root / "indexes" / snapshot
    stage.mkdir(parents=True, exist_ok=True)
    with BuildLock(root / "build.lock"):
        checkpoint_path = stage / "checkpoint.json"
        checkpoint = read_json(checkpoint_path) if checkpoint_path.exists() else {"members": {}}
        for number, (path, member) in enumerate(jobs):
            key = str(number)
            done = checkpoint["members"].get(key)
            if done and all((stage / f).exists() for f in done["shards"]):
                continue
            shards, buffer, buffered_bytes = [], [], 0
            ordinal = 0
            deleted = 0

            def flush():
                nonlocal buffer, buffered_bytes
                if not buffer:
                    return
                space_guard(root, reserve_gb)
                name = f"source-{number:04d}-batch-{len(shards):06d}.parquet"
                target = stage / name
                pl.DataFrame(buffer).sort("bucket").write_parquet(target.with_suffix(".tmp"),
                                                     compression="zstd", row_group_size=128)
                target.with_suffix(".tmp").replace(target)
                shards.append(name)
                buffer, buffered_bytes = [], 0

            for pmid, is_deleted, payload in events(path, member):
                encoded = json.dumps(payload, ensure_ascii=False)
                buffer.append({"pmid": pmid, "deleted": is_deleted, "payload": encoded,
                               "source_order": number, "ordinal": ordinal, "bucket": int(pmid) % buckets})
                ordinal += 1
                deleted += int(is_deleted)
                buffered_bytes += len(encoded.encode("utf-8"))
                if len(buffer) >= batch or buffered_bytes >= 16 * 1024 * 1024:
                    flush()
            flush()
            if not ordinal:
                raise ValueError(f"Source contains no supported records: {member or path}")
            checkpoint["members"][key] = {"source": identity[number], "shards": shards,
                                            "events": ordinal, "deletions": deleted, "validated": True}
            atomic_json(checkpoint_path, checkpoint)
            print(f"Validated {number + 1}/{len(jobs)}: {member or path}; {ordinal:,} events", flush=True)

        store.mkdir(parents=True, exist_ok=True)
        if not (store / "manifest.json").exists():
            files = [str(stage / s) for m in checkpoint["members"].values() for s in m["shards"]]
            parts = stage / "reconciled"
            parts.mkdir(exist_ok=True)
            for bucket in range(buckets):
                bucket_outputs = [parts / f"{table}-{bucket:04d}.parquet" for table in SCHEMAS]
                if all(valid_parquet(path, output_columns(table))
                       for table, path in zip(SCHEMAS, bucket_outputs)):
                    print(f"Reconciled bucket {bucket + 1}/{buckets}: already complete", flush=True)
                    continue
                space_guard(root, reserve_gb)
                winners = (pl.scan_parquet(files).filter(pl.col("bucket") == bucket)
                           .sort("source_order", "ordinal").unique("pmid", keep="last")
                           .collect(engine="streaming"))
                tables = {t: [] for t in SCHEMAS}
                provenance = []
                for row in winners.iter_rows(named=True):
                    if row["deleted"]:
                        tables["deleted_pmids"].append([row["pmid"]])
                        continue
                    payload = json.loads(row["payload"])
                    for table in SCHEMAS:
                        tables[table].extend(payload.get(table, []))
                    provenance.append({"pmid": row["pmid"], "source_member": identity[row["source_order"]]["name"],
                                       "source_order": row["source_order"], "source_ordinal": row["ordinal"],
                                       "date_revised_full": payload["revision"], "content_hash": digest(payload)})
                for table, rows in tables.items():
                    df = table_frame(table, rows)
                    if table == "articles":
                        p = pl.DataFrame(provenance, schema={"pmid": pl.String, "source_member": pl.String,
                            "source_order": pl.Int64, "source_ordinal": pl.Int64,
                            "date_revised_full": pl.String, "content_hash": pl.String})
                        df = df.join(p, on="pmid", how="left")
                    target = parts / f"{table}-{bucket:04d}.parquet"
                    temporary = target.with_suffix(target.suffix + ".tmp")
                    df.write_parquet(temporary, compression="zstd")
                    temporary.replace(target)
                print(f"Reconciled bucket {bucket + 1}/{buckets}", flush=True)
            counts = {}
            for table in SCHEMAS:
                destination = store / f"{table}.parquet"
                if valid_parquet(destination, output_columns(table)):
                    counts[table] = pq.ParquetFile(destination).metadata.num_rows
                    print(f"Finalized {table}: already complete ({counts[table]:,} rows)", flush=True)
                    continue
                table_parts = sorted(parts.glob(f"{table}-*.parquet"))
                if table in OWNER_PARTITIONED:
                    counts[table] = finalize_owner_partitioned(table, table_parts, destination)
                else:
                    # Journals and authors may repeat across PMID buckets, so they
                    # still require a global key deduplication. These dimensions
                    # are much smaller than the article-owned relationship tables.
                    lf = pl.scan_parquet(table_parts)
                    keys = DEDUP_KEYS.get(table)
                    if keys:
                        lf = lf.sort(keys).unique(keys, keep="last", maintain_order=False)
                    temporary = destination.with_suffix(destination.suffix + ".tmp")
                    temporary.unlink(missing_ok=True)
                    lf.sink_parquet(temporary, compression="zstd")
                    if not valid_parquet(temporary, output_columns(table)):
                        temporary.unlink(missing_ok=True)
                        raise RuntimeError(f"Invalid finalized table: {temporary}")
                    temporary.replace(destination)
                    counts[table] = pq.ParquetFile(destination).metadata.num_rows
                print(f"Finalized {table}: {counts[table]:,} rows", flush=True)
            vocabulary = store / "vocabulary.parquet"
            if not valid_parquet(vocabulary, ("concept_id", "concept_name", "kind", "code",
                                               "n_papers", "name_lc")):
                build_vocabulary(str(store))
            if mesh:
                from load_mesh_synonyms import load
                load(str(mesh), str(store))
            atomic_json(store / "manifest.json", {"snapshot": snapshot, "parser": VERSION, "counts": counts,
                "sources": identity, "source_members": len(jobs), "available_members": len(all_jobs),
                "all_selected_members_validated": True, "mesh_sha256": mesh_hash,
                "scope": "Selected source files only; historical baseline completeness unknown"})
        index.mkdir(parents=True, exist_ok=True)
        manifest = {"dataset_id": root.name, "snapshot": snapshot, "status": "ready",
                    "store": str(store.relative_to(root)), "index": str(index.relative_to(root)),
                    "pilot": len(jobs) < len(all_jobs)}
        atomic_json(root / "dataset.json", manifest)
        print(f"Snapshot ready: {root / 'dataset.json'}", flush=True)
        return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--root", default="covid-files")
    ap.add_argument("--max-files", type=int)
    ap.add_argument("--sample", action="store_true", help="Spread selected members across the source order")
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--buckets", type=int, default=128)
    ap.add_argument("--reserve-gb", type=float, default=2)
    ap.add_argument("--mesh", help="Optional NLM descriptor XML, fingerprinted with the snapshot")
    a = ap.parse_args()
    build(a.src, a.root, a.max_files, a.batch, a.buckets, a.reserve_gb, a.sample, a.mesh)
