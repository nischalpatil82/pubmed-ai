"""Optional, snapshot-bound SQLite lookups for small sets of article records."""
import functools
from contextlib import closing
import hashlib
import os
from pathlib import Path
import sqlite3
import uuid

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from analytics_index import _footer_fingerprint, _signature
from dataset import BuildLock, atomic_json, paths, read_json, space_guard

MARKER = "article-lookup.json"
FORMAT = "article-lookup-v2"


def _connect(path):
    # Read-only connections are short-lived, so Windows updates do not retain handles.
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)


@functools.lru_cache(maxsize=2)
def _data_metadata(path, signature):
    # Only footers are retained, never article text or open Windows file handles.
    return pq.read_metadata(path)


@functools.lru_cache(maxsize=8)
def _validate(store, index, snapshot, source_signature, marker_signature):
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
        data_name = manifest.get("data_file", "")
        if not data_name or Path(data_name).name != data_name:
            return None
        data = (directory / data_name).resolve()
        if (data != artifact.with_suffix(".parquet") or data.parent != directory.resolve()
                or not data.is_file()):
            return None
        fingerprint, rows = _footer_fingerprint(Path(store) / "articles.parquet")
        if (manifest.get("source_footer_sha256") != fingerprint
                or manifest.get("rows") != rows or manifest.get("bytes") != artifact.stat().st_size):
            return None
        if (manifest.get("data_bytes") != data.stat().st_size
                or _data_metadata(data, _signature(data)).num_rows != rows
                or pl.read_parquet_schema(data) != pl.read_parquet_schema(Path(store) / "articles.parquet")):
            return None
        with artifact.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != manifest.get("sha256"):
                return None
        with data.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != manifest.get("data_sha256"):
                return None
        with closing(_connect(artifact)) as connection:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            if metadata != {"format": FORMAT, "snapshot": snapshot, "rows": str(rows)}:
                return None
        return artifact, data, (_signature(artifact), _signature(data))
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error):
        return None


def article_lookup_path(store, index, snapshot):
    try:
        source = Path(store) / "articles.parquet"
        marker = Path(index) / "analytics" / MARKER
        if not marker.is_file():
            return None
        arguments = (str(Path(store).resolve()), str(Path(index).resolve()), snapshot,
                     _signature(source), _signature(marker))
        found = _validate(*arguments)
        if found and (_signature(found[0]), _signature(found[1])) != found[2]:
            _validate.cache_clear()
            found = _validate(*arguments)
        return found[0] if found else None
    except OSError:
        return None


def build_article_lookup(reserve_gb=2):
    store, index, cfg = paths()
    directory = index / "analytics"
    directory.mkdir(parents=True, exist_ok=True)
    with BuildLock(directory / "article-build.lock"):
        existing = article_lookup_path(store, index, cfg["snapshot"])
        if existing is not None:
            return {**read_json(directory / MARKER), "reused": True}
        space_guard(directory, reserve_gb)
        source = store / "articles.parquet"
        before = _signature(source)
        fingerprint, rows = _footer_fingerprint(source)
        artifact = directory / f"article-lookup-{uuid.uuid4().hex}.sqlite"
        temporary = artifact.with_suffix(".tmp.sqlite")
        data = artifact.with_suffix(".parquet")
        data_temporary = data.with_suffix(".tmp.parquet")
        connection = None
        writer = None
        try:
            connection = sqlite3.connect(temporary)
            connection.execute("PRAGMA cache_size=-8192")
            connection.execute("CREATE TABLE records (pmid TEXT PRIMARY KEY, ordinal INTEGER NOT NULL, row_group INTEGER NOT NULL) WITHOUT ROWID")
            connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID")
            ordinal = 0
            source_reader = pq.ParquetFile(source)
            writer = pq.ParquetWriter(data_temporary, source_reader.schema_arrow, compression="zstd")
            for group, batch in enumerate(source_reader.iter_batches(batch_size=1000)):
                ids = batch.column(batch.schema.get_field_index("pmid")).to_pylist()
                writer.write_table(pa.Table.from_batches([batch]), row_group_size=1000)
                connection.executemany("INSERT INTO records VALUES (?, ?, ?)",
                                       ((pmid, ordinal + offset, group) for offset, pmid in enumerate(ids)))
                ordinal += len(ids)
                if group % 20 == 0:
                    connection.commit()
                space_guard(directory, reserve_gb)
            if ordinal != rows or _signature(source) != before:
                raise RuntimeError("Article source changed during lookup preparation")
            connection.executemany("INSERT INTO metadata VALUES (?, ?)",
                                   [("format", FORMAT), ("snapshot", cfg["snapshot"]), ("rows", str(rows))])
            connection.commit()
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise RuntimeError("Article lookup integrity check failed")
            connection.close()
            connection = None
            writer.close()
            writer = None
            if pq.read_metadata(data_temporary).num_rows != rows:
                raise RuntimeError("Article lookup data row count does not match source")
            os.replace(temporary, artifact)
            os.replace(data_temporary, data)
            with artifact.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            with data.open("rb") as stream:
                data_checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            manifest = {"format": FORMAT, "snapshot": cfg["snapshot"], "rows": rows,
                        "source_footer_sha256": fingerprint, "file": artifact.name,
                        "bytes": artifact.stat().st_size, "sha256": checksum,
                        "data_file": data.name, "data_bytes": data.stat().st_size,
                        "data_sha256": data_checksum}
            atomic_json(directory / MARKER, manifest)
            _validate.cache_clear()
            return {**manifest, "reused": False}
        finally:
            if connection is not None:
                connection.close()
            if writer is not None:
                writer.close()
            temporary.unlink(missing_ok=True)
            Path(str(temporary) + "-journal").unlink(missing_ok=True)
            data_temporary.unlink(missing_ok=True)


def article_records(store, index, snapshot, pmids, columns=None):
    """Lookup selected records; older/missing/stale indexes use the original scan."""
    source = Path(store) / "articles.parquet"
    schema = pl.read_parquet_schema(source)
    columns = list(schema) if columns is None else list(columns)
    projected = {name: schema[name] for name in columns}
    ids = list(dict.fromkeys(str(pmid) for pmid in pmids))
    if not ids:
        return pl.DataFrame(schema=projected)
    artifact = article_lookup_path(store, index, snapshot)
    if artifact is not None:
        try:
            found = []
            with closing(_connect(artifact)) as connection:
                # Stay below the SQLite bind-variable limit on all supported builds.
                for start in range(0, len(ids), 500):
                    subset = ids[start:start + 500]
                    found.extend(connection.execute(
                        "SELECT pmid, ordinal, row_group FROM records WHERE pmid IN ("
                        + ",".join("?" for _ in subset) + ")", subset).fetchall())
            if not found:
                return pl.DataFrame(schema=projected)
            data = artifact.with_suffix(".parquet")
            wanted = {item[0] for item in found}
            records = []
            read_columns = list(dict.fromkeys(["pmid", *columns]))
            with pq.ParquetFile(data, metadata=_data_metadata(data, _signature(data))) as reader:
                groups = sorted({item[2] for item in found})
                # Decode a bounded set of groups, then filter before Python objects.
                for start in range(0, len(groups), 8):
                    table = reader.read_row_groups(groups[start:start + 8], columns=read_columns)
                    records.extend(pl.from_arrow(table).filter(pl.col("pmid").is_in(list(wanted))).to_dicts())
            ordinals = {item[0]: item[1] for item in found}
            if {row["pmid"] for row in records} != wanted or len(records) != len(found):
                raise ValueError("Lookup record identity mismatch")
            records.sort(key=lambda row: ordinals[row["pmid"]])
            return pl.DataFrame([{column: row[column] for column in columns} for row in records], schema=projected)
        except (sqlite3.Error, OSError, ValueError, TypeError, KeyError, pa.ArrowInvalid):
            # Optional acceleration must never make the source records unavailable.
            pass
    return (pl.scan_parquet(source).filter(pl.col("pmid").is_in(ids))
            .select(columns).collect(engine="streaming"))
