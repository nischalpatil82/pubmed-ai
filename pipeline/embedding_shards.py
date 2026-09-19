"""Resumable multi-worker embedding parts for temporary GPU runtimes.

Workers read the same immutable articles.parquet but own disjoint PMIDs. Each
completed article batch produces one text Parquet file, one vector-only Parquet
file, and a checksummed manifest. Temporary files are never treated as complete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from dataset import atomic_json, digest, paths, read_json, space_guard
from retrieval import CHUNK_VERSION, _encoder, token_passages

FORMAT = "external-passage-parts-v1"


def encoder_pool(model):
    """Start one persistent encoder process per configured GPU."""
    value = os.environ.get("PUBMED_EMBED_DEVICES", "").strip()
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if len(devices) < 2:
        return None, devices
    return model.start_multi_process_pool(target_devices=devices), devices


def sha256_file(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def article_file():
    store, index, dataset = paths()
    return store / "articles.parquet", index, dataset


def batches(rows_per_part):
    articles, _, _ = article_file()
    for number, arrow in enumerate(pq.ParquetFile(articles).iter_batches(batch_size=rows_per_part)):
        yield number, pl.from_arrow(arrow)


def descriptor(model, dataset, model_name):
    revision = getattr(model[0].auto_model.config, "_commit_hash", None)
    if not revision:
        raise ValueError("Embedding model must resolve to a pinned revision")
    special_tokens = (model.tokenizer.num_special_tokens_to_add(pair=False)
                      if hasattr(model.tokenizer, "num_special_tokens_to_add") else 2)
    content_tokens = model.max_seq_length - special_tokens
    if content_tokens < 1:
        raise ValueError("Model sequence length leaves no room for content tokens")
    value = {"format": FORMAT, "snapshot": dataset["snapshot"],
             "model": model_name,
             "revision": revision, "chunk_version": CHUNK_VERSION,
             "max_tokens": model.max_seq_length, "normalized": True,
             "special_tokens": special_tokens, "content_tokens": content_tokens,
             "vector_dtype": "float32",
             "dimensions": model.get_sentence_embedding_dimension()}
    value["descriptor_hash"] = digest(value)
    return value


def make_passages(frame, model_info):
    rows = []
    for article in frame.iter_rows(named=True):
        sources = [("title", article.get("title") or "")]
        if article.get("abstract"):
            sources.append(("abstract", article["abstract"]))
        for section, source in sources:
            chunks = token_passages(source, model_info["tokenizer"],
                                     model_info["content_tokens"])
            for part, text in enumerate(chunks):
                chunk_id = digest([model_info["descriptor_hash"], article["pmid"],
                                   section, part, text])
                rows.append({"chunk_id": chunk_id, "pmid": article["pmid"],
                             "section": section, "part": part, "text": text,
                             "title": article.get("title") or "",
                             "pub_year": article.get("pub_year") or 0,
                             "source_member": article.get("source_member")})
    return rows


def part_paths(index, workers, worker, number):
    label = f"w{workers}-{worker}"
    passage = index / "passage-parts" / label / f"part-{number:06d}.parquet"
    vector = index / "vector-parts" / label / f"part-{number:06d}.parquet"
    manifest = index / "embedding-checkpoints" / label / f"part-{number:06d}.json"
    return passage, vector, manifest


def manifest_valid(path, expected, verify_files=False, imported=None):
    if not path.exists():
        return False
    try:
        saved = read_json(path)
        if any(saved.get(key) != value for key, value in expected.items()):
            return False
        for item in ("passage_file", "vector_file"):
            target = (path.parents[2] / saved[item]).resolve()
            if not target.is_relative_to(path.parents[2].resolve()):
                return False
            relative = target.relative_to(path.parents[2].resolve()).as_posix()
            already_imported = (item == "vector_file" and imported
                                and imported.get(relative) == saved[item + "_sha256"])
            if not target.exists() and already_imported:
                continue
            if not target.exists() or target.stat().st_size != saved[item + "_bytes"]:
                return False
            if verify_files and sha256_file(target) != saved[item + "_sha256"]:
                return False
        return True
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def write_frame(frame, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    frame.write_parquet(temporary, compression="zstd")
    os.replace(temporary, target)


def run_worker(model_name, worker, workers, rows_per_part=10000,
               gpu_batch=128, max_parts=None, reserve_gb=2,
               max_runtime_minutes=None):
    if workers < 1 or worker < 0 or worker >= workers:
        raise ValueError("worker must be between 0 and workers - 1")
    if rows_per_part < 1 or gpu_batch < 1:
        raise ValueError("Batch sizes must be positive")
    if max_runtime_minutes is not None and max_runtime_minutes <= 0:
        raise ValueError("Runtime limit must be positive")
    _, index, dataset = article_file()
    index.mkdir(parents=True, exist_ok=True)
    model = _encoder(model_name, threads=None)
    pool, devices = encoder_pool(model)
    info = descriptor(model, dataset, model_name)
    info.update({"tokenizer": model.tokenizer, "max_tokens": model.max_seq_length})
    worker_plan = {key: value for key, value in info.items() if key != "tokenizer"}
    worker_plan.update({"worker": worker, "workers": workers,
                        "rows_per_part": rows_per_part, "gpu_batch": gpu_batch})
    plan_path = index / "embedding-checkpoints" / f"w{workers}-{worker}" / "plan.json"
    if plan_path.exists():
        previous = read_json(plan_path)
        fixed = ("descriptor_hash", "worker", "workers", "rows_per_part")
        if any(previous.get(key) != worker_plan.get(key) for key in fixed):
            raise ValueError("Existing worker plan differs; choose another output index")
    atomic_json(plan_path, worker_plan)

    total_articles = pq.ParquetFile(article_file()[0]).metadata.num_rows
    complete_articles = complete_passages = new_parts = 0
    stopped_for_time_limit = False
    started = time.perf_counter()
    for number, frame in batches(rows_per_part):
        selected = frame.filter(pl.col("pmid").cast(pl.UInt64) % workers == worker)
        expected = {"snapshot": dataset["snapshot"],
                    "descriptor_hash": info["descriptor_hash"],
                    "worker": worker, "workers": workers, "part": number,
                    "articles": selected.height}
        passage_path, vector_path, manifest_path = part_paths(index, workers, worker, number)
        if manifest_valid(manifest_path, expected):
            saved = read_json(manifest_path)
            complete_articles += saved["articles"]
            complete_passages += saved["passages"]
            continue
        if max_parts is not None and new_parts >= max_parts:
            break
        if (max_runtime_minutes is not None and
                time.perf_counter() - started >= max_runtime_minutes * 60):
            stopped_for_time_limit = True
            print(f"worker {worker}: runtime limit reached; resume this cell later", flush=True)
            break
        space_guard(index, reserve_gb)
        passages = make_passages(selected, info)
        if passages:
            vectors = model.encode([row["text"] for row in passages],
                                   batch_size=gpu_batch, normalize_embeddings=True,
                                   show_progress_bar=True, convert_to_numpy=True,
                                   pool=pool)
            vectors = np.asarray(vectors, dtype=np.float32)
            passage_frame = pl.DataFrame(passages)
            relative_passage = passage_path.relative_to(index).as_posix()
            vector_frame = passage_frame.select("chunk_id", "pmid", "pub_year").with_columns(
                pl.Series("passage_file", [relative_passage] * len(passages)),
                pl.Series("vector", vectors.tolist(), dtype=pl.Array(pl.Float32, vectors.shape[1])))
        else:
            passage_frame = pl.DataFrame(schema={"chunk_id": pl.String, "pmid": pl.String,
                "section": pl.String, "part": pl.Int64, "text": pl.String,
                "title": pl.String, "pub_year": pl.Int64, "source_member": pl.String})
            vector_frame = pl.DataFrame(schema={"chunk_id": pl.String, "pmid": pl.String,
                "pub_year": pl.Int64, "passage_file": pl.String,
                "vector": pl.Array(pl.Float32, info["dimensions"])})
        write_frame(passage_frame, passage_path)
        write_frame(vector_frame, vector_path)
        record = {**expected, "passages": len(passages),
                  "passage_file": passage_path.relative_to(index).as_posix(),
                  "passage_file_bytes": passage_path.stat().st_size,
                  "passage_file_sha256": sha256_file(passage_path),
                  "vector_file": vector_path.relative_to(index).as_posix(),
                  "vector_file_bytes": vector_path.stat().st_size,
                  "vector_file_sha256": sha256_file(vector_path)}
        atomic_json(manifest_path, record)
        complete_articles += selected.height
        complete_passages += len(passages)
        new_parts += 1
        elapsed = time.perf_counter() - started
        print(f"worker {worker}: part {number}, {complete_articles:,} articles, "
              f"{complete_passages:,} passages, {complete_passages / max(elapsed, 1):.1f} passages/s",
              flush=True)
    if pool is not None:
        model.stop_multi_process_pool(pool)
    return {"worker": worker, "workers": workers, "articles": complete_articles,
            "passages": complete_passages, "corpus_articles": total_articles,
            "new_parts": new_parts, "stopped_for_time_limit": stopped_for_time_limit,
            "devices": devices or [str(model.device)],
            "descriptor": worker_plan}


def benchmark(model_name, articles=10000, gpu_batch=128):
    _, _, dataset = article_file()
    model = _encoder(model_name, threads=None)
    pool, devices = encoder_pool(model)
    info = descriptor(model, dataset, model_name)
    info.update({"tokenizer": model.tokenizer, "max_tokens": model.max_seq_length})
    selected = []
    for _, frame in batches(min(articles, 10000)):
        selected.append(frame.head(max(0, articles - sum(f.height for f in selected))))
        if sum(f.height for f in selected) >= articles:
            break
    frame = pl.concat(selected)
    passages = make_passages(frame, info)
    texts = [row["text"] for row in passages]
    if texts:
        model.encode(texts[:min(gpu_batch, len(texts))], batch_size=gpu_batch,
                     normalize_embeddings=True, show_progress_bar=False, pool=pool)
    started = time.perf_counter()
    model.encode(texts, batch_size=gpu_batch, normalize_embeddings=True,
                 show_progress_bar=True, convert_to_numpy=True, pool=pool)
    elapsed = time.perf_counter() - started
    if pool is not None:
        model.stop_multi_process_pool(pool)
    total_articles = pq.ParquetFile(article_file()[0]).metadata.num_rows
    ratio = len(passages) / max(frame.height, 1)
    projected = round(total_articles * ratio)
    average_text_bytes = sum(len(text.encode("utf-8")) for text in texts) / max(len(texts), 1)
    dimensions = model.get_sentence_embedding_dimension()
    return {"snapshot": dataset["snapshot"],
            "devices": devices or [str(model.device)],
            "model": model_name, "articles": frame.height, "passages": len(passages),
            "seconds": round(elapsed, 2), "passages_per_second": round(len(passages) / elapsed, 2),
            "observed_passages_per_article": round(ratio, 3),
            "projected_corpus_passages": projected,
            "dimensions": dimensions,
            "projected_raw_vector_gb": round(projected * dimensions * 4 / 1e9, 2),
            "projected_passage_text_gb": round(projected * average_text_bytes / 1e9, 2),
            "projected_runtime_hours": round(projected / (len(passages) / elapsed) / 3600, 2)}


def expected_parts(rows_per_part):
    articles, _, _ = article_file()
    return math.ceil(pq.ParquetFile(articles).metadata.num_rows / rows_per_part)


def verify(workers, rows_per_part, model_name=None, checksums=True, imported=None):
    _, index, dataset = article_file()
    plans = []
    for worker in range(workers):
        plan_path = index / "embedding-checkpoints" / f"w{workers}-{worker}" / "plan.json"
        if not plan_path.exists():
            raise ValueError(f"Worker {worker} has not started")
        plans.append(read_json(plan_path))
    hashes = {plan["descriptor_hash"] for plan in plans}
    if len(hashes) != 1 or any(plan["snapshot"] != dataset["snapshot"] for plan in plans):
        raise ValueError("Workers used different model/chunk or dataset versions")
    if any(plan["rows_per_part"] != rows_per_part for plan in plans):
        raise ValueError("rows-per-part differs from the saved worker plan")
    articles = passages = vector_bytes = passage_bytes = 0
    manifests = []
    for number in range(expected_parts(rows_per_part)):
        for worker in range(workers):
            _, _, manifest_path = part_paths(index, workers, worker, number)
            expected = {"snapshot": dataset["snapshot"], "descriptor_hash": plans[0]["descriptor_hash"],
                        "worker": worker, "workers": workers, "part": number}
            if not manifest_valid(manifest_path, expected, checksums, imported=imported):
                raise ValueError(f"Missing or invalid worker {worker}, part {number}")
            item = read_json(manifest_path)
            manifests.append(str(manifest_path.resolve()))
            articles += item["articles"]
            passages += item["passages"]
            vector_bytes += item["vector_file_bytes"]
            passage_bytes += item["passage_file_bytes"]
    corpus_articles = pq.ParquetFile(article_file()[0]).metadata.num_rows
    if articles != corpus_articles:
        raise ValueError(f"Worker article total {articles:,} != corpus {corpus_articles:,}")
    return {"complete": True, "snapshot": dataset["snapshot"],
            "descriptor_hash": plans[0]["descriptor_hash"], "articles": articles,
            "passages": passages, "vector_bytes": vector_bytes,
            "passage_bytes": passage_bytes, "workers": workers,
            "rows_per_part": rows_per_part, "manifests": manifests, "plan": plans[0]}


def finalize(workers, rows_per_part, ann=True, reserve_gb=2, consume_vector_parts=False):
    import lancedb
    _, index, dataset = article_file()
    plan_path = index / "embedding-checkpoints" / f"w{workers}-0" / "plan.json"
    if not plan_path.exists():
        raise ValueError("Worker 0 has not started")
    table_name = "vectors_" + read_json(plan_path)["descriptor_hash"][:24]
    checkpoint_path = index / f"{table_name}.finalize.json"
    checkpoint = read_json(checkpoint_path) if checkpoint_path.exists() else {"imported": {}}
    summary = verify(workers, rows_per_part, checksums=True,
                     imported=checkpoint["imported"] if consume_vector_parts else None)
    db = lancedb.connect(str(index))
    existing = db.table_names()
    table = db.open_table(table_name) if table_name in existing else None
    expected_chunks = 0
    for manifest_path in summary["manifests"]:
        item = read_json(manifest_path)
        vector_path = (index / item["vector_file"]).resolve()
        if not vector_path.is_relative_to(index.resolve()):
            raise ValueError("Unsafe vector part path")
        key = vector_path.relative_to(index).as_posix()
        expected_chunks += item["passages"]
        if checkpoint["imported"].get(key) == item["vector_file_sha256"]:
            if consume_vector_parts:
                vector_path.unlink(missing_ok=True)
            continue
        space_guard(index, reserve_gb)
        arrow = pl.read_parquet(vector_path).to_arrow()
        if table is None:
            table = db.create_table(table_name, arrow)
        elif arrow.num_rows:
            table.merge_insert("chunk_id").when_matched_update_all().when_not_matched_insert_all().execute(arrow)
        checkpoint["imported"][key] = item["vector_file_sha256"]
        atomic_json(checkpoint_path, checkpoint)
        if consume_vector_parts:
            vector_path.unlink()
        print(f"finalize: {len(checkpoint['imported'])}/{len(summary['manifests'])} parts", flush=True)
    count = table.count_rows() if table else 0
    if count != expected_chunks:
        raise ValueError(f"Final vector count {count:,} != expected {expected_chunks:,}")
    ann_ready = False
    if ann and count >= 1024:
        _create_ann(table, count)
        ann_ready = True
    plan = summary["plan"]
    config = {"format": FORMAT, "snapshot": dataset["snapshot"],
              "model": plan["model"], "revision": plan["revision"],
              "backend": "torch", "table": table_name, "docs": count,
              "expected_chunks": expected_chunks, "complete": True,
              "ann": ann_ready, "passage_glob": "passage-parts/**/*.parquet",
              "workers": workers, "rows_per_part": rows_per_part,
              "vector_parts_retained": not consume_vector_parts,
              "chunk_version": plan["chunk_version"],
              "descriptor_hash": summary["descriptor_hash"]}
    atomic_json(index / "vector_model.json", config)
    return config


def _create_ann(table, count, num_partitions=None, num_sub_vectors=48):
    """Build a compressed cosine index without duplicating all float32 vectors."""
    from lancedb.index import BTree, IvfPq

    dimension = table.schema.field("vector").type.list_size
    if num_sub_vectors < 1 or dimension % num_sub_vectors:
        raise ValueError(
            f"ANN sub-vectors must divide vector dimension {dimension}; got {num_sub_vectors}"
        )
    partitions = num_partitions or max(1, int(math.sqrt(count)))
    table.create_index(
        "vector",
        config=IvfPq(
            distance_type="cosine",
            num_partitions=partitions,
            num_sub_vectors=num_sub_vectors,
            num_bits=8,
        ),
        replace=True,
    )
    table.create_index("pub_year", config=BTree(), replace=True)
    return {"type": "IVF_PQ", "metric": "cosine", "num_partitions": partitions,
            "num_sub_vectors": num_sub_vectors, "num_bits": 8}


def build_ann(reserve_gb=2, num_partitions=None, num_sub_vectors=48):
    """Add a compressed ANN index to an already finalized external vector table."""
    import lancedb

    _, index, dataset = article_file()
    marker = index / "vector_model.json"
    if not marker.exists():
        raise ValueError("Finalize the vector parts before building ANN")
    config = read_json(marker)
    if not config.get("complete") or config.get("snapshot") != dataset["snapshot"]:
        raise ValueError("Vector table is incomplete or belongs to another snapshot")
    space_guard(index, reserve_gb)
    table = lancedb.connect(str(index)).open_table(config["table"])
    count = table.count_rows()
    if count != config["expected_chunks"]:
        raise ValueError(f"Final vector count {count:,} != expected {config['expected_chunks']:,}")
    if count < 1024:
        raise ValueError("ANN requires at least 1,024 passages")
    ann_config = _create_ann(table, count, num_partitions, num_sub_vectors)
    config.update({"ann": True, "ann_config": ann_config})
    atomic_json(marker, config)
    return config


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["benchmark", "worker", "verify", "finalize", "index-ann"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rows-per-part", type=int, default=10000)
    ap.add_argument("--gpu-batch", type=int, default=128)
    ap.add_argument("--benchmark-articles", type=int, default=10000)
    ap.add_argument("--max-parts", type=int)
    ap.add_argument("--max-runtime-minutes", type=float)
    ap.add_argument("--reserve-gb", type=float, default=2)
    ap.add_argument("--no-ann", action="store_true")
    ap.add_argument("--consume-vector-parts", action="store_true",
                    help="delete each local vector part only after its checked import")
    ap.add_argument("--ann-partitions", type=int)
    ap.add_argument("--ann-sub-vectors", type=int, default=48)
    ap.add_argument("--skip-checksums", action="store_true")
    args = ap.parse_args()
    os.environ["PUBMED_DATASET"] = str(Path(args.dataset).resolve())
    if args.command == "benchmark":
        result = benchmark(args.model, args.benchmark_articles, args.gpu_batch)
    elif args.command == "worker":
        result = run_worker(args.model, args.worker, args.workers, args.rows_per_part,
                            args.gpu_batch, args.max_parts, args.reserve_gb,
                            args.max_runtime_minutes)
    elif args.command == "verify":
        result = verify(args.workers, args.rows_per_part, args.model,
                        checksums=not args.skip_checksums)
    elif args.command == "finalize":
        result = finalize(args.workers, args.rows_per_part,
                          ann=not args.no_ann, reserve_gb=args.reserve_gb,
                          consume_vector_parts=args.consume_vector_parts)
    else:
        result = build_ann(args.reserve_gb, args.ann_partitions, args.ann_sub_vectors)
    print(json.dumps(result, indent=2))
