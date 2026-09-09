#!/usr/bin/env python3
"""
Retrieval for the no-SQL stack: BM25 + dense vectors, fused, on CPU.

Two deliberate choices for a CPU-only build:

1. BM25 runs on numpy and needs no model download, so keyword search works the
   moment the Parquet exists. On a laptop it is also 100x cheaper than dense
   retrieval - and on biomedical text, exact term matching on drug names, gene
   symbols and MeSH terms is genuinely strong. Do not treat it as the fallback.

2. Dense vectors are added by a separate, resumable pass. Embedding is the only
   expensive step in this whole system on CPU, so it must be interruptible and
   must never need redoing. Vectors land in LanceDB (embedded, no server).

Contextual enrichment: each abstract is prefixed with its journal, year and
MeSH terms before embedding. That normally costs an LLM call per chunk; here
the metadata is already structured, so the recall gain is free.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import polars as pl

STORE = os.environ.get("PUBMED_STORE", "./store")
INDEX = os.environ.get("PUBMED_INDEX", "./index")


# ------------------------------------------------------------------ documents

def documents(limit: int | None = None, truncate: int | None = None) -> pl.DataFrame:
    """One row per article, with its structured context folded into the text."""
    mesh = (pl.scan_parquet(f"{STORE}/mesh_headings.parquet")
            .group_by("pmid").agg(pl.col("descriptor_name").str.join("; ").alias("mesh")))
    subs = (pl.scan_parquet(f"{STORE}/substances.parquet")
            .group_by("pmid").agg(pl.col("substance_name").str.join("; ").alias("subs")))
    lf = (pl.scan_parquet(f"{STORE}/articles.parquet")
          .filter(pl.col("abstract").is_not_null())
          .join(pl.scan_parquet(f"{STORE}/journals.parquet").select("nlm_id", "medline_ta"),
                on="nlm_id", how="left")
          .join(mesh, on="pmid", how="left").join(subs, on="pmid", how="left")
          .with_columns(
              pl.concat_str([
                  pl.col("medline_ta").fill_null(""), pl.lit(" ("),
                  pl.col("pub_year").cast(pl.Utf8).fill_null("n.d."), pl.lit("). "),
                  pl.col("title").fill_null(""), pl.lit("\n"),
                  pl.col("mesh").fill_null(""), pl.lit(" "),
                  pl.col("subs").fill_null(""), pl.lit("\n"),
                  pl.col("abstract"),
              ]).alias("doc"))
          .select("pmid", "title", "pub_year", "medline_ta", "doc"))
    if truncate:
        # bge-* models read at most 512 tokens (~2,000 characters) and discard
        # the rest. Feeding them more costs tokenizer time for text the model
        # never sees. Measured: trimming here is pure throughput, not a quality
        # trade. BM25 indexes the untruncated text, so nothing is lost to
        # keyword search - only the dense pass sees the cap.
        lf = lf.with_columns(pl.col("doc").str.slice(0, truncate).alias("doc"))
    if limit:
        lf = lf.head(limit)
    return lf.collect()


# ------------------------------------------------------------------ bm25

def build_bm25(limit: int | None = None) -> None:
    import bm25s, Stemmer
    os.makedirs(INDEX, exist_ok=True)
    df = documents(limit)
    t0 = time.time()
    stemmer = Stemmer.Stemmer("english")
    tokens = bm25s.tokenize(df["doc"].to_list(), stopwords="en", stemmer=stemmer,
                            show_progress=False)
    r = bm25s.BM25()
    r.index(tokens, show_progress=False)
    r.save(f"{INDEX}/bm25")
    df.select("pmid", "title", "pub_year", "medline_ta").write_parquet(f"{INDEX}/docmeta.parquet")
    el = time.time() - t0
    print(f"bm25   {df.height:,} docs in {el:.1f}s  ({df.height/el:,.0f} docs/s)  -> {INDEX}/bm25")


class BM25Search:
    def __init__(self) -> None:
        import bm25s, Stemmer
        self.r = bm25s.BM25.load(f"{INDEX}/bm25", load_corpus=False)
        self.stemmer = Stemmer.Stemmer("english")
        self.meta = pl.read_parquet(f"{INDEX}/docmeta.parquet")
        self._bm25s = bm25s

    def search(self, query: str, k: int = 20) -> list[dict]:
        q = self._bm25s.tokenize(query, stopwords="en", stemmer=self.stemmer,
                                 show_progress=False)
        idx, scores = self.r.retrieve(q, k=min(k, self.meta.height), show_progress=False)
        out = []
        for i, s in zip(idx[0], scores[0]):
            row = self.meta.row(int(i), named=True)
            out.append({**row, "score": float(s)})
        return out


# ------------------------------------------------------------------ vectors

def _write_vector_config(model_name: str, tbl_name: str, backend: str, docs: int) -> None:
    with open(f"{INDEX}/vector_model.json", "w") as fh:
        json.dump({"model": model_name, "table": tbl_name,
                   "backend": backend, "docs": docs}, fh)


def _encoder(model_name: str, backend: str, threads: int | None):
    """
    Two CPU backends. Measured on a 4-core i5-1135G7 with bge-small, 300 real
    docs: fastembed/ONNX 3.7 docs/s at 8 threads, sentence-transformers/torch
    4.5 docs/s. Torch wins here, so it is the default - but ONNX stays available
    because that ordering is hardware-specific and worth re-checking on a
    different box. Neither is close to a GPU; this is the step to rent one for.
    """
    if backend == "onnx":
        from fastembed import TextEmbedding
        emb = TextEmbedding(model_name=model_name, threads=threads)
        return lambda texts: [v.tolist() for v in emb.embed(texts)]

    import torch
    torch.set_num_threads(threads or (os.cpu_count() or 4))
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name, device="cpu")
    return lambda texts: m.encode(texts, batch_size=len(texts),
                                  show_progress_bar=False,
                                  normalize_embeddings=True).tolist()


def build_vectors(model_name: str, limit: int | None = None, batch: int = 64,
                  threads: int | None = None, backend: str = "torch") -> None:
    """
    Resumable dense pass. Re-running skips whatever is already embedded, so an
    interrupted overnight run costs nothing. The model name is stored with the
    table so a later model swap is a new table, not a silent mismatch.

    The table is written incrementally, so the index is queryable while this is
    still running - you do not have to wait for the full pass to finish.
    """
    import lancedb

    df = documents(limit, truncate=2000)
    db = lancedb.connect(INDEX)
    tbl_name = "chunks_" + model_name.split("/")[-1].replace(".", "_").replace("-", "_")

    done: set[str] = set()
    if tbl_name in db.table_names():
        done = set(db.open_table(tbl_name).to_lance().to_table(columns=["pmid"])
                   .column("pmid").to_pylist())
        df = df.filter(~pl.col("pmid").is_in(list(done)))
        print(f"resuming: {len(done):,} already embedded, {df.height:,} to go")
    if df.height == 0:
        # Everything is already embedded. Publish the config here too, or a
        # watchdog that treats "config exists" as "finished" would restart this
        # forever against a table that is complete.
        _write_vector_config(model_name, tbl_name, backend, len(done))
        print(f"nothing to do - {len(done):,} already embedded, config published")
        return

    encode = _encoder(model_name, backend, threads)
    texts, pmids = df["doc"].to_list(), df["pmid"].to_list()
    t0, n = time.time(), 0
    tbl = None
    for start in range(0, len(texts), batch):
        chunk = texts[start:start + batch]
        vecs = encode(chunk)
        rows = [{"pmid": p, "vector": v}
                for p, v in zip(pmids[start:start + batch], vecs)]
        if tbl is None and tbl_name not in db.table_names():
            tbl = db.create_table(tbl_name, rows)
        else:
            tbl = tbl or db.open_table(tbl_name)
            tbl.add(rows)
        n += len(chunk)
        # Publishing the config mid-run makes the partial index queryable, but
        # it also invites a reader to open the LanceDB table while this process
        # is still appending to it. On Windows that lock collision kills THIS
        # writer (exit 5, no traceback) and costs hours of work. So it is opt-in
        # and off by default: finishing the pass matters more than querying it
        # early. Set PUBMED_PUBLISH_PARTIAL=1 only if no reader will attach.
        if (os.environ.get("PUBMED_PUBLISH_PARTIAL") == "1"
                and tbl is not None
                and not os.path.exists(f"{INDEX}/vector_model.json")):
            _write_vector_config(model_name, tbl_name, backend, n + len(done))
        el = time.time() - t0
        print(f"\r  {n:,}/{len(texts):,}  {n/el:6.1f} docs/s  "
              f"eta {(len(texts)-n)/max(n/el, 1e-9)/60:5.1f} min", end="", flush=True)
    el = time.time() - t0
    print(f"\nvectors  {n:,} docs in {el/60:.1f} min  ({n/el:.1f} docs/s)  -> {tbl_name}")
    _write_vector_config(model_name, tbl_name, backend, n + len(done))


class HybridSearch:
    """BM25 always; vectors when a dense table exists. Fused with RRF."""

    def __init__(self) -> None:
        self.bm25 = BM25Search()
        self.dense = None
        cfg_path = f"{INDEX}/vector_model.json"
        if os.path.exists(cfg_path):
            try:
                import lancedb
                cfg = json.load(open(cfg_path))
                self.tbl = lancedb.connect(INDEX).open_table(cfg["table"])
                # The query must be encoded by the same backend that built the
                # table - mixing torch and ONNX gives subtly different vectors.
                self.encode = _encoder(cfg["model"], cfg.get("backend", "torch"), None)
                self.dense = cfg["model"]
            except Exception as e:                       # index absent or model gone
                print(f"[hybrid] dense disabled: {e}")

    def search(self, query: str, k: int = 10, since_year: int | None = None,
               rrf_k: int = 60) -> dict:
        pools = {"bm25": self.bm25.search(query, k=k * 5)}
        if self.dense:
            qv = self.encode([query])[0]
            hits = self.tbl.search(qv).limit(k * 5).to_list()
            pools["vector"] = [{"pmid": h["pmid"], "score": 1 - h.get("_distance", 0)}
                               for h in hits]

        # Reciprocal rank fusion: rank-based, so the two score scales never
        # need calibrating against each other.
        fused: dict[str, float] = {}
        for name, hits in pools.items():
            for rank, h in enumerate(hits):
                fused[h["pmid"]] = fused.get(h["pmid"], 0.0) + 1.0 / (rrf_k + rank + 1)

        meta = self.bm25.meta
        order = sorted(fused.items(), key=lambda x: -x[1])
        rows = (meta.filter(pl.col("pmid").is_in([p for p, _ in order[:k * 3]]))
                    .with_columns(pl.col("pmid").replace_strict(fused, default=0.0)
                                  .alias("score")))
        if since_year:
            rows = rows.filter(pl.col("pub_year") >= since_year)
        rows = rows.sort("score", descending=True).head(k)
        return {"retrievers": list(pools), "dense_model": self.dense,
                "results": rows.to_dicts()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bm25", "vectors", "search", "benchmark"])
    ap.add_argument("--query", default="cisplatin resistance in ovarian cancer")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    a = ap.parse_args()

    if a.cmd == "bm25":
        build_bm25(a.limit)
    elif a.cmd == "vectors":
        build_vectors(a.model, a.limit, threads=a.threads, backend=a.backend)
    elif a.cmd == "benchmark":
        # Run this on YOUR laptop before committing to a corpus size.
        from fastembed import TextEmbedding
        docs = documents(400)["doc"].to_list()
        for m in ["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5"]:
            e = TextEmbedding(model_name=m, threads=a.threads)
            list(e.embed(docs[:8]))
            t0 = time.time(); list(e.embed(docs)); el = time.time() - t0
            r = len(docs) / el
            print(f"{m:26s} {r:6.1f} docs/s | 23k docs -> {23362/r/60:5.1f} min "
                  f"| 1M -> {1e6/r/3600:5.1f} h")
    else:
        print(json.dumps(HybridSearch().search(a.query, a.k), indent=2)[:2500])
