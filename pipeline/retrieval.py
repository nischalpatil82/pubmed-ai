"""Bounded lexical builds, versioned passage vectors and filter-aware retrieval."""
from __future__ import annotations
import argparse
import json
import math
import os
import re
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from dataset import paths, atomic_json, read_json, digest, space_guard, BuildLock

CHUNK_VERSION = "token-offset-v1"
BGE_SMALL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


def article_batches(batch=2000):
    store, _, _ = paths()
    for arrow in pq.ParquetFile(store / "articles.parquet").iter_batches(batch_size=batch):
        yield pl.from_arrow(arrow)


def documents(limit=None, truncate=None):
    """Small benchmark helper; production builds stream batches without truncation."""
    store, _, _ = paths()
    lf = pl.scan_parquet(store / "articles.parquet").select("pmid", "title", "abstract", "pub_year")
    if limit is not None:
        lf = lf.head(limit)
    return lf.with_columns(pl.concat_str([pl.col("title").fill_null(""),
             pl.col("abstract").fill_null("")], separator="\n").alias("doc")).collect()


def build_bm25(limit=None, shard_size=20000):
    import bm25s, Stemmer
    _, index, cfg = paths()
    index.mkdir(parents=True, exist_ok=True)
    with BuildLock(index / "build.lock"):
        shards, total = [], 0
        for n, df in enumerate(article_batches(shard_size)):
            if limit is not None:
                df = df.head(max(0, limit - total))
            if not df.height:
                break
            df = df.with_columns(pl.concat_str([pl.col("title").fill_null(""),
                          pl.col("abstract").fill_null("")], separator="\n").alias("doc"))
            fingerprint = digest([cfg["snapshot"], n, df.height, "title-abstract-v1", shard_size])
            shard = index / f"lexical-{n:05d}-{fingerprint[:12]}"
            marker = shard / "ready.json"
            if not marker.exists() or read_json(marker).get("fingerprint") != fingerprint:
                space_guard(index)
                retriever = bm25s.BM25()
                tokens = bm25s.tokenize(df["doc"].to_list(), stopwords="en",
                                       stemmer=Stemmer.Stemmer("english"), show_progress=False)
                if not tokens.vocab:
                    raise ValueError("No searchable text in lexical shard")
                retriever.index(tokens, show_progress=False)
                retriever.save(str(shard))
                df.select("pmid", "title", "pub_year",
                          pl.col("abstract").is_not_null().alias("has_abstract")).write_parquet(shard / "meta.parquet")
                atomic_json(marker, {"fingerprint": fingerprint, "docs": df.height})
            shards.append(shard.name)
            total += df.height
            print(f"Lexical: {total:,} records", flush=True)
        expected = sum(df.height for df in article_batches())
        atomic_json(index / "lexical.json", {"snapshot": cfg["snapshot"], "shards": shards,
                    "docs": total, "expected": expected, "complete": total == expected,
                    "scoring": "BM25 per-shard statistics; evaluate cross-shard ranking"})


class BM25Search:
    def __init__(self):
        import bm25s, Stemmer
        _, self.index, cfg = paths()
        self.cfg = read_json(self.index / "lexical.json")
        if self.cfg["snapshot"] != cfg["snapshot"] or not self.cfg["complete"]:
            raise RuntimeError("Lexical index is incomplete or from another snapshot")
        self.module = bm25s
        self.stemmer = Stemmer.Stemmer("english")
        # Loading 161 full-corpus shards from disk for every query made repeated
        # searches unnecessarily expensive. Keep the memory-mapped retriever and
        # its compact metadata table after the first use in this process.
        self._loaded = {}

    def close(self):
        """Release NumPy memory maps, especially important for Windows files."""
        loaded = getattr(self, "_loaded", {})
        for retriever, _ in loaded.values():
            for value in getattr(retriever, "scores", {}).values():
                mapping = getattr(value, "_mmap", None)
                if mapping is not None:
                    mapping.close()
        loaded.clear()

    def __del__(self):
        self.close()

    def _load_shard(self, name):
        cached = self._loaded.get(name)
        if cached is None:
            shard = self.index / name
            # Windows prevents deletion/replacement of active memory maps.
            # Load arrays normally there; other platforms keep the lower-RAM
            # memory-mapped path. PUBMED_BM25_MMAP can override either choice.
            mmap_default = os.name != "nt"
            mmap = os.environ.get("PUBMED_BM25_MMAP", "1" if mmap_default else "0") == "1"
            cached = (self.module.BM25.load(str(shard), load_corpus=False, mmap=mmap,
                                            show_progress=False),
                      pl.read_parquet(shard / "meta.parquet"))
            self._loaded[name] = cached
        return cached

    def warm(self):
        """Load every lexical shard before the service reports startup complete."""
        for name in self.cfg["shards"]:
            self._load_shard(name)

    def search(self, query, k=20, since_year=None, until_year=None, allowed_pmids=None):
        out = []
        tokens = self.module.tokenize([query], stopwords="en", stemmer=self.stemmer,
                                     return_ids=False, show_progress=False)[0]
        for name in self.cfg["shards"]:
            r, meta = self._load_shard(name)
            mask = np.ones(meta.height, dtype=bool)
            if since_year is not None:
                mask &= (meta["pub_year"] >= since_year).fill_null(False).to_numpy()
            if until_year is not None:
                mask &= (meta["pub_year"] <= until_year).fill_null(False).to_numpy()
            if allowed_pmids is not None:
                mask &= meta["pmid"].is_in(allowed_pmids).to_numpy()
            scores = r.get_scores(tokens)
            eligible = np.flatnonzero(mask & (scores > 0))
            order = eligible[np.argsort(-scores[eligible], kind="stable")[:k]]
            out.extend({**meta.row(int(i), named=True), "score": float(scores[i])} for i in order)
        return sorted(out, key=lambda row: (-row["score"], row["pmid"]))[:k]


def token_passages(text, tokenizer, max_tokens, overlap=48):
    """Use original text offsets, retaining the final section of long abstracts."""
    if not text:
        return []
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False)["offset_mapping"]
    if max_tokens <= overlap or max_tokens < 1:
        raise ValueError("Token budget must exceed overlap")
    chunks = []
    for start in range(0, len(offsets), max_tokens - overlap):
        end = min(start + max_tokens, len(offsets))
        left = 0 if start == 0 else offsets[start][0]
        right = len(text) if end == len(offsets) else offsets[end][0]
        chunks.append(text[left:right])
        if end == len(offsets):
            break
    return chunks


def lexical_passages(query, row):
    """Return bounded verbatim windows around query terms, including late matches."""
    body = row["abstract"] or row["title"] or ""
    words = list(re.finditer(r"\S+", body))
    terms = set(re.findall(r"\w+", query.lower()))
    candidates = []
    for start in range(0, len(words), 130):
        end = min(start + 160, len(words))
        left = 0 if start == 0 else words[start].start()
        right = len(body) if end == len(words) else words[end].start()
        passage = body[left:right]
        score = len(terms & set(re.findall(r"\w+", passage.lower())))
        candidates.append((score, start, passage))
        if end == len(words):
            break
    candidates.sort(key=lambda p: (-p[0], p[1]))
    return [{"section": "abstract" if row["abstract"] else "title", "text": p[2],
             "chunk_id": digest([row["pmid"], p[1], p[2]])} for p in candidates[:2]]


def _encoder(model_name, backend="torch", threads=None):
    if backend != "torch":
        raise ValueError("Versioned passages currently support the torch backend only")
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(threads or min(4, os.cpu_count() or 4))
    revision = os.environ.get("PUBMED_EMBED_REVISION")
    if revision is None and model_name == "BAAI/bge-small-en-v1.5":
        revision = BGE_SMALL_REVISION
    return SentenceTransformer(model_name, device=os.environ.get("PUBMED_EMBED_DEVICE", "cpu"),
                               revision=revision,
                               local_files_only=os.environ.get("PUBMED_ALLOW_MODEL_DOWNLOAD") != "1")


def build_vectors(model_name, limit=None, batch=32, threads=None, backend="torch", ann=True):
    import lancedb
    _, index, cfg = paths()
    if (limit is not None and limit < 1) or batch < 1:
        raise ValueError("Limits and batches must be positive")
    index.mkdir(parents=True, exist_ok=True)
    with BuildLock(index / "build.lock"):
        model = _encoder(model_name, backend, threads)
        revision = getattr(model[0].auto_model.config, "_commit_hash", None)
        if revision is None:
            raise ValueError("Embedding model must resolve to a pinned hub revision")
        descriptor = {"snapshot": cfg["snapshot"], "model": model_name, "revision": revision,
                      "chunk_version": CHUNK_VERSION, "max_tokens": model.max_seq_length,
                      "batch": batch, "backend": backend, "normalized": True}
        table_name = "passages_" + digest(descriptor)[:24]
        db = lancedb.connect(str(index))
        table = db.open_table(table_name) if table_name in db.table_names() else None
        marker = index / f"{table_name}.json"
        state = read_json(marker) if marker.exists() and table else {"batches": {}, "complete": False}
        processed, expected_chunks = 0, 0
        total_articles = sum(df.height for df in article_batches())
        for number, df in enumerate(article_batches(batch)):
            rows = []
            for article in df.iter_rows(named=True):
                sources = [("title", article["title"] or "")]
                if article["abstract"]:
                    sources.append(("abstract", article["abstract"]))
                for section, source in sources:
                    for part, passage in enumerate(token_passages(source, model.tokenizer, model.max_seq_length - 2)):
                        key = digest([article["pmid"], section, part, passage, descriptor])
                        rows.append({"chunk_id": key, "pmid": article["pmid"], "section": section,
                                     "part": part, "text": passage, "title": article["title"] or "",
                                     "pub_year": article["pub_year"] or 0, "content_hash": digest(source)})
            fingerprint = digest([r["chunk_id"] for r in rows])
            expected_chunks += len(rows)
            if limit is not None and processed >= limit:
                continue
            completed = state["batches"].get(str(number))
            if not completed or completed["fingerprint"] != fingerprint:
                space_guard(index)
                for start in range(0, len(rows), batch):
                    group = rows[start:start + batch]
                    vectors = model.encode([r["text"] for r in group], batch_size=batch,
                                          normalize_embeddings=True, show_progress_bar=False).tolist()
                    encoded = [{**r, "vector": v} for r, v in zip(group, vectors)]
                    if table is None:
                        table = db.create_table(table_name, encoded)
                    else:
                        table.merge_insert("chunk_id").when_matched_update_all().when_not_matched_insert_all().execute(encoded)
                state["batches"][str(number)] = {"fingerprint": fingerprint, "chunks": len(rows), "articles": df.height}
                atomic_json(marker, state)
            processed += df.height
            print(f"Vectors: {processed:,}/{total_articles:,} articles", flush=True)
        count = table.count_rows() if table else 0
        state.update({**descriptor, "table": table_name, "docs": count, "expected_chunks": expected_chunks,
                      "articles": processed, "expected_articles": total_articles,
                      "complete": processed == total_articles and count == expected_chunks, "ann": False})
        if state["complete"] and ann and count >= 1024:
            table.create_index(metric="cosine", index_type="IVF_FLAT",
                               num_partitions=max(1, int(math.sqrt(count))), replace=True)
            table.create_scalar_index("pub_year", replace=True)
            state["ann"] = True
        atomic_json(marker, state)
        atomic_json(index / "vector_model.json", {k: v for k, v in state.items() if k != "batches"})


class HybridSearch:
    def __init__(self, rerank=None):
        self.bm25 = BM25Search()
        self.dense = None
        self.vector_status = "absent"
        self.dense_error = None
        self.reranker = None
        # A verified ANN index is the normal production path. Set this to 0 for
        # exhaustive audit queries or ANN recall evaluation.
        self.ann_requested = os.environ.get("PUBMED_USE_ANN", "1") != "0"
        self.use_ann = False
        self.store, self.index, self.dataset = paths()
        path = self.index / "vector_model.json"
        if path.exists():
            cfg = read_json(path)
            self.vector_status = "incomplete"
            if cfg.get("complete") and cfg.get("snapshot") == self.dataset["snapshot"]:
                import lancedb
                self.tbl = lancedb.connect(str(self.index)).open_table(cfg["table"])
                self.use_ann = self.ann_requested and bool(cfg.get("ann"))
                # A full Lance row count can add more than a minute to cold
                # startup at this corpus size. Finalization and release checks
                # verify it once; operators can request the expensive startup
                # check explicitly when diagnosing an installation.
                if (os.environ.get("PUBMED_VERIFY_VECTOR_COUNT") == "1" and
                        self.tbl.count_rows() != cfg["expected_chunks"]):
                    raise RuntimeError("Vector count does not match committed manifest")
                try:
                    self.model = _encoder(cfg["model"], cfg["backend"])
                    if getattr(self.model[0].auto_model.config, "_commit_hash", None) != cfg["revision"]:
                        raise RuntimeError("Query encoder revision differs from indexed model")
                    self.dense = cfg["model"]
                    self.vector_status = "complete"
                except (OSError, RuntimeError, ValueError) as error:
                    self.vector_status = "unavailable"
                    self.dense_error = str(error)
        rerank = rerank or os.environ.get("PUBMED_RERANKER")
        if rerank:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(rerank, device="cpu")

    def close(self):
        bm25 = getattr(self, "bm25", None)
        if bm25 is not None:
            bm25.close()

    def __del__(self):
        self.close()

    def _external_passages(self, hits):
        """Read source text only for returned vector hits, grouped by part file."""
        grouped = {}
        for hit in hits:
            grouped.setdefault(hit["passage_file"], []).append(hit["chunk_id"])
        found = {}
        for relative, ids in grouped.items():
            source = (self.index / relative).resolve()
            if not source.is_relative_to(self.index.resolve()) or not source.exists():
                raise RuntimeError("Vector index references an invalid passage file")
            rows = (pl.scan_parquet(source).filter(pl.col("chunk_id").is_in(ids))
                    .select("chunk_id", "section", "text").collect())
            found.update({row["chunk_id"]: row for row in rows.iter_rows(named=True)})
        return found

    def search(self, query, k=10, since_year=None, rrf_k=60, until_year=None, allowed_pmids=None):
        if not isinstance(k, int) or not 1 <= k <= 50:
            raise ValueError("k must be between 1 and 50")
        if not query.strip() or len(query) > 4000:
            raise ValueError("Query must contain 1 to 4000 characters")
        if since_year is not None and until_year is not None and since_year > until_year:
            raise ValueError("Invalid year range")
        pools = {"bm25": self.bm25.search(query, k * 5, since_year, until_year, allowed_pmids)}
        passages = {}
        if self.dense:
            query_text = "Represent this sentence for searching relevant passages: " + query if "bge-" in self.dense else query
            vector = self.model.encode([query_text], normalize_embeddings=True)[0].tolist()
            search = self.tbl.search(vector).distance_type("cosine")
            if self.use_ann:
                search = search.nprobes(int(os.environ.get("PUBMED_ANN_NPROBES", "64")))
                search = search.refine_factor(int(os.environ.get("PUBMED_ANN_REFINE", "10")))
            else:
                search = search.bypass_vector_index()
            predicates = []
            if since_year is not None:
                predicates.append(f"pub_year >= {int(since_year)}")
            if until_year is not None:
                predicates.append(f"pub_year > 0 AND pub_year <= {int(until_year)}")
            if allowed_pmids is not None:
                if any(not str(p).isdigit() for p in allowed_pmids):
                    raise ValueError("Invalid PMID filter")
                predicates.append("pmid IN (" + ",".join("'" + str(p) + "'" for p in allowed_pmids) + ")" if allowed_pmids else "pub_year < 0")
            if predicates:
                search = search.where(" AND ".join(predicates), prefilter=True)
            hits = search.limit(k * 10).to_list()
            external = self._external_passages(hits) if hits and "passage_file" in hits[0] else None
            unique = {}
            for hit in hits:
                unique.setdefault(hit["pmid"], hit)
                evidence = external.get(hit["chunk_id"]) if external is not None else {
                    key: hit[key] for key in ("chunk_id", "section", "text")}
                if evidence:
                    passages.setdefault(hit["pmid"], []).append(evidence)
            pools["vector"] = list(unique.values())
        fused = {}
        for hits in pools.values():
            for rank, hit in enumerate(hits):
                fused[hit["pmid"]] = fused.get(hit["pmid"], 0) + 1 / (rrf_k + rank + 1)
        order = sorted(fused, key=lambda p: (-fused[p], p))[:k * 5]
        records = pl.scan_parquet(self.store / "articles.parquet").filter(pl.col("pmid").is_in(order)).collect().to_dicts()
        results = []
        for row in records:
            evidence = passages.get(row["pmid"], [])[:2] or lexical_passages(query, row)
            results.append({"pmid": row["pmid"], "title": row["title"], "pub_year": row["pub_year"],
                "score": fused[row["pmid"]], "has_abstract": bool(row["abstract"]), "evidence": evidence,
                "source_member": row.get("source_member"), "url": f"https://pubmed.ncbi.nlm.nih.gov/{row['pmid']}/"})
        results.sort(key=lambda r: (-r["score"], r["pmid"]))
        if self.reranker and results:
            query_tokens = len(self.reranker.tokenizer(query, add_special_tokens=False)["input_ids"])
            budget = (self.reranker.max_length or 512) - query_tokens - 3
            if budget <= 48:
                raise ValueError("Query is too long for the selected reranker")
            pairs, owners = [], []
            for n, row in enumerate(results):
                for evidence in row["evidence"]:
                    for part in token_passages(evidence["text"], self.reranker.tokenizer, budget):
                        pairs.append((query, part))
                        owners.append(n)
                row["rerank_score"] = float("-inf")
            for n, score in zip(owners, self.reranker.predict(pairs)):
                results[n]["rerank_score"] = max(results[n]["rerank_score"], float(score))
            results.sort(key=lambda r: -r["rerank_score"])
        return {"retrievers": list(pools), "dense_model": self.dense, "vector_status": self.vector_status,
                "dense_error": self.dense_error,
                "snapshot": self.dataset["snapshot"], "filters": {"since_year": since_year, "until_year": until_year},
                "ann": self.use_ann, "exact": bool(self.dense and not self.use_ann), "results": results[:k],
                "caveat": "Ranked evidence from selected files, not a corpus total. Title-only records provide no abstract findings."}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bm25", "vectors", "search"])
    ap.add_argument("--dataset")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--query", default="cisplatin resistance")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--backend", default="torch")
    a = ap.parse_args()
    if a.dataset:
        os.environ["PUBMED_DATASET"] = a.dataset
    if a.cmd == "bm25":
        build_bm25(a.limit)
    elif a.cmd == "vectors":
        build_vectors(a.model, a.limit, a.batch, a.threads, a.backend)
    else:
        print(json.dumps(HybridSearch().search(a.query), indent=2))
