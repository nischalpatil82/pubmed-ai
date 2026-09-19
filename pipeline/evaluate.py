"""Held-out retrieval/answer evaluation and ANN-versus-exact benchmarking.

JSONL cases: id, query, relevant_pmids, optional filters, tool/args/expected,
expected_refusal. Human-reviewed relevance and biomedical support remain required.
"""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from dataset import atomic_json


def evaluate(cases, answer_mode=None, rerank=None):
    from retrieval import HybridSearch
    engine = HybridSearch(rerank=rerank)
    rows = []
    for case in cases:
        started = time.perf_counter()
        result = engine.search(case["query"], k=10, **case.get("filters", {}))
        pmids = [r["pmid"] for r in result["results"]]
        relevant = set(str(p) for p in case.get("relevant_pmids", []))
        row = {"id": case["id"], "retrieval_ms": (time.perf_counter() - started) * 1000,
               "pmids": pmids, "recall_at_10": len(set(pmids) & relevant) / len(relevant) if relevant else None,
               "reciprocal_rank": next((1 / (i + 1) for i, p in enumerate(pmids) if p in relevant), 0) if relevant else None}
        if case.get("tool"):
            import tools
            actual = tools.call(case["tool"], case.get("args", {}))
            row["typed_fields_correct"] = all(actual.get(k) == v for k, v in case["expected"].items())
        if answer_mode:
            import agent
            began = time.perf_counter()
            answer = agent.run(case["query"], adaptive=answer_mode == "adaptive", filters=case.get("filters"))
            row.update({"answer_ms": (time.perf_counter() - began) * 1000,
                        "answer": answer["answer"], "tool_calls": len(answer["calls"]),
                        "usage": answer.get("usage", {}),
                        "refusal_correct": answer["refused"] == case.get("expected_refusal", False),
                        "human_support_review": "pending"})
        rows.append(row)
    latencies = [row["retrieval_ms"] for row in rows]
    recalls = [row["recall_at_10"] for row in rows if row["recall_at_10"] is not None]
    return {"snapshot": engine.dataset["snapshot"], "mode": answer_mode or "retrieval",
            "reranker": rerank, "queries": len(rows), "rows": rows,
            "mean_recall_at_10": float(np.mean(recalls)) if recalls else None,
            "p50_ms": float(np.percentile(latencies, 50)) if rows else None,
            "p95_ms": float(np.percentile(latencies, 95)) if rows else None,
            "limitations": "Single-process sequential timing. Not a concurrency, clinical-accuracy or provider-cost certification."}


def ann_benchmark(queries, k=10):
    from retrieval import HybridSearch
    engine = HybridSearch()
    if not engine.dense:
        raise ValueError("A complete vector index is required")
    nprobes = int(os.environ.get("PUBMED_ANN_NPROBES", "64"))
    refine = int(os.environ.get("PUBMED_ANN_REFINE", "10"))
    rows = []
    for query in queries:
        text = "Represent this sentence for searching relevant passages: " + query if "bge-" in engine.dense else query
        vector = engine.model.encode([text], normalize_embeddings=True)[0].tolist()
        start = time.perf_counter()
        exact = engine.tbl.search(vector).distance_type("cosine").bypass_vector_index().limit(k).to_list()
        exact_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        approximate = (engine.tbl.search(vector).distance_type("cosine")
                       .nprobes(nprobes).refine_factor(refine).limit(k).to_list())
        gold = {r["chunk_id"] for r in exact}
        rows.append({"query": query, "recall": len(gold & {r["chunk_id"] for r in approximate}) / len(gold) if gold else None,
                     "exact_ms": exact_ms, "ann_ms": (time.perf_counter() - start) * 1000})
    recalls = [row["recall"] for row in rows if row["recall"] is not None]
    return {"snapshot": engine.dataset["snapshot"], "nprobes": nprobes,
            "refine_factor": refine, "mean_recall_at_10": float(np.mean(recalls)) if recalls else None,
            "rows": rows,
            "limitations": "Single-process query sample; expand with lead-reviewed biomedical questions before release acceptance."}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--answers", choices=["fixed", "adaptive"])
    ap.add_argument("--reranker")
    ap.add_argument("--ann", action="store_true")
    a = ap.parse_args()
    os.environ["PUBMED_DATASET"] = str(Path(a.dataset).resolve())
    cases = [json.loads(line) for line in Path(a.cases).read_text(encoding="utf-8").splitlines() if line.strip()]
    atomic_json(a.output, ann_benchmark([c["query"] for c in cases]) if a.ann else evaluate(cases, a.answers, a.reranker))
