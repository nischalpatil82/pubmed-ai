# Implementation validation — 15 September 2026

Code and data changes are local on `covid-files`. No remote publication or
cloud model call was performed. The input remains the replacement ZIP only;
MeSH is a reference glossary, not an additional article collection.

## Verified full-corpus result

- All 159 XML members were parsed and reconciled into immutable snapshot
  `6f269c0288fb98ea0c96`.
- The store contains 3,217,739 articles, including 2,835,880 with abstracts and
  381,859 without abstracts. It also contains 14,098 journals, 7,600,560
  authors, 21,231,442 authorship rows, 18,582,442 MeSH rows, 3,224,288 substance
  rows, 11,642,048 keyword rows, 64,150,985 citation rows, 4,787,323 publication
  type rows, 2,238,560 grant rows, 85,005 databank links and 13,246 current
  deletion tombstones.
- The article Parquet SHA-256 is
  `65787d98506a65162a218727f3305a38867107e597c297f76cb672f867fcc9c6`.
- The BM25 manifest has 161 ready shards whose physical marker counts sum to
  exactly 3,217,739 documents.
- Kaggle produced 322 checksummed embedding parts with 6,290,649 passages from
  all 3,217,739 articles. The local LanceDB table physically contains exactly
  6,290,649 rows and uses pinned model `BAAI/bge-small-en-v1.5` revision
  `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`.
- Vector parts were deleted only after checked LanceDB import. Passage Parquets,
  the Lance table, per-part manifests, the original source ZIP and Kaggle output
  remain as recovery material. The verified 8.16 GB full-build staging folder
  was removed after final reconciliation.
- A compressed cosine IVF_PQ index covers all 6,290,649 vector rows. It has
  2,508 partitions, 48 sub-vectors and 8-bit codes; the physical vector index is
  about 339 MB and the publication-year B-tree is about 31 MB.

## Search and service measurements

- Exact dense top-10 search took 18.2–19.0 seconds per query on this laptop.
- IVF_PQ with 64 probes and refinement factor 10 matched exact dense top-10 on
  all seven meaningful smoke queries. ANN vector search p50 was about 101 ms and
  p95 about 108 ms. This measures agreement with exact BGE retrieval, not
  biomedical relevance.
- The first BM25 shard load took about 21 seconds and a repeated BM25 query took
  about 0.24 seconds. The service now loads the BM25 cache during startup.
- Production-style warm hybrid searches took 2.1–3.9 seconds across the sampled
  queries. The eight-query retrieval run had p50 2.80 seconds. Its cold-inclusive
  p95 was 22.43 seconds because the first query loaded every BM25 shard; the
  service now performs that load before reporting ready. A broad year-filtered
  `diabetes` API smoke search took 6.7 seconds in a separate run.
  Service startup remains slow because local BGE model loading took about
  56–57 seconds in fresh Python processes.
- The full API smoke passed health, snapshot and count checks, a filtered hybrid
  ANN search, invalid-limit rejection and a direct count answer. The direct
  answer made zero model calls and the second cached count completed in under
  one measured millisecond.
- All 20 regression tests passed after the full-build and retrieval changes.
  They cover source parsing/reconciliation, bounded finalization, filtering,
  passage preservation, resumable vectors, evidence checks and release safety.
- `covid-files/release.json` inventories 2,440 files totaling 18.66 GB with
  SHA-256 and byte size. Nothing was uploaded.

Machine-readable evidence is in `full-retrieval-smoke.json`,
`full-retrieval-smoke-ann.json`, `full-ann-benchmark.json`,
`full-ann-tuning.json` and `full-service-integration.json` in this directory.
The complete store, checkpoints, indexes and release inventory are under the
ignored `covid-files/` directory.

## Still unverified or deferred

Real lead-provided questions, reviewed relevant PMIDs and acceptable answers are
still needed for held-out quality evaluation. The seven-query ANN comparison is
too small to establish production recall, biomedical relevance or answer
correctness. Cross-shard BM25 score comparability, intended concurrency,
reranker benefit and sustained-load latency still need measurement.

The local Ollama answer model has not been accepted for final generation quality
or speed. Cloud policy, approved provider, target hardware and provider costs are
undecided. No cloud data was sent.

The separate artifact repository and application deployment destination remain
unnamed, so publication and live deployment have not occurred. Clinical-trial
ingestion still needs its own supplied source files and schema. Fine-tuning
remains deferred until held-out evaluation identifies a repeated behavior error
that retrieval, routing or evidence handling cannot fix.
