# Search and overview performance

Measured on the development laptop, using snapshot `6f269c0288fb98ea0c96`,
whole collection, inclusive years 2020–2024. These are Python calculation
timings, not browser latency or measurements on the remote deployment PC.
The before/after JSON files record the initial panel calculations. Repeated
requests share a bounded, five-minute response cache.

The initial overview calculations remained about seven seconds in total;
the optimization does not claim to accelerate uncached citation aggregation.
The UI now starts overview independently of ranked search and displays completed
panels without waiting for citation results. Cached calculations each measured
under 1 ms. Verified totals remained 2,335,190 dated papers, 76 study types,
and 32,308 registry links. The cited-paper list retained 20 results.

Search now hydrates card metadata in one article scan instead of two and reads
vector passage text only for selected papers or reranking candidates. ANN
probes/refinement, candidate pools, scope filters, fusion and reranking settings
remain unchanged. Synthetic retrieval tests verify title-only results, year
filters, metadata and bounded passage reads.

Full-dataset search timing could not be measured: the initial baseline attempt
failed with a memory allocation error, and the development laptop had only
about 30 MB free on C:. No search speedup percentage is established. Free disk
space before attempting another full search benchmark; no user files were
removed to work around this limitation.

To reproduce panel timings on another installation, update code and run:

```powershell
python ops/benchmark_views.py --dataset covid-files/dataset.json --output ops/logs/overview-performance.json
```

The benchmark does not read `llm.env`, call a language model, rebuild indexes,
or load the full search engine. A fresh process measures the first calculation;
the operating system's file cache may already be warm.

## Citation summaries and topic membership reuse

An optional citation summary was built locally from all 64,150,985 reference
rows. It occupies 73,859,214 bytes (about 70 MiB) under the selected index's
`analytics` directory. Source tables, the dataset manifest and vector embeddings
were not changed. The summary contains 14,611,624 cited identifiers, including
references to papers outside the selected article collection; query-time scope
joins continue to exclude those outside papers.

`overview-stage2-covid.json` and `overview-stage2-whole.json` compare raw and
optimized calculations using the same code and deterministic tie breaks. Raw
mode disables both topic membership reuse and the citation summary. The five-
minute response cache is not the source of the initial-request improvements.
Both modes returned equal complete panel responses. Runs used warm OS file
caches and overlapped other verification work; they are indicative local
measurements, not latency guarantees or measurements on the deployment PC.

For COVID-19 with inclusive publication years 2020–2024, measured raw versus
optimized server calculation times were:

| Panel | Raw | Optimized |
|---|---:|---:|
| Publication trend | 1.99 s | 0.74 s |
| Study types | 0.88 s | 0.23 s |
| Most cited | 2.90 s | 1.12 s |
| Registry links | 1.70 s | 0.58 s |

The whole-collection most-cited panel went from 2.74 s to 1.15 s. Topic reuse
helps topic-scoped requests; whole-collection requests have no membership list
to reuse. Retrieving the cached COVID-19 membership list took 0.173 ms and
retained 982,240 bytes for 122,780 PMIDs in an earlier paired run. Its membership
is independent of year bounds; 109,655 of those papers fall in 2020–2024.

To enable the citation optimization after updating an existing installation:

```powershell
python pipeline/manage.py analytics --dataset covid-files/dataset.json
```

The builder verifies the aggregate count and publishes an atomic manifest only
after successful completion. First use checks snapshot/source metadata, schema
and summary checksum. Missing, stale or damaged summaries fall back to raw
calculations. Valid summaries are included in future release inventories. No
Hugging Face release or remote installation was changed during this work.

## Remaining improvements: indexed records, lexical shortcuts and query plans

All three remaining code changes are implemented locally. The optional article
lookup covers 3,217,739 records and uses a 73,519,104-byte SQLite PMID-to-group map
plus a 2,063,144,868-byte Parquet file with 1,000-record groups. Together with the
citation summary, these derived artifacts add about 2.2 GB to an installation.
The first larger lookup attempt hit the disk reserve and was discarded; the
compact build completed and a second preparation reused both valid indexes.
No source article tables, dataset selection, lexical scores or embeddings changed.

`article-lookup-performance.json` selects 20 records distributed over original
row groups. Complete projected records match the source scan. With warm OS
caches, indexed hydration measured 75.577 ms versus 98.615 ms for a source scan.
The initial artifact checksum/schema verification took 2.135 seconds separately.
The server performs this optional verification at startup, and retains only
two Parquet footers, not open files or article text. Lookup batches decode at
most eight groups before filtering out unrelated rows and creating Python objects.
For search without reranking, only the returned papers are hydrated; all reranking
candidates are still hydrated when a reranker is selected. Missing candidate
records retain the original behavior of filling from later candidates.

BM25 now filters compact shard metadata before loading/scoring score arrays,
skips shards with no eligible records, and uses partial top-k selection. Boundary
ties keep the exact original stable row order; the global merge keeps its PMID
tie break. Total positive-score matches are counted before top-k truncation.
The targeted tests compare against full sorting, cover empty scopes and inclusive
years, and verify that ineligible shards are never scored. Full end-to-end search
latency is still not established; these shortcuts are not a measured ANN speedup.

The actual overview plans are saved in `research/query-plans/`. Inspection
confirmed article scans project only 2 of 18 columns for scope/trend queries,
year predicates are applied in the article scans before topic joins, and the
most-cited sort is bounded to the requested top 20. Trial queries now carry
only PMID/year and registry fields through their joins; titles are loaded after
selecting returned rows. Exact registry totals are still calculated before limiting.

`overview-stage3-covid.json` compares raw and optimized calculations for COVID-19
in 2020–2024, with artifact verification separately recorded as startup work.
Complete responses match in all four panels:

| Panel | Raw | Optimized |
|---|---:|---:|
| Publication trend | 1.63 s | 0.89 s |
| Study types | 1.14 s | 0.20 s |
| Most cited | 2.77 s | 0.53 s |
| Registry links | 0.93 s | 0.15 s |

Measurements exclude browser/network time and may use warm OS caches. Repeat
response-cache hits were below 1 ms; this does not imply browser latency below 1 ms.

`manage.py performance` prepares both optional indexes. The Windows supervisor
runs it before starting its owned server, reuses existing valid artifacts, and
continues with source-file fallback if preparation fails. The task helper is
`ops/enable_local_sync.ps1`; it was syntax-checked, not installed on this laptop
or verified on the remote PC. Commit/push, task activation/restart and live
deployment verification remain separate. No Hugging Face dataset release was
uploaded during this work. Valid derived artifacts are included when a future
release inventory is generated.
