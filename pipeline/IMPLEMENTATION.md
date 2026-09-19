# New-archive implementation

The active dataset is selected by `PUBMED_DATASET`, defaulting to
`covid-files/dataset.json`. The branch name is not a topic filter. The selected
ZIP supplies the article records; the optional NLM MeSH file supplies terminology.
No SQL database is used. A missing dataset fails instead of loading old artifacts.

## Build and run

From the repository root in PowerShell:

```powershell
# Three source files spread across the archive, including updates/deletions.
python pipeline/stream_store.py --src 'C:\Users\User\Downloads\pubmed25n1509 (1).zip' --root covid-files --max-files 3 --sample --buckets 32 --mesh raw/desc2026.xml
python pipeline/manage.py bm25
python pipeline/manage.py serve
```

Open http://127.0.0.1:8010. `start_all.ps1` serves the selected snapshot in the
foreground; stop it with Ctrl+C. It does not start an expensive embedding job
or stop unrelated processes. `progress.ps1` reports the selected manifest.
Set `--dataset <path>` on management commands to select another saved snapshot.

For a full import, omit `--max-files` and `--sample`. The selected 159-member
archive completed as snapshot `6f269c0288fb98ea0c96`. Source files are streamed
without extraction. Event batches are capped by record count or 16 MiB of JSON,
then reconciled in PMID buckets. Memory for reconciliation scales with one
bucket, not all article bodies; increase `--buckets` for larger inputs. Global
vocabulary/dimension aggregation and the vocabulary lookup still require memory
proportional to distinct terms/names. This is a laptop implementation, not a
claim of fixed memory at unlimited scale.

The 2 GB free-space reserve pauses writes when crossed. It is not a forecast of
the space required for the next full build. Compressed staging, reconciled
partitions and final tables are retained until verification. For this completed
snapshot, the 8.16 GB full-build staging folder was removed only after the store,
embeddings and counts passed verification. The original ZIP and Kaggle output
remain recovery sources. No personal-file cleanup is performed automatically.

Each complete source member has an atomic checkpoint. CRC/XML failures do not
commit that member. Resume repeats an unfinished member, safely upserting its
events. Source order, source ordinal, the full revision date and content hash
are retained. The latest event owns all child collections, so a removed author,
keyword or citation cannot survive a revision. Deletions and later reintroductions
are reconciled in source order. Unsupported book records fail explicitly.

## Embeddings and search

For a resumable free Google Colab alternative, use
[`colab/pubmed_embedding.ipynb`](../colab/pubmed_embedding.ipynb) and follow
[`colab/README.md`](../colab/README.md). That workflow writes checksummed vector
parts, survives normal runtime loss, and finalizes only after the worker covers
the complete article table exactly once. The CLI still supports multiple workers
in environments where parallel workers are permitted.

```powershell
# Use cached weights without network access when already installed.
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
python pipeline/retrieval.py vectors --dataset covid-files/dataset.json --threads 4 --batch 32
```

The default model is BGE-small. Titles and entire abstracts are split by the
model tokenizer's offsets, with overlap and source text retained. Long abstracts
keep their concluding passages. Model revision, source snapshot, text hashes,
chunk policy and batch settings identify each vector generation. A limited
build rounds to an article batch and stays incomplete. Idempotent chunk upserts
recover from a crash between writing vectors and checkpointing. Changed corpus
snapshots get separate generations: superseded/deleted chunks cannot be served
by the new generation. Unchanged vectors across different snapshots are not yet
reused, so budget for re-embedding a new full snapshot.

The keyword index covers title-only records as well as abstracts. The full build
contains 161 resumable shards. The long-lived search process caches loaded
shards; Windows uses normal arrays to avoid locked memory-map files, while other
platforms default to memory maps. BM25 statistics are per shard;
cross-shard relevance needs measurement. Date/concept filters are applied before
candidate selection. Search is a ranked sample; typed tools compute corpus counts.
Controlled terms also match equivalent author keywords, with the scope disclosed.

Incomplete or mismatched vectors cannot silently join hybrid search. The full
snapshot uses a compressed IVF_PQ index with 2,508 partitions, 48 sub-vectors,
64 search probes and refinement factor 10. It matched exact dense top-10 on all
seven meaningful smoke queries while reducing vector search from about 18.5
seconds to about 0.1 seconds. ANN is used automatically when its verified marker
exists; set `PUBMED_USE_ANN=0` for an exhaustive audit. Override tuning with
`PUBMED_ANN_NPROBES` and `PUBMED_ANN_REFINE`. `PUBMED_RERANKER=<model>` enables an optional cross-encoder experiment;
its memory, latency and relevance must be evaluated before adoption. Reranker
inputs are split to fit its tokenizer budget too. Cached embedding weights are
used by default; set `PUBMED_ALLOW_MODEL_DOWNLOAD=1` when deliberately downloading
a new model. An unavailable query encoder is disclosed and keyword search remains available.

## Answer contract

Simple PMID/statistics requests skip generation. Other requests use bounded named
tools. Totals and rows are rendered from returned fields, including valid zeros.
Evidence answers currently use verbatim abstract excerpts with verified PMIDs;
free-form biomedical claims are not certified by a number-matching heuristic.
Title-only results cannot supply findings. Invalid citations/quotes cause a retry
or an explicit insufficient-evidence response. Comparison evidence is collected
for both sides. This verifies source attribution, not clinical relevance or entailment.

Request state is isolated per question and carries snapshot, filters, results,
passages and tool history. Duplicate actions stop rather than loop. JSON payloads
drop whole rows with an omission count. Tool/model steps, response tokens, context
size and elapsed time have limits. Token and time budgets are checked between
calls; a running local tool is not forcibly killed. Provider usage is reported
when available; monetary cost needs the selected provider's prices.

Local Ollama is the default. Cloud calls require all of `PUBMED_LLM=cloud`,
`PUBMED_ALLOW_CLOUD=1`, `PUBMED_API_BASE`, `PUBMED_API_KEY` and
`PUBMED_CLOUD_MODEL`. Launchers no longer silently select cloud from a saved key.
No cloud provider was called during implementation validation.

## Validation and release

```powershell
python -m unittest discover -s tests -v
python tests/service_smoke.py research/full-service-integration.json
python tests/ann_smoke.py research/ann-integration.json
python pipeline/evaluate.py --dataset covid-files/dataset.json --cases evaluation/smoke.jsonl --output research/full-retrieval-evaluation.json
python pipeline/evaluate.py --dataset covid-files/dataset.json --cases evaluation/smoke.jsonl --output research/full-ann-benchmark.json --ann
python pipeline/manage.py release --output covid-files/release.json
```

The evaluation harness accepts reviewed `id`, `query`, `relevant_pmids`, optional
`filters`, and optional `tool`/`args`/`expected` cases. Run identical held-out cases
with `--answers fixed` and `--answers adaptive`; compare reranking separately via
`--reranker`. `--ann` compares ANN and exact vectors. Sequential p50/p95 and token
usage do not establish performance at the intended concurrency. The example case
is a format example, not a labelled accuracy benchmark. Lead-provided questions,
gold answers and biomedical review are still needed.

Release preparation hashes only the selected complete artifacts without copying
gigabytes. The current inventory contains 2,440 files totaling 18.66 GB.
Publication requires explicit separate destinations:

```powershell
python deploy/upload.py --dataset covid-files/dataset.json --data-repo OWNER/NEW-DATASET
python deploy/upload.py --space-repo OWNER/NEW-SPACE
```

Repositories default to private when newly created. Existing repositories retain
their existing visibility. Configure the resulting immutable dataset commit as
`PUBMED_DATA_REVISION` and its repository as `PUBMED_DATA_REPO`. Container startup
verifies every release file before activating the dataset. Keep the previous
manifest/commit for rollback and restart the server after selecting it. The
GitHub workflow is manual and requires a destination; it no longer publishes
automatically to the earlier Space. No remote publication has been performed.
