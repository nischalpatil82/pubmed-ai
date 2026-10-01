# PubMed Literature Intelligence

New to the project? Read the [complete beginner walkthrough](PROJECT_WALKTHROUGH.md)
for the architecture, data flow, every production file/function, deployment,
tests, limitations and a ready-to-use demonstration script.

Build and launch instructions: [pipeline/IMPLEMENTATION.md](pipeline/IMPLEMENTATION.md).
Completed Kaggle embedding workflow: [kaggle/README.md](kaggle/README.md).
Windows deployment and automatic code updates: [ops/README.md](ops/README.md).

For the new `covid-files` dataset, see [the current project plan](PROJECT_PLAN.md)
for agreed requirements, completed work and remaining steps.

The recorded full snapshot contains 3,217,739 biomedical article records from
159 selected XML files. The active manifest and `/api/stats` identify the selected
snapshot; these records are **not complete PubMed coverage or full-text articles**.
Structured analytics aggregate records within the supported scope, while evidence
search returns a ranked sample of source passages.

Counts are computed from structured records, but ranking panels display limited
rows, not every entity. Counts are not universally clickable drill-downs. Exact
aggregation does not imply complete source coverage or accurate entity identity.

---

## Folder layout

| Path | What | Ships to production? |
|---|---|---|
| `pipeline/` | all the code | yes |
| `covid-files/dataset.json` | selects the active immutable snapshot | yes |
| `covid-files/stores/<snapshot>/` | 12 Parquet fact tables | yes |
| `covid-files/indexes/<snapshot>/` | 161 BM25 shards, 6,290,649 passage vectors, IVF_PQ and evidence passages | yes |
| `C:/Users/User/Downloads/pubmed25n1509 (1).zip` | selected 159-member source archive | no |
| `logs/` | runtime logs | no |
| `llm.env` | your API key | **never** |

The checksummed release inventory contains 2,440 files totaling 18.66 GB. Copy
the inventoried store and index files to run the same snapshot elsewhere without
embedding again.

## Hugging Face deployment

The dataset is intentionally ignored by Git. Publish the verified release to a
separate Hugging Face **dataset** repository first, then deploy the application
to a Docker Space:

```powershell
$env:HF_TOKEN = "your-write-token"
python deploy/upload.py --dataset covid-files/dataset.json --data-repo owner/dataset-name
python deploy/upload.py --space-repo owner/space-name
```

The first command prints the immutable dataset commit hash. Configure the Space
with `PUBMED_DATA_REPO=owner/dataset-name` and
`PUBMED_DATA_REVISION=<that-exact-40-character-commit-hash>`. If the dataset is
private, add `HF_TOKEN` to the Space as a **secret** so it can download the
release. The Space downloads and checks every release file before it starts; it
does not create embeddings. The selected hybrid release is 18.66 GB, so use a
Space storage allocation comfortably above that size and expect cold starts to
depend on the download cache and network speed.

## Run it

```powershell
powershell -ExecutionPolicy Bypass -File start_all.ps1
```

Then open <http://127.0.0.1:8010>. A historical local run took about 80 seconds
to load the BGE model and warm BM25 shards. This is a measurement, not a startup
guarantee; hardware, cache state, selected snapshot and configuration affect time.
The launcher serves only; it does not start embedding jobs.

```powershell
powershell -ExecutionPolicy Bypass -File progress.ps1   # what is running
python view.py                                          # list selected store tables
python view.py journals 20
python view.py articles 3 --wide
python view.py journals 20 --dataset another-dataset/dataset.json
python view.py journals --dataset another-dataset/dataset.json --csv
```

The viewer uses `pipeline.dataset.paths()`: explicit `--dataset` overrides
`PUBMED_DATASET`; without either, explicit `PUBMED_STORE`/`PUBMED_INDEX` are
honoured, otherwise `covid-files/dataset.json` selects the store. A missing or
unready manifest does not silently fall back to legacy `store/`. Positional
`table` and `rows` remain supported with `--wide` and `--csv`. CSV exports the
**entire table**, regardless of preview rows, to `<selected-store-parent>/<table>.csv`
(overwriting that export if it exists); large tables can exceed spreadsheet limits.

## The two lanes

Counting questions and open questions need different machinery, and conflating
them is the usual reason "AI over documents" projects disappoint.

```
                   your question
                         |
              +----------+----------+
              |                     |
        counting?               open-ended?
              |                     |
     parquet aggregation      BM25 + vectors
     exact, whole corpus      ranked sample
              |                     |
              +----------+----------+
                         |
                    the answer
```

The UI has five tabs, with the overview first:

- **Research overview**: the exact keyword-match count for the submitted query,
  publication trend, study types, in-collection citations and registry links.
- **Evidence**: ranked papers with source passages, record details and CSV export.
- **Entities**: limited rankings for journals, named substances, researchers,
  countries and institutions, with structured counts and caveats.
- **Compare**: two recognised concepts' publication counts and overlap, not
  clinical effectiveness.
- **Ask**: a dedicated research-question field, visible retrieval/generation
  progress, and opt-in typed count/list answers or verified extractive evidence;
  insufficient evidence or provider limits can produce a refusal.

The matching-paper metric counts every positive BM25 keyword match after the
selected concept and year filters; the same value appears beside the search filters
and in Research overview. Other analytics use the selected concept and supported
year filters. Without a concept selection, those panels cover the collection within
supported year filters. Panel errors (including HTTP 200 `{error}` responses) are
displayed separately from empty results; other successful panels remain visible.

Count/list tool outputs are rendered deterministically rather than rewritten as
model-generated numerical prose. Findings are validated as verbatim abstract
excerpts tied to retrieved PMIDs. This is not a guarantee of biomedical correctness,
clinical effectiveness, or comprehensive retrieval.

**Scope enforcement:** analytics, comparison and Ask accept inclusive `since_year`
and `until_year` bounds. Ask also receives the selected `concept_id` and refuses
tool paths that cannot represent required filters rather than silently dropping them.
Years must be integers from 1000 to 3000, with the start no later than the end.
The UI warns if returned Ask `filters` do not confirm the requested scope.
Changing or clearing the committed search/concept invalidates cached Ask answers
and prevents older replies from restoring them; edit the search fields and submit
Search evidence to commit a new question/year scope.

`/health` returns 503 until the loaded lexical engine has warmed successfully and
its complete index manifest matches the selected snapshot. Lexical-only service
returns 200 with `degraded: true`; `dense_ready`, `retrievers`, `search_error` and
`vector_error` distinguish hybrid readiness from partial service. Health checks do
not load models or perform searches.

## Rebuilding from another XML snapshot

```powershell
python pipeline/stream_store.py --src 'C:\path\to\new.zip' --root another-dataset --buckets 128
```

New source snapshots are isolated under a separate dataset root. Verify and
index them before switching the manifest used by the service. The current
implementation does not reuse unchanged embeddings across snapshots.

## Search and Research overview performance

The service caches deterministic search and analytics responses for five minutes,
using up to 16 MiB per server process. Keys include the dataset snapshot, concept,
inclusive year bounds, result limit and retrieval settings. Identical concurrent
requests share one calculation. Failed requests are retried normally, and Ask
model calls are never cached. Set `PUBMED_CACHE_MB=0` to disable this cache or use
a value up to 64 to change its byte budget. Restarting the service clears it.

Research overview loads independently of ranked search and displays each panel
as it finishes. Pending counts display as unavailable until computed. The first
request for a new scope still scans the relevant tables; caching does not change
which records count or reduce the ANN search accuracy settings. Search reads
article card metadata once and fetches vector passages only for selected papers
(or reranking candidates).

Run `python ops/benchmark_views.py` on the application server to measure first
and repeated overview calculations without invoking an answer model or loading
the search engine. These timings exclude browser/network costs. Updating code
and restarting is required for an existing deployment to use these changes;
the dataset and embeddings do not need rebuilding.

For faster first requests to the most-cited panel, build the optional citation
summary once after downloading or creating a dataset:

```powershell
python pipeline/manage.py analytics --dataset covid-files/dataset.json
```

This derives counts from the existing references and leaves article tables,
dataset selection and embeddings unchanged. It writes a checksummed summary
under the selected index's `analytics` directory. The app verifies its snapshot,
source footer, schema and checksum before first use, then reuses that validation
while the files remain unchanged. Missing, damaged or stale summaries use the
raw reference calculation. Valid summaries are included in newly prepared
release inventories; older releases continue to work without them.

Topic membership tables are also reused across search and analytics panels,
with a separate 32 MiB retained-data budget per process. Each panel still
applies its own year bounds. Set `PUBMED_SCOPE_CACHE_MB=0` to disable reuse or
choose a budget up to 128 MiB. Very large topics can exceed the cache budget
and will be recalculated; this budget is not a limit on total query memory.
Citation and registry-link lists use stable paper-ID tie breaks so equal values
do not shuffle when switching between cached and raw execution.

Use `python ops/benchmark_views.py --concept-id D000086382 --compare-raw` to
compare the optimization against raw calculations on the same installation.
The benchmark checks that the complete panel responses match. The process and
operating-system file caches may already be warm; timings exclude browser costs.

To prepare both the citation summary and indexed article lookup:

```powershell
python pipeline/manage.py performance --dataset covid-files/dataset.json
```

The article lookup uses a small SQLite PMID index and a Parquet copy with 1,000-
record groups, so fetching selected records reads only their groups and requested
columns. It uses about 2.2 GB of additional disk on the full snapshot, with a
2 GB free-space reserve. Preparation streams batches, validates row counts and
source identity, and publishes the files atomically. Missing, stale or damaged
indexes fall back to the original article scan. Valid lookup files are included
in release inventories; serving does not create them automatically on Spaces.

Keyword search checks shard metadata before scoring and skips shards with no
eligible papers. Partial top-k selection replaces full sorting while retaining
historical score ties, exact match counts, and filters. ANN parameters are unchanged.
Overview applies year filters before topic joins and loads article titles only
for returned citation/trial rows. To save the actual query plans for inspection,
add `--plans-dir ops/logs/query-plans` to the overview benchmark command.

The Windows updater now prepares optional performance indexes before starting
its server. See [Windows automatic updates](ops/README.md) for the one-time task
activation and restart instructions; local file edits only reach other machines
after a tested commit is pushed.

## The language model

Ask now lets each user choose between **GPT OSS 120B through Groq** and **local
Ollama**. When the Groq configuration below is enabled, it is selected by default.
The local choice uses `qwen2.5:7b` unless `PUBMED_MODEL` is set, and appears as
unavailable until that model is installed and Ollama is running on the **machine
hosting the app**. Run `ollama pull qwen2.5:7b` there if needed. A remote Hugging
Face Space cannot use Ollama running on a visitor's laptop. The dense retrieval
model shown in Search status is separate from either answer model. Historical
local CPU answers took roughly 95 seconds to five minutes in earlier runs, not
a current latency promise; the default answer request budget is 120 seconds.

`start_all.ps1` optionally reads `llm.env` beside the launcher. It accepts only
literal uppercase `PUBMED_*` assignments; blank lines and full-line `#` comments
are ignored. Surrounding single/double quotes are stripped; values are never
executed, interpolated or printed. Inline comments and escapes are not parsed.
Unsupported/malformed entries receive a line-number-only warning. Existing process
environment entries win, and the first accepted file entry wins over duplicates.
Keep this file private and out of version control.

A key alone **never selects or enables cloud**. Cloud requires all of the following
explicit configuration after policy approval: `PUBMED_LLM=cloud`,
`PUBMED_ALLOW_CLOUD=1`, an approved `PUBMED_API_BASE`, `PUBMED_API_KEY`, and
`PUBMED_CLOUD_MODEL`. `llm.env.example` illustrates endpoint/model/key entries but
does not supply the required cloud consent flag. Do not enable it implicitly.
Cloud requests can send questions and retrieved source data to that endpoint.
The API key stays on the server; the browser sends only the provider choice.

Direct Python commands (including the viewer and direct service/agent entrypoints)
**do not load `llm.env`**: configure their inherited process environment yourself,
or use the launcher for serving. The launcher's `-Dataset` parameter selects its
manifest (default `covid-files/dataset.json`); viewer selection is independent as
described above.

## Known limits — state these before someone finds them

- **Author identity is provisional.** Many records have no stable author ID and
  fall back to name-based keys, which can merge or split real people. Every
  researcher ranking carries this caveat on screen.
- **Coverage is snapshot-specific.** These selected XML records are not a
  representative or complete PubMed census. Earlier year-distribution percentages
  should not be reused for a different snapshot without measuring it.
- **MeSH coverage varies.** Indexing can lag publication; author keywords are a
  second concept source, not proof that missing MeSH concepts are absent.
- **Country and institution labels are heuristic.** They are extracted from
  free-text affiliations; country parsing uses the last comma-separated segment.
  Labels are not reliably normalised or disambiguated. No validated country
  accuracy percentage is established for the selected snapshot.
- **No reviewed evaluation set yet.** ANN agreement and smoke behavior were
  measured, but biomedical retrieval and answer accuracy remain unmeasured.

## Requirements

```
pip install -r requirements.txt          # to run
pip install -r requirements-build.txt    # to rebuild from XML
```
