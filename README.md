# PubMed Literature Intelligence

New to the project? Read the [complete beginner walkthrough](PROJECT_WALKTHROUGH.md)
for the architecture, data flow, every production file/function, deployment,
tests, limitations and a ready-to-use demonstration script.

Build and launch instructions: [pipeline/IMPLEMENTATION.md](pipeline/IMPLEMENTATION.md).
Completed Kaggle embedding workflow: [kaggle/README.md](kaggle/README.md).

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

The UI has five tabs:

- **Evidence**: ranked papers with source passages, record details and CSV export.
- **Research overview**: publication trend, study types, in-collection citations
  and registry links.
- **Entities**: limited rankings for journals, named substances, researchers,
  countries and institutions, with structured counts and caveats.
- **Compare**: two recognised concepts' publication counts and overlap, not
  clinical effectiveness.
- **Ask**: opt-in typed count/list answers or verified extractive evidence;
  insufficient evidence or request limits can produce a refusal.

Analytics use the selected concept and supported year filters, not the free-text
query used by Evidence. Without a concept selection, analytics cover the collection
within supported year filters. Panel errors (including HTTP 200 `{error}` responses)
are displayed separately from empty results; other successful panels remain visible.

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

## The language model

The default provider is **Ollama**, with local generation model `qwen2.5:7b`
unless `PUBMED_MODEL` is explicitly set. The dense retrieval model shown in Search
status is separate from the answer model; Ask shows the model/provider reported
by the answer response. Historical CPU answers took roughly 95 seconds to five
minutes in earlier runs, not a current latency promise. Request budgets can stop
an answer sooner (the agent's default request budget is 120 seconds).

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
