# Local, no-SQL build — runbook

This is the historical single-file runbook. For the current `covid-files`
dataset, use [IMPLEMENTATION.md](IMPLEMENTATION.md); its manifest-aware commands
replace the legacy build and launch commands below.

No database server. No SQL. Everything runs on your laptop CPU.
Verified end to end on `pubmed26n1443.xml.gz` (23,362 articles).

```
XML  →  CSV  →  Parquet (facts)  ─┬─►  Polars tool functions   ← exact counts
                                  └─►  BM25 + LanceDB vectors  ← semantic search
                                            ▲
                                    Ollama local model
                                    (fills tool arguments only —
                                     never writes a query, never a number)
```

## Why this is not "RAG without SQL"

The counting questions still get exact answers. Polars aggregates the whole
Parquet store — 23,362 articles in **0.5 seconds** — with a Python API instead
of SQL. `tools.py` exposes those aggregations as typed functions.

**The tool signature is the contract.** When you add Postgres later, only the
function bodies change. Tool schemas, agent, prompts and eval set stay
identical. That is what makes "SQL later" a swap, not a rewrite.

---

## Setup

```bash
pip install lxml polars bm25s PyStemmer lancedb fastembed ollama

ollama pull qwen3:8b          # ~5GB, trained for tool calling
```

On a CPU-only machine, prefer **qwen3:8b** or **granite4** — both are trained
for function calling, which matters far more here than raw size. The model only
emits a small JSON tool call and two sentences of prose, so an 8B model at
~10 tok/s is genuinely usable. A 20B+ model is not, on CPU.

## Build

```bash
# 1. parse XML -> CSV        (15s per file)
python pubmed_ingest.py --src ./pubmed26n1443.xml.gz --out ./csv

# 2. CSV -> Parquet facts    (7s)   78MB CSV -> 22MB Parquet
python build_store.py --csv ./csv --out ./store

# 3. keyword index           (8s)   no model download needed
python retrieval.py bm25

# 4. MEASURE before committing to a corpus size
python retrieval.py benchmark

# 5. dense vectors — resumable; safe to Ctrl-C and rerun
python retrieval.py vectors --model BAAI/bge-small-en-v1.5
```

## Ask it things

```bash
python agent.py "which journals publish on cisplatin?"
python agent.py "top KOLs for cisplatin since 2020"
python agent.py --dry "..."        # show the tool path without the LLM
```

---

## The three things that decide accuracy here

### 1. Load MeSH synonyms — do this first

Right now `spot_concepts("ovarian cancer")` returns **nothing**, because MeSH
calls it *Ovarian Neoplasms*. Users will not type MeSH headings.

```bash
curl -O https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.xml
python load_mesh_synonyms.py desc2026.xml --out ./store
```

~600,000 official synonyms plus the tree numbers for topic expansion. This is
the single highest-value hour in the whole local build.

### 2. Don't skip BM25 for vectors

BM25 needs no model, indexes 20,728 docs in 8 seconds, and on biomedical text
it beats dense retrieval on drug names, gene symbols and MeSH terms — the exact
tokens users search for. It is a peer of vector search here, not a fallback.
`HybridSearch` fuses both with reciprocal rank fusion, so the two score scales
never need calibrating.

### 3. The hallucination guard is load-bearing

Small models invent numbers. `check_answer()` extracts every 2+ digit number
from the answer and verifies it appeared in a tool result; if not, the agent is
told to try again. Cheap, and it catches the failure mode that would otherwise
end the project.

---

## Scaling reality on CPU

Embedding is the only expensive step. Rough CPU rates (**run
`retrieval.py benchmark` on your own machine — do not trust these**):

| Corpus | bge-small (384d) | bge-base (768d) |
|---|---|---|
| 1 file · 20,728 abstracts | ~15–40 min | ~1–2 hrs |
| 20 files · ~415k | ~5–14 hrs | ~1–2 days |
| Full 31M | not feasible on CPU | not feasible |

So: **build on one file, demo on twenty, and rent a GPU for a day when you need
the full corpus.** Nothing in this design changes when you do — `build_vectors`
is resumable and stores the model name alongside the table, so switching to a
better embedding model later is a new table, not a migration.

Meanwhile the Parquet fact layer covers **100% of whatever you have parsed**,
instantly, with no embedding at all. Every counting and ranking question is
already answerable across the full corpus. Only semantic search is gated on
embedding — which is another reason not to lead with vectors.

---

## Files

| File | Role |
|---|---|
| `pubmed_ingest.py` | XML → CSV, streaming, parallel |
| `xml_profiler.py` | fingerprint any unknown XML, draft a mapping |
| `build_store.py` | CSV → Parquet fact store + vocabulary |
| `tools.py` | the typed functions the model calls — **the contract** |
| `retrieval.py` | BM25 + LanceDB hybrid search, resumable embedding |
| `agent.py` | Ollama tool-calling loop + hallucination guard |
| `load_mesh_synonyms.py` | MeSH synonyms and tree — do this first |

## Known limits

- **Author identity is provisional.** "Yun Wang" spans 414 institutions in the
  sample. `rank_kols` returns a `caveat` field saying so; surface it in the UI.
- **No citation counts** in PubMed XML. iCite or OpenAlex if you need influence.
- **Country parsing is a heuristic** (last comma-separated segment). Good to
  roughly 90%. Match against ROR when it needs to be better.
