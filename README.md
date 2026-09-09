# PubMed Literature Intelligence

Ask questions in plain English about 277,042 biomedical papers and get exact,
citable answers.

```
"heart attack"  ->  280 journals · 226 drugs · 6,313 researchers   (< 1 second)
```

The counting answers are **complete**, not a top-10 sample: every number is a
row count over the whole corpus, and every one can be clicked to see the papers
behind it.

---

## Folder layout

| Path | What | Ships to production? |
|---|---|---|
| `pipeline/` | all the code | yes |
| `store/` | 15 parquet tables, 337 MB | yes — **upload, do not rebuild** |
| `index/` | BM25 + 238,224 vectors, 594 MB | yes — **upload, do not rebuild** |
| `raw/` | the 20 source `.xml.gz` + MeSH `desc2026.xml`, 1.2 GB | no |
| `logs/` | runtime logs | no |
| `llm.env` | your API key | **never** |

`store/` and `index/` are the build output. Producing them took ~15 hours of
CPU embedding. They are ordinary files: copy them and the system works
elsewhere without embedding anything again.

## Run it

```powershell
powershell -ExecutionPolicy Bypass -File start_all.ps1
```

Then open <http://127.0.0.1:8010>. Allow ~40 seconds on a cold start — the
query-embedding model loads first.

```powershell
powershell -ExecutionPolicy Bypass -File progress.ps1   # what is running
python view.py                                          # look inside the data
python view.py journals 20
python view.py articles 3 --wide
```

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

Journals / Drugs / Researchers / Countries / Institutions are **exact totals**.
The Papers tab is a **ranked sample** and says so. A search-only system cannot
answer the first kind at all — it would list 8 journals out of 280.

The language model never produces a number. It picks a tool, reads what comes
back, and writes a sentence over it. Every figure in an answer is checked
against a tool result before it is shown.

## Adding more XML

```powershell
# drop new .xml.gz into raw\  then:
powershell -ExecutionPolicy Bypass -File add_files.ps1
```

Steps 1-3 rebuild from every file present, so duplicates are removed properly.
Step 4 embeds **only the new articles** — minutes, not another 15 hours.

## The language model

`llm.env` (copy `llm.env.example`) selects it:

- **cloud** — any OpenAI-compatible endpoint. Groq's free tier answers in ~3s.
- **ollama** — fully local and private, but 95 s to 5 min per answer on CPU.

Both use identical tool schemas; switching is one line.

## Known limits — state these before someone finds them

- **Author identity is provisional.** 76% of author rows have no ORCID and are
  matched on surname + first initial, so distinct people merge. 894,494
  "authors" is an upper bound. Every researcher ranking carries this caveat on
  screen.
- **95% of the corpus is 2014-2026.** Rankings describe recent activity, not
  all-time standing.
- **MeSH lags publication.** Only 38.8% of 2026 papers are MeSH-indexed, which
  is why author keywords are a second concept source.
- **Country is a heuristic** — the last comma-separated segment of an
  affiliation, roughly 90% right, and not normalised ("China" and "People's
  Republic of China" count separately).
- **No evaluation set yet.** The system has not been measured for accuracy.
  This is the most important thing still missing.

## Requirements

```
pip install -r requirements.txt          # to run
pip install -r requirements-build.txt    # to rebuild from XML
```
