---
title: PubMed Literature Intelligence
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Evidence-grounded search over 3.2 million PubMed records
---

# PubMed Literature Intelligence

Search 3,217,739 supplied PubMed records, inspect the source behind each result,
and ask for evidence-grounded explanations.

```
"heart attack"  →  280 journals · 226 drugs · 6,313 researchers
```

## Why this is not a normal RAG demo

Most "AI over documents" systems retrieve the top 10–20 passages and let a model
write over them. That cannot answer *"list every journal publishing on this"* —
it will confidently name eight out of 280.

This one runs two lanes and routes between them:

| Lane | Used for | Answer |
|---|---|---|
| Parquet aggregation | journals, drugs, researchers, countries | **exact, whole corpus** |
| BM25 + dense vectors | open-ended questions | ranked sample, labelled as such |

The language model never produces a number. It picks a tool, reads the result,
and writes a sentence over it — and every figure is checked against a tool
result before it is shown. If no tool returned rows, it refuses rather than
inventing an answer.

## What is inside

- **3,217,739** supplied PubMed records from the selected 159-file snapshot
- **6,290,649** searchable passages embedded with `BAAI/bge-small-en-v1.5`
- Structured tables for journals, authors, citations, MeSH concepts and trials
- Hybrid BM25 and vector retrieval over the same immutable snapshot

## Known limits

- **Author identity is provisional.** 76% of author rows have no ORCID and are
  matched on surname + initial, so distinct people merge. Rankings carry this
  caveat on screen.
- **95% of the corpus is 2014–2026.** Rankings describe recent activity.
- **MeSH lags publication** — only 38.8% of 2026 papers are indexed, which is
  why author keywords are a second concept source.
- **Country is a heuristic** and is not normalised.
- **Not yet evaluated for accuracy.** That is the next piece of work.

## Configuration

| Secret / variable | Purpose |
|---|---|
| `PUBMED_DATA_REPO` | dataset repo holding `store/` and `index/` |
| `HF_TOKEN` | only if that dataset is private |
| `PUBMED_LLM=cloud` | enables the Ask tab |
| `PUBMED_API_KEY` | key for the OpenAI-compatible endpoint |
| `PUBMED_API_BASE` | e.g. `https://api.groq.com/openai/v1` |
| `PUBMED_CLOUD_MODEL` | `openai/gpt-oss-120b` for the recommended Groq setup |
| `PUBMED_ALLOW_CLOUD=1` | explicit consent to send questions and selected passages to the endpoint |

Without the LLM variables the Ask tab is disabled; the other research views work
normally, since none of them use a language model.
