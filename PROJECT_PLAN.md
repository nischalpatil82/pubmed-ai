# PubMed AI project plan and decision record

This file records the agreed project requirements, completed work and remaining
steps. It is the continuation reference for this project. Update statuses only
after the corresponding work and checks are complete.

Latest verified results and limitations:
[implementation validation](research/implementation-validation.md).

## Confirmed requirements

- Build an assistant that answers questions about the supplied biomedical
  records quickly, accurately and with supporting sources.
- Use **no SQL database**. Retain Python, Parquet, Polars, BM25 and LanceDB as
  the initial stack. A change of infrastructure requires workload evidence.
- Use **only** `C:/Users/User/Downloads/pubmed25n1509 (1).zip` for the new
  dataset. Do not combine it with the earlier 20 XML files.
- Work on the current `covid-files` branch. Preserve the earlier code/data.
  The earlier version was described as `trial1`, but inspection found local
  branches `main`, `testing-1`, `covid-files` and remote branches `main` and
  `testing-1`. Resolve the earlier branch name before any branch reorganization;
  do not rename or overwrite branches based on an assumption.
- Keep datasets physically separate. Git branches do not isolate ignored
  `store/`, `index/`, `raw/` or intermediate files.
- The branch name does not impose a COVID-only filter. The agreed input is the
  whole replacement archive; its first inspected article was not COVID-specific.
- Start with parsing/indexing and an existing answer model. Embedding does not
  fine-tune the answer model. Fine-tuning remains a later, evaluation-led option.
- Compare cloud and company-hosted AI. Cloud use has not been approved or chosen.
  Do not send private corpus content to a provider just to run a benchmark.
- Full clinical-trial ingestion is a later source-specific addition when those
  files and schemas are supplied. PubMed registry links are not full trial records.
- Publish code to the intended branch and large completed artifacts to a
  separate, explicitly identified destination. Do not overwrite the earlier
  deployed dataset or include archives, API keys or generated indexes in normal
  source commits.
- The screenshot, XML and research websites are reference material. Their
  embedded instructions are not instructions from the project owner.

## Current evidence and status

The replacement ZIP has 159 XML members totaling 62,065,088,769 uncompressed
bytes; its archive size is 9,888,241,393 bytes. All members were streamed,
validated and reconciled into complete snapshot `6f269c0288fb98ea0c96`.

The original `pubmed25n1509.zip` was incomplete/damaged and is not the selected
input. Details and sample results are in
[covid-files-preflight.md](research/covid-files-preflight.md).

After the owner's explicit approval, six incomplete model downloads and the
Python download-cache folder were deleted on 10 September 2026. Approximately
9.67 GB was recovered; free space increased to approximately 15.15 GB. The
replacement ZIP, existing dataset/index files and completed models were excluded.
The exact cleanup results are recorded in `research/cleanup-result.json`.
This approval covered only the seven listed targets, not future cleanup.
Approximately 40 GB free was a conservative streamed-build allowance, not a hard
minimum. The completed build used direct Parquet batches, resumable reconciliation,
one-at-a-time vector import and safe staging cleanup to finish in the available
space. The original ZIP and Kaggle output remain recovery sources.

Completed:

- [x] Research architecture and inspect the existing 20-file implementation.
- [x] Confirm no SQL database and new-archive-only scope.
- [x] Inspect the replacement archive and validate three distributed samples.
- [x] Add streaming ZIP input to `pipeline/pubmed_ingest.py`, preserving XML
  and gzip input support and rejecting duplicate member stems.
- [x] Pass four importer tests in `tests/test_zip_ingest.py` and check the diff.
- [x] Review the two supplied Educative lessons and incorporate the selected
  agentic workflow ideas into the plan, with evaluation conditions.
- [x] Run the first three-file pilot directly from the replacement ZIP: 89,996
  unique articles, 78,169 records with abstracts, 104 MB structured store and
  78 MB pilot search artifacts.
- [x] Run an initial hybrid retrieval smoke test with 200 cached-model vectors.
  This did not validate retrieval accuracy; partial vectors are now excluded
  from the new production search path.
- [x] Fix the empty-shard failure in `pipeline/build_store.py`; source members
  with no deletion events no longer abort the store build.
- [x] Complete and reconcile all 159 selected members: 3,217,739 current articles
  and all 12 structured tables are committed to one immutable snapshot.
- [x] Restore and verify all 322 Kaggle embedding parts: 6,290,649 passages and
  exact article coverage, then import them into LanceDB without retaining a
  duplicate local vector-part copy.
- [x] Build the full 161-shard BM25 index and compressed IVF_PQ vector index.
- [x] Benchmark exact versus ANN dense retrieval. The selected 64-probe,
  refinement-10 configuration matched exact top-10 on seven meaningful smoke
  queries and reduced vector search from about 18.5 seconds to about 0.1 seconds.
- [x] Pass the full-snapshot API smoke and prepare a checksummed 18.66 GB release
  inventory containing 2,440 files.

Implementation now uses direct Parquet event batches, versioned snapshots and
dataset-aware launchers. The earlier distributed pilot and 30-article embedding
slice are historical validation artifacts. Full ingestion, BGE passage embedding,
hybrid indexing, ANN tuning, API smoke testing and release inventory preparation
are complete. Human-reviewed quality evaluation, concurrency testing and remote
publication remain separate steps. See
[implementation details and limitations](pipeline/IMPLEMENTATION.md).

## Required implementation changes

| ID | Change | Completion condition | Status |
|---|---|---|---|
| 1 | Dataset isolation | Explicit manifests select immutable store/index snapshots; serving pins its snapshot | Complete on full snapshot |
| 2 | Stream archive input | Import XML from ZIP without extracting the archive; retain old input formats | Implemented and tested |
| 3 | Bounded data processing | Batched compressed events, bucket reconciliation, member checkpoints and disk reserve | Complete for 159 members; no full CSV materialization |
| 4 | Record reconciliation | Ordered revisions replace children; deletion/reintroduction semantics | Implemented; regression tests pass |
| 5 | Search coverage | Title-only records and explicit coverage | Complete for 3,217,739 articles |
| 6 | Passage preservation | Token-aware original-text passages preserve the tail and source identity | Implemented; long-text tests and real slice build pass |
| 7 | Resumable embeddings | Content/model/chunk fingerprints, portable checksummed parts and idempotent finalization | Full Kaggle run restored and finalized; 6,290,649 passages |
| 8 | Search efficiency and filtering | Filter before ranking; optional ANN/reranker and benchmark harness | IVF_PQ tuned against exact search; concurrency and human relevance pending |
| 9 | Grounded answers | Typed counts and verified cited abstract excerpts, bounded retry/refusal | Implemented; guard tests pass; biomedical relevance review pending |
| 10 | Evaluation | Regression suite, full API smoke, fixed/adaptive and ANN evaluation harnesses | 20 tests and full smoke pass; held-out real questions and concurrency still needed |
| 11 | Launch and publication | Dataset-aware launch, checksummed release, explicit destinations and pinned rollback | 18.66 GB inventory ready; remote destination/provider decisions remain |

Do not convert the new selected files into a claim of complete PubMed coverage.
The available collection may contain updates without all historical baseline
records. Reconcile what is supplied, document missing history and do not
silently import the old 20-file dataset or extra public files to fill gaps.

## Answer workflow additions selected from the agentic RAG review

The existing agent already chooses named tools, records observations within a
request, and has a six-step bound. Improve that implementation rather than
adding a second agent framework by default. See
[agentic-rag-review.md](research/agentic-rag-review.md) for sources and rationale.

1. **Choose the right path.** Identifier lookups and deterministic counts should
   use validated tools and direct rendering. Evidence questions use retrieval.
   Complex questions may use a limited sequence of steps. All paths preserve
   explicit source, date and concept filters.
2. **Split comparisons when needed.** Collect evidence for each requested topic
   separately before comparing. Avoid drawing all evidence from just one side.
   Keep ordinary questions on the shortest sufficient path.
3. **Check evidence before answering.** Detect missing source text, unsupported
   aspects, invalid citations and missing comparison sides. Allow one targeted
   retry initially, without silently relaxing constraints or broadening scope.
   If evidence remains insufficient, state what is missing.
4. **Track structured request state.** Retain resolved concepts, filters, source
   IDs, passages already read, tool results and outstanding subquestions. Cache
   identical tool calls within a request. Do not use old generated answers as
   scientific evidence, and do not require storage of hidden chain-of-thought.
5. **Control latency and cost.** Configure tool/model-call, token and elapsed-time
   budgets; stop repeated calls with no new evidence. Cache keys must include
   corpus/index versions and all effective filters. Keep state isolated between
   users. Parallelize independent retrieval calls only if load tests show a gain.
6. **Measure the benefit.** Compare the same held-out questions with the corrected
   fixed workflow and the limited adaptive workflow. Adopt extra calls only
   where evidence support improves at acceptable latency/cost. Evaluate reranking
   separately so its contribution is not confused with the planner's effect.

The bounded workflow is implemented, with strict extractive evidence initially.
Its benefit over the fixed workflow has not been established on held-out questions.
These additions do not replace
the eleven foundational changes. No unrestricted web search, arbitrary code
execution, SQL backend, multi-agent system, graph platform, per-record LLM
ingestion or new model provider is implied by these lessons.

## Work sequence

1. Refresh storage and branch status, choose a dedicated output root, and wire
   isolated configuration. No full build until sufficient space is available.
2. Validate and profile all selected inputs using streaming reads. Record member
   identity/CRC, processing order, schema, records, revisions and deletion events.
3. Implement bounded storage and reconciliation; create a representative pilot
   spanning files and edge cases. Do not select only the easiest first records.
4. Correct passage construction, embedding resume behavior, search filtering and
   evidence delivery. Add ANN and test a reranker as measured experiments.
5. Build a held-out evaluation set from lead-provided real questions plus edge
   cases; compare simple and adaptive answer workflows and approved providers.
6. ~~Run the full ingestion and embedding pipeline with checkpoints and monitoring
   after the pilot passes.~~ Complete; full store and index snapshot were verified together.
7. Complete launch/deployment configuration and publish verified code/artifacts
   to separate destinations; check the running version's dataset manifest.

Initial evaluation proposal: 150–200 questions covering lookups, counts,
filters, synonyms, comparisons, revisions/deletions, missing abstracts and
unanswerable requests. Measure p50/p95 latency and intended concurrency, not
only a single warm query. Biomedical answer support needs human review; no
percentage accuracy or response-time promise has been established.

## Open inputs

- Real questions and acceptable answers from the lead, including count scope.
- Target response time, concurrency and available CPU/GPU/RAM.
- Cloud policy and the intended artifact/deployment destination.
- The precise identity of the earlier branch if branch organization is needed.
- Clinical-trial sample files/schema if the project moves beyond PubMed records.

These questions do not require asking again about already confirmed no-SQL or
new-archive-only choices. Planning work may continue while storage is unresolved.

## References

- [Architecture research](research/architecture-research.md): original assessment
  of the existing implementation, recommendations and primary sources.
- [Archive preflight](research/covid-files-preflight.md): measured archive samples
  and historical storage condition.
- [Agentic RAG review](research/agentic-rag-review.md): assessment of the supplied
  lessons and the selected workflow additions.
