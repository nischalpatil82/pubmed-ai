# Free Colab embedding

Use [pubmed_embedding.ipynb](pubmed_embedding.ipynb) as an alternative workflow
after a selected dataset has a complete `articles.parquet`. The current local
snapshot already has a completed 159-member Kaggle embedding run, so do not run
Colab again for snapshot `6f269c0288fb98ea0c96`.

## Before opening Colab

1. Use `pubmed-ai-colab-code.zip` with the notebook's default direct-upload
   mode. After the branch is pushed, `USE_GIT = True` can clone it instead.
2. Put the complete dataset directory in Google Drive. The folder must include
   `dataset.json` and the referenced
   `stores/<snapshot>/articles.parquet` and `indexes/<snapshot>/` paths.
3. Confirm enough persistent storage. Temporary Colab disk is erased when a
   session ends. The vector parts alone are likely around 10–13 GB for the full
   archive; passage files and the final Lance index need additional space.

## Running one resumable worker

For an immediate speed test, upload `pubmed-pilot-articles.zip`, set
`UPLOAD_PILOT_ARCHIVE = True` and `ALLOW_PILOT = True`, and run only the
benchmark cell. It contains 51,124 pilot articles and is not the full corpus.
Return both flags to `False` before the full run.

Open the notebook in one account. Keep `TOTAL_WORKERS = 1`, `WORKER_ID = 0`,
`ROWS_PER_PART = 10000`, and `MAX_RUNTIME_MINUTES = 270`. The worker writes
completed parts to Google Drive and stops cleanly after four and a half hours,
leaving time within a roughly five-hour runtime for setup and checkpoint saving.

Run the 10,000-article benchmark first. Use its measured passages/second and
projected hours rather than a generic T4 estimate. Then run the worker cell.
Completed parts have both data files and a checksummed JSON manifest. If Colab
disconnects, reconnect, enter the same worker number and rerun the worker cell;
completed parts are skipped.

After the worker finishes, run `verify`. It checks every expected part,
checksums, and the exact total number of assigned articles. Only then run
`finalize` once. Finalization builds one
Lance table containing IDs, filter fields and vectors. Passage text remains in
compressed Parquet and is fetched only for returned hits.

Finalization temporarily requires both the durable vector parts and the new
Lance table. Do not manually delete vector parts until the release is verified
and stored elsewhere. No cleanup is automatic.

Google says free managed Colab runtimes may terminate distributed-computing
workers. This notebook therefore defaults to one free runtime. Multiple workers
remain available for an environment where parallel workers are permitted.

## What the checkpoints guarantee

- A `.partial` file is never accepted as completed work.
- Changing the corpus snapshot, model revision, chunking rules, worker count or
  article-part size invalidates incompatible checkpoints.
- Changing GPU batch size does not change chunk IDs, so it may be lowered after
  an out-of-memory error without discarding completed parts.
- Worker verification checks that the worker covers the article table exactly
  once.
- Final import uses chunk IDs as upsert keys, so a crash between a Lance write
  and its finalize checkpoint cannot create duplicate vectors.

This verifies data completeness and file integrity. Retrieval relevance still
requires the held-out questions and reviewed PMIDs described in
`pipeline/IMPLEMENTATION.md`.
