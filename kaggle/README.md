# Kaggle T4 x2 embedding and automatic download

Use `pubmed_embedding.ipynb` with the existing private `pubmed` Kaggle input.
Keep Internet enabled and select **Settings > Accelerator > GPU T4 x2**.

Start the job with **Save Version > Save & Run All**. Do not use Quick Save and
do not run the cells manually. Kaggle executes the notebook from a clean
runtime, and a successful saved version publishes everything written under
`/kaggle/working`.

The notebook performs these steps in order:

1. verifies both T4 GPUs and the exact article-file SHA-256;
2. runs the measured 10,000-article benchmark;
3. embeds every article with the pinned BGE model revision;
4. verifies all 322 part manifests plus vector and passage checksums;
5. packages the parts into about 13 stored ZIP bundles;
6. writes `DOWNLOAD_READY.json` last.

The bundles avoid Kaggle output-file pagination problems and let the laptop
download, verify, extract, and delete one archive at a time. The saved version
is complete only when its Output tab contains the ready marker, the output
manifest, and every archive named in that manifest.

## Laptop watcher

Install the official Kaggle CLI and authenticate once:

```powershell
python -m pip install --user kaggle
kaggle auth login
```

After clicking **Save & Run All**, keep the laptop awake and start:

```powershell
python kaggle/auto_download.py
```

The watcher polls `nischalg/notebook1b4a41db61`. It ignores the old tiny saved
version because that version has no `DOWNLOAD_READY.json`. When the new run is
complete, it downloads each bundle separately, verifies SHA-256, extracts it to
`covid-files/indexes/6f269c0288fb98ea0c96`, deletes the downloaded bundle, and
writes `covid-files/DOWNLOAD_COMPLETE.json` only after every restored part has
passed verification.

The laptop needs enough permanent free space for the restored embeddings. It
does not need double that space because archives are deleted one at a time.

## Completed run

The saved Kaggle run completed all 3,217,739 articles and 6,290,649 passages in
322 parts. The watcher restored and verified all 13 bundles locally and wrote
`covid-files/DOWNLOAD_COMPLETE.json`. Those vector parts were then imported into
LanceDB and consumed one at a time; the 322 passage Parquets remain for evidence.
The Kaggle output is still the recovery copy, so this embedding run does not need
to be repeated for the current snapshot.
