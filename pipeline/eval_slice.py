"""Make a small, explicitly labelled validation slice AFTER reconciliation."""
import argparse
import os
from pathlib import Path
import polars as pl
from build_store import SCHEMAS, build_vocabulary
from dataset import paths, atomic_json, digest, read_json


def make_slice(destination, per_source=10):
    store, _, parent = paths()
    root = Path(destination).resolve()
    if root.exists():
        raise ValueError("Choose a new validation folder; existing data is preserved")
    articles = (pl.scan_parquet(store / "articles.parquet").sort("pmid")
                .group_by("source_member").head(per_source).collect())
    pmids = articles["pmid"].to_list()
    snapshot = digest([parent["snapshot"], sorted(pmids), "validation-slice-v1"])[:20]
    output = root / "store"
    output.mkdir(parents=True)
    authorships = pl.scan_parquet(store / "authorships.parquet").filter(pl.col("pmid").is_in(pmids)).collect()
    for table in SCHEMAS:
        lf = pl.scan_parquet(store / f"{table}.parquet")
        if table == "articles":
            lf = articles.lazy()
        elif table == "authors":
            lf = lf.filter(pl.col("author_key").is_in(authorships["author_key"].unique().to_list()))
        elif table == "journals":
            lf = lf.filter(pl.col("nlm_id").is_in(articles["nlm_id"].drop_nulls().unique().to_list()))
        else:
            lf = lf.filter(pl.col("citing_pmid" if table == "citations" else "pmid").is_in(pmids))
        lf.sink_parquet(output / f"{table}.parquet", compression="zstd")
    build_vocabulary(str(output))
    (root / "index").mkdir()
    atomic_json(output / "manifest.json", {"snapshot": snapshot, "parent": parent,
        "scope": "Small validation subset of reconciled records; not an accuracy evaluation", "articles": articles.height})
    atomic_json(root / "dataset.json", {"dataset_id": root.name, "snapshot": snapshot,
        "status": "ready", "store": "store", "index": "index", "pilot": True})
    print(f"Validation slice: {articles.height} articles; {root / 'dataset.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-source", type=int, default=10)
    a = ap.parse_args()
    os.environ["PUBMED_DATASET"] = str(Path(a.dataset).resolve())
    make_slice(a.out, a.per_source)
