"""Compare selected article lookups with source scans, without loading search/LLMs."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.dataset:
        os.environ["PUBMED_DATASET"] = str(Path(args.dataset).resolve())
    sys.path.insert(0, str(ROOT / "pipeline"))
    import article_lookup
    from dataset import paths
    store, index, cfg = paths()
    with pq.ParquetFile(store / "articles.parquet") as source:
        groups = source.num_row_groups
        ids = [source.read_row_group(i, columns=["pmid"]).column(0)[0].as_py()
               for i in range(0, groups, max(1, groups // 20))][:20]
    start = time.perf_counter()
    artifact = article_lookup.article_lookup_path(store, index, cfg["snapshot"])
    verification_ms = (time.perf_counter() - start) * 1000
    if artifact is None:
        raise RuntimeError("Prepare the article lookup with manage.py performance first")
    columns = ["pmid", "title", "abstract", "pub_year", "nlm_id", "doi", "source_member"]
    timings = {}
    with patch.object(article_lookup, "article_lookup_path", return_value=None):
        start = time.perf_counter()
        raw = article_lookup.article_records(store, index, cfg["snapshot"], ids, columns)
        timings["source_scan_ms"] = round((time.perf_counter() - start) * 1000, 3)
    start = time.perf_counter()
    indexed = article_lookup.article_records(store, index, cfg["snapshot"], ids, columns)
    timings["indexed_lookup_ms"] = round((time.perf_counter() - start) * 1000, 3)
    if not indexed.sort("pmid").equals(raw.sort("pmid")):
        raise RuntimeError("Indexed article records differ from source scan")
    report = {"snapshot": cfg["snapshot"], "selected_records": len(ids), "records_equal": True,
              "first_verification_ms": round(verification_ms, 3), **timings,
              "note": "Server calculation only; OS caches may be warm. Not full search latency."}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
