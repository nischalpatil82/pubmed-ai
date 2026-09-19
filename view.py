#!/usr/bin/env python3
"""
Read any table in the store. Parquet is binary, so a text editor shows noise -
this prints it as a table instead.

    python view.py                      list every table with its row count
    python view.py articles             first 10 rows
    python view.py articles 30          first 30 rows
    python view.py articles 5 --wide    show every column, one per line
    python view.py journals --csv       export the complete table as CSV
    python view.py articles 5 --dataset another-dataset/dataset.json

Selection follows pipeline.dataset.paths(): --dataset, PUBMED_DATASET,
explicit PUBMED_STORE/PUBMED_INDEX, then covid-files/dataset.json.
"""
import argparse
import os

import polars as pl

from pipeline.dataset import paths


def tables(store):
    return sorted(path.stem for path in store.glob("*.parquet"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", nargs="?", help="table name (omit to list tables)")
    parser.add_argument("rows", nargs="?", type=int, default=10, help="preview rows (default: 10)")
    parser.add_argument("--dataset", help="dataset manifest; overrides PUBMED_DATASET")
    parser.add_argument("--wide", action="store_true", help="display columns one per line")
    parser.add_argument("--csv", action="store_true", help="export the entire table beside the selected store")
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("rows must be a positive integer")
    if args.dataset:
        os.environ["PUBMED_DATASET"] = args.dataset
    try:
        store, _, _ = paths()
        if not store.is_dir():
            parser.error(f"selected store does not exist: {store}")
        available = tables(store)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.error(f"cannot select dataset: {exc}")

    if not args.table:
        print(f"tables in {store}\n")
        for t in available:
            path = store / f"{t}.parquet"
            n = pl.scan_parquet(path).select(pl.len()).collect().item()
            mb = path.stat().st_size / 1e6
            print(f"  {t:24s} {n:>10,} rows   {mb:>7.1f} MB")
        print("\npython view.py [table] [rows] [--dataset manifest] [--wide] [--csv]")
        return

    name = args.table
    if name not in available:
        parser.error(f"no table '{name}'. Available: {', '.join(available)}")
    rows = args.rows

    path = store / f"{name}.parquet"
    total = pl.scan_parquet(path).select(pl.len()).collect().item()

    if args.csv:
        out = store.parent / f"{name}.csv"
        pl.read_parquet(path).write_csv(out)
        print(f"wrote {total:,} rows -> {out}")
        print("CSV includes every row, not just the preview; large tables may exceed spreadsheet limits.")
        return

    df = pl.scan_parquet(path).head(rows).collect()
    print(f"{name}.parquet   {total:,} rows, {len(df.columns)} columns\n")

    if args.wide:
        for i, row in enumerate(df.iter_rows(named=True), 1):
            print(f"--- row {i} " + "-" * 52)
            for k, v in row.items():
                print(f"  {k:16s} {str(v)[:90]}")
    else:
        pl.Config.set_tbl_rows(rows)
        pl.Config.set_tbl_cols(10)
        pl.Config.set_fmt_str_lengths(30)
        print(df)


if __name__ == "__main__":
    main()
