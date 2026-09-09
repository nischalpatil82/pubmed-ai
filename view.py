#!/usr/bin/env python3
"""
Read any table in the store. Parquet is binary, so a text editor shows noise -
this prints it as a table instead.

    python view.py                      list every table with its row count
    python view.py articles             first 10 rows
    python view.py articles 30          first 30 rows
    python view.py articles 5 --wide    show every column, one per line
    python view.py journals --csv       write journals.csv you can open in Excel
"""
import os
import sys

import polars as pl

STORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "store")


def tables():
    return sorted(f[:-8] for f in os.listdir(STORE) if f.endswith(".parquet"))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    if not args:
        print(f"tables in {STORE}\n")
        for t in tables():
            path = os.path.join(STORE, f"{t}.parquet")
            n = pl.scan_parquet(path).select(pl.len()).collect().item()
            mb = os.path.getsize(path) / 1e6
            print(f"  {t:24s} {n:>10,} rows   {mb:>7.1f} MB")
        print("\npython view.py <table> [rows] [--wide] [--csv]")
        return

    name = args[0]
    if name not in tables():
        sys.exit(f"no table '{name}'. Available: {', '.join(tables())}")
    rows = int(args[1]) if len(args) > 1 else 10

    path = os.path.join(STORE, f"{name}.parquet")
    total = pl.scan_parquet(path).select(pl.len()).collect().item()

    if "--csv" in flags:
        out = os.path.join(os.path.dirname(STORE), f"{name}.csv")
        pl.read_parquet(path).write_csv(out)
        print(f"wrote {total:,} rows -> {out}")
        print("open it in Excel. Note: articles.csv will be large (~150 MB).")
        return

    df = pl.scan_parquet(path).head(rows).collect()
    print(f"{name}.parquet   {total:,} rows, {len(df.columns)} columns\n")

    if "--wide" in flags:
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
