#!/usr/bin/env python3
"""
Stage 1 of the no-SQL stack: CSV -> Parquet fact store.

Parquet + Polars gives exact aggregation with a Python API. No server, no SQL,
no schema migrations - just columnar files on disk that Polars reads lazily.
Later, pointing the same tool functions at Postgres changes nothing upstream.
"""
import argparse
import glob
import os

import polars as pl

SCHEMAS = {
    "articles": ["pmid", "nlm_id", "title", "abstract", "pub_year", "pub_month",
                 "volume", "issue", "language", "doi", "pmcid", "status", "date_revised"],
    "journals": ["nlm_id", "medline_ta", "title", "iso_abbrev", "issn_linking", "country"],
    "authors": ["author_key", "display_name", "last_name", "fore_name", "initials",
                "orcid", "is_group"],
    "authorships": ["pmid", "author_key", "position", "is_first", "is_last",
                    "n_authors", "affiliation"],
    "mesh_headings": ["pmid", "descriptor_ui", "descriptor_name", "is_major", "qualifier_uis"],
    "substances": ["pmid", "substance_ui", "substance_name", "registry_number"],
    "keywords": ["pmid", "term", "is_major"],
    "citations": ["citing_pmid", "cited_pmid"],
    "publication_types": ["pmid", "type_ui", "type_name"],
    "grants": ["pmid", "grant_id", "agency", "country"],
    "databank_links": ["pmid", "databank", "accession"],
    "deleted_pmids": ["pmid"],
}

INT_COLS = {"pub_year", "pub_month", "date_revised", "position", "n_authors"}
BOOL_COLS = {"is_group", "is_first", "is_last", "is_major"}

# The parser dedups journals and authors only WITHIN one file, because the
# Postgres path relies on an upsert to dedup globally. There is no upsert here,
# so an author appearing in 12 of the 20 files lands as 12 identical rows - and
# any join through that table multiplies the joined rows by 12. That silently
# inflated every is_first / is_last count in rank_kols. Dedup on load.
DEDUP_KEYS = {
    "articles": ["pmid"],
    "journals": ["nlm_id"],
    "authors": ["author_key"],
    "mesh_headings": ["pmid", "descriptor_ui"],
    "substances": ["pmid", "substance_ui"],
    "publication_types": ["pmid", "type_ui"],
    "authorships": ["pmid", "position"],
    "keywords": ["pmid", "term"],
    "citations": ["citing_pmid", "cited_pmid"],
    "deleted_pmids": ["pmid"],
}


def country_from_affiliation(col: pl.Expr) -> pl.Expr:
    """
    Cheap, honest country extraction: last comma-separated segment, stripped of
    trailing e-mail noise. Gets ~90% and is transparent about the rest.
    """
    return (col.str.replace_all(r"(?i)\s*(electronic address|email)\s*:.*$", "")
               .str.replace_all(r"\.\s*$", "")
               .str.split(",").list.last().str.strip_chars()
               .str.replace_all(r"^\d{4,6}\s+", "")
               .str.slice(0, 60))


def build(csv_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for table, cols in SCHEMAS.items():
        files = sorted(glob.glob(os.path.join(csv_dir, f"{table}__*.csv")))
        if not files:
            print(f"  {table:20s} (no files)")
            continue

        lf = pl.scan_csv(files, has_header=False, new_columns=cols,
                         schema_overrides={c: pl.Utf8 for c in cols},
                         infer_schema_length=0, truncate_ragged_lines=True,
                         ignore_errors=True)
        casts = []
        for c in cols:
            if c in INT_COLS:
                casts.append(pl.col(c).cast(pl.Int32, strict=False).alias(c))
            elif c in BOOL_COLS:
                casts.append((pl.col(c) == "1").alias(c))
        if casts:
            lf = lf.with_columns(casts)
        if table == "authorships":
            lf = lf.with_columns(country_from_affiliation(pl.col("affiliation")).alias("country"))

        dest = os.path.join(out_dir, f"{table}.parquet")
        keys = DEDUP_KEYS.get(table)
        if keys:
            # For articles, a PMID revised in a later file should win.
            if table == "articles":
                lf = lf.sort("date_revised", nulls_last=False)
            df = lf.collect()
            before = df.height
            df = df.unique(subset=keys, keep="last", maintain_order=False)
            dropped = before - df.height
            df.write_parquet(dest, compression="zstd")
            n, note = df.height, (f"  (-{dropped:,} dup)" if dropped else "")
        else:
            lf.sink_parquet(dest, compression="zstd")
            n = pl.scan_parquet(dest).select(pl.len()).collect().item()
            note = ""
        mb = os.path.getsize(dest) / 1e6
        print(f"  {table:20s} {n:>9,} rows   {mb:>7.1f} MB{note}")

    # ---- vocabulary: every concept a user might name, in one file.
    # This is the entity-resolution surface. Free-text "cisplatin" or
    # "carotid stenosis" gets matched here BEFORE any retrieval happens.
    mesh = (pl.scan_parquet(os.path.join(out_dir, "mesh_headings.parquet"))
              .group_by("descriptor_ui", "descriptor_name")
              .agg(pl.col("pmid").n_unique().alias("n_papers"))
              .rename({"descriptor_ui": "concept_id", "descriptor_name": "concept_name"})
              .with_columns(pl.lit("mesh").alias("kind"), pl.lit(None, pl.Utf8).alias("code")))
    subs = (pl.scan_parquet(os.path.join(out_dir, "substances.parquet"))
              .group_by("substance_ui", "substance_name", "registry_number")
              .agg(pl.col("pmid").n_unique().alias("n_papers"))
              .rename({"substance_ui": "concept_id", "substance_name": "concept_name",
                       "registry_number": "code"})
              .with_columns(pl.lit("substance").alias("kind")))
    # Author keywords are the third source, and on a recent slice they are the
    # decisive one: MeSH indexing lags publication, so 2026 papers are only
    # 41.7% MeSH-covered while 76.7% carry keywords. A MeSH-only vocabulary is
    # blind to most of the newest corpus. Keywords have no UI, so the concept
    # id is synthesised from the term itself.
    kws = (pl.scan_parquet(os.path.join(out_dir, "keywords.parquet"))
             .with_columns(pl.col("term").str.strip_chars().str.to_lowercase().alias("name_lc"))
             .filter(pl.col("name_lc").str.len_chars() > 2)
             .group_by("name_lc")
             .agg(pl.col("pmid").n_unique().alias("n_papers"),
                  pl.col("term").first().alias("concept_name"))
             # singletons are mostly typos and one-off phrases; they bloat the
             # in-memory lookup without ever being queried.
             .filter(pl.col("n_papers") >= 2)
             .with_columns((pl.lit("kw:") + pl.col("name_lc")).alias("concept_id"),
                           pl.lit("keyword").alias("kind"),
                           pl.lit(None, pl.Utf8).alias("code")))

    vocab = (pl.concat([mesh.select("concept_id", "concept_name", "kind", "code", "n_papers"),
                        subs.select("concept_id", "concept_name", "kind", "code", "n_papers"),
                        kws.select("concept_id", "concept_name", "kind", "code", "n_papers")])
               .with_columns(pl.col("concept_name").str.to_lowercase().alias("name_lc"))
               .sort("n_papers", descending=True))
    dest = os.path.join(out_dir, "vocabulary.parquet")
    vocab.sink_parquet(dest, compression="zstd")
    n = pl.scan_parquet(dest).select(pl.len()).collect().item()
    print(f"  {'vocabulary':20s} {n:>9,} concepts")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="./csv")
    ap.add_argument("--out", default="./store")
    a = ap.parse_args()
    build(a.csv, a.out)
    print(f"\nfact store -> {a.out}")
