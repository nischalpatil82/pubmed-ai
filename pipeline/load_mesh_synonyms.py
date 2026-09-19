#!/usr/bin/env python3
"""
Fix the biggest accuracy hole in the local build: users type "ovarian cancer",
MeSH calls it "Ovarian Neoplasms", and the corpus returns nothing.

NLM publishes every synonym it recognises as <Term> entries inside the
descriptor file, plus the tree numbers that let a topic query also match its
narrower terms. Both are free and take one download.

    curl -O https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.xml
    python load_mesh_synonyms.py desc2026.xml --out ./store

Adds to the store:
    synonyms.parquet   synonym -> concept_id      (~600k rows)
    mesh_tree.parquet  concept_id -> tree_number  (topic expansion)
"""
from __future__ import annotations

import argparse
import gzip
import os

import polars as pl
from lxml import etree


def open_any(p):
    return gzip.open(p, "rb") if p.endswith(".gz") else open(p, "rb")


def load(desc_path: str, out_dir: str) -> None:
    syn_rows, tree_rows = [], []
    with open_any(desc_path) as fh:
        ctx = etree.iterparse(fh, events=("end",), tag="DescriptorRecord",
                              load_dtd=False, no_network=True, resolve_entities=False)
        for _, rec in ctx:
            ui = rec.findtext("DescriptorUI")
            name = rec.findtext("DescriptorName/String")
            if not ui:
                continue
            seen = set()
            for term in rec.findall(".//Term/String"):
                t = (term.text or "").strip().lower()
                if t and t not in seen:
                    seen.add(t)
                    syn_rows.append((t, ui, name))
            for tn in rec.findall("TreeNumberList/TreeNumber"):
                tree_rows.append((ui, (tn.text or "").strip(), name))
            rec.clear()
            while rec.getprevious() is not None:
                del rec.getparent()[0]

    os.makedirs(out_dir, exist_ok=True)
    pl.DataFrame(syn_rows, schema=["synonym", "concept_id", "concept_name"],
                 orient="row").write_parquet(f"{out_dir}/synonyms.parquet",
                                             compression="zstd")
    pl.DataFrame(tree_rows, schema=["concept_id", "tree_number", "concept_name"],
                 orient="row").write_parquet(f"{out_dir}/mesh_tree.parquet",
                                             compression="zstd")
    print(f"synonyms   {len(syn_rows):,} rows -> {out_dir}/synonyms.parquet")
    print(f"mesh_tree  {len(tree_rows):,} rows -> {out_dir}/mesh_tree.parquet")
    print("\ntools.py picks these up automatically on next run.")


def descendants(concept_id: str, store: str = "./store") -> list[str]:
    """
    Every descriptor at or below a concept in the MeSH tree. Without this, a
    query for 'Ovarian Neoplasms' misses every paper indexed only under a
    narrower term such as 'Carcinoma, Ovarian Epithelial'.
    """
    tree = pl.scan_parquet(f"{store}/mesh_tree.parquet")
    roots = (tree.filter(pl.col("concept_id") == concept_id)
                 .select("tree_number").collect()["tree_number"].to_list())
    if not roots:
        return [concept_id]
    expr = pl.lit(False)
    for r in roots:
        expr = expr | (pl.col("tree_number") == r) | pl.col("tree_number").str.starts_with(r + ".")
    return (tree.filter(expr).select("concept_id").unique()
                .collect()["concept_id"].to_list())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("desc_xml", help="desc2026.xml from NLM")
    ap.add_argument("--out", default="./store")
    a = ap.parse_args()
    load(a.desc_xml, a.out)
