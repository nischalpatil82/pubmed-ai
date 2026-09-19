#!/usr/bin/env python3
"""
`PubMed XML, gzipped XML, or ZIP archives -> flat CSVs for the fact-store build.

Streams XML elements without extracting ZIP archives. Emits one CSV per table
and source member into --out. Use a separate output folder for each dataset.

Usage:
    python pubmed_ingest.py --src /data/pubmed --out /data/csv --workers 8
    python pubmed_ingest.py --src ./pubmed26n1443.xml.gz --out ./csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import os
import sys
import zipfile
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed

from lxml import etree

TABLES = (
    "articles",
    "journals",
    "authors",
    "authorships",
    "mesh_headings",
    "substances",
    "keywords",
    "citations",
    "publication_types",
    "grants",
    "databank_links",
    "deleted_pmids",
)

MONTHS = {m: i for i, m in enumerate(
    "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), start=1)}


# ---------------------------------------------------------------- helpers

def text(node, path, default=None):
    el = node.find(path)
    if el is None:
        return default
    # itertext() keeps content that is broken up by inline markup such as
    # <i>, <sup>, <b> - common in titles and abstracts.
    s = "".join(el.itertext()).strip()
    return s or default


def pub_year(article):
    """Best available year: PubDate, else MedlineDate, else ArticleDate."""
    y = text(article, ".//Journal/JournalIssue/PubDate/Year")
    if y and y.isdigit():
        return int(y)
    md = text(article, ".//Journal/JournalIssue/PubDate/MedlineDate")
    if md:
        for tok in md.replace("-", " ").split():
            if len(tok) == 4 and tok.isdigit():
                return int(tok)
    y = text(article, ".//ArticleDate/Year")
    return int(y) if y and y.isdigit() else None


def pub_month(article):
    m = text(article, ".//Journal/JournalIssue/PubDate/Month")
    if not m:
        return None
    if m.isdigit():
        return int(m)
    return MONTHS.get(m[:3])


def abstract_text(article):
    """Structured abstracts have several <AbstractText Label=...> children."""
    parts = []
    for at in article.findall(".//Abstract/AbstractText"):
        body = "".join(at.itertext()).strip()
        if not body:
            continue
        label = at.get("Label")
        parts.append(f"{label}: {body}" if label else body)
    return " ".join(parts) or None


def author_key(last, fore, initials, orcid):
    """
    Provisional identity key. Prefer a supplied ORCID; otherwise use surname
    and the available full forename/initials. Names alone do not identify a
    unique person; no automatic disambiguation is claimed.
    """
    if orcid:
        return "orcid:" + orcid
    if not last:
        return None
    ini = (fore or initials or "").strip()
    return f"name:{last.lower()}|{ini.lower()}"


def clean_orcid(raw):
    if not raw:
        return None
    o = raw.strip().rstrip("/").split("/")[-1].replace("-", "").upper()
    if len(o) != 16:
        return None
    return f"{o[0:4]}-{o[4:8]}-{o[8:12]}-{o[12:16]}"


# ---------------------------------------------------------------- per record

def handle_article(node, w, seen_journals, seen_authors):
    cit = node.find("MedlineCitation")
    if cit is None:
        return
    pmid = text(cit, "PMID")
    if not pmid:
        return
    article = cit.find("Article")
    if article is None:
        return

    # ---- journal (dedup within the file; DB upsert dedups globally)
    nlm_id = text(cit, "MedlineJournalInfo/NlmUniqueID")
    if nlm_id and nlm_id not in seen_journals:
        seen_journals.add(nlm_id)
        w["journals"].writerow([
            nlm_id,
            text(cit, "MedlineJournalInfo/MedlineTA"),
            text(article, "Journal/Title"),
            text(article, "Journal/ISOAbbreviation"),
            text(cit, "MedlineJournalInfo/ISSNLinking"),
            text(cit, "MedlineJournalInfo/Country"),
        ])

    ids = {a.get("IdType"): (a.text or "").strip()
           for a in node.findall("PubmedData/ArticleIdList/ArticleId")}

    w["articles"].writerow([
        pmid,
        nlm_id,
        text(article, "ArticleTitle"),
        abstract_text(article),
        pub_year(article),
        pub_month(article),
        text(article, "Journal/JournalIssue/Volume"),
        text(article, "Journal/JournalIssue/Issue"),
        text(article, "Language"),
        ids.get("doi"),
        ids.get("pmc"),
        cit.get("Status"),
        text(cit, "DateRevised/Year"),
    ])

    # ---- authors + authorships
    authors = article.findall("AuthorList/Author")
    n = len(authors)
    for pos, a in enumerate(authors, start=1):
        last = text(a, "LastName")
        fore = text(a, "ForeName")
        initials = text(a, "Initials")
        collective = text(a, "CollectiveName")
        orcid = clean_orcid(text(a, 'Identifier[@Source="ORCID"]'))

        if collective and not last:
            key = "group:" + collective.lower()
            display = collective
        else:
            key = author_key(last, fore, initials, orcid)
            display = " ".join(x for x in (fore or initials, last) if x)
        if not key:
            continue

        if key not in seen_authors:
            seen_authors.add(key)
            w["authors"].writerow([key, display, last, fore, initials, orcid,
                                   1 if collective and not last else 0])

        affs = [("".join(x.itertext()).strip())
                for x in a.findall("AffiliationInfo/Affiliation")]
        w["authorships"].writerow([
            pmid, key, pos,
            1 if pos == 1 else 0,
            1 if (n > 1 and pos == n) else 0,
            n,
            affs[0] if affs else None,
        ])

    # ---- MeSH
    for mh in cit.findall("MeshHeadingList/MeshHeading"):
        d = mh.find("DescriptorName")
        if d is None:
            continue
        quals = [q.get("UI") for q in mh.findall("QualifierName")]
        w["mesh_headings"].writerow([
            pmid, d.get("UI"), (d.text or "").strip(),
            1 if d.get("MajorTopicYN") == "Y" else 0,
            "|".join(x for x in quals if x) or None,
        ])

    # ---- chemicals / substances  (this is the drug list)
    for ch in cit.findall("ChemicalList/Chemical"):
        nm = ch.find("NameOfSubstance")
        if nm is None:
            continue
        w["substances"].writerow([
            pmid, nm.get("UI"), (nm.text or "").strip(),
            text(ch, "RegistryNumber"),
        ])

    # ---- author keywords. MeSH lags publication by months, so the newest
    # papers - the bulk of a recent baseline slice - often carry no MeSH at
    # all. Measured on 1443: 2026 papers are 41.7% MeSH-indexed but 76.7%
    # keyworded. The two sources are complementary; the vocabulary needs both.
    for kw in cit.findall(".//KeywordList/Keyword"):
        term = "".join(kw.itertext()).strip()
        if term:
            w["keywords"].writerow([pmid, term,
                                    1 if kw.get("MajorTopicYN") == "Y" else 0])

    # ---- outgoing references. PubMed carries no citation COUNTS, but 36.4%
    # of records carry the reference list itself (257,658 PMID edges in file
    # 1443 alone). Inverting these across the corpus yields in-corpus citation
    # counts for free - the influence signal a KOL ranking needs.
    for ref in node.findall(".//ReferenceList/Reference"):
        for aid in ref.findall(".//ArticleIdList/ArticleId"):
            if aid.get("IdType") == "pubmed":
                cited = (aid.text or "").strip()
                if cited:
                    w["citations"].writerow([pmid, cited])

    for pt in article.findall("PublicationTypeList/PublicationType"):
        w["publication_types"].writerow([pmid, pt.get("UI"), (pt.text or "").strip()])

    for g in article.findall("GrantList/Grant"):
        w["grants"].writerow([pmid, text(g, "GrantID"), text(g, "Agency"),
                              text(g, "Country")])

    # ---- clinical-trial registry links (join key into CT.gov XML)
    for db in article.findall("DataBankList/DataBank"):
        name = text(db, "DataBankName")
        for acc in db.findall("AccessionNumberList/AccessionNumber"):
            w["databank_links"].writerow([pmid, name, (acc.text or "").strip()])


# ---------------------------------------------------------------- per file

@contextmanager
def open_source(path, member=None):
    """Read an XML source without extracting ZIP members onto the disk."""
    if member is not None:
        with zipfile.ZipFile(path) as archive, archive.open(member) as fh:
            if member.lower().endswith(".gz"):
                with gzip.GzipFile(fileobj=fh) as xml:
                    yield xml
            else:
                yield fh
    else:
        opener = gzip.open if path.lower().endswith(".gz") else open
        with opener(path, "rb") as fh:
            yield fh


def source_jobs(src, max_files=None):
    """Return deterministic source order; reject ambiguous output filenames."""
    if os.path.isfile(src):
        if src.lower().endswith(".zip"):
            with zipfile.ZipFile(src) as archive:
                jobs = [(src, item.filename) for item in archive.infolist()
                        if not item.is_dir()
                        and item.filename.lower().endswith((".xml", ".xml.gz"))]
        elif src.lower().endswith((".xml", ".xml.gz")):
            jobs = [(src, None)]
        else:
            raise ValueError("source must be .xml, .xml.gz, .zip, or an XML directory")
    else:
        jobs = [(path, None) for path in glob.glob(os.path.join(src, "*"))
                if os.path.isfile(path) and path.lower().endswith((".xml", ".xml.gz"))]
    jobs.sort(key=lambda job: (job[1] or job[0]).replace("\\", "/"))
    stems = set()
    for path, member in jobs:
        stem = os.path.basename((member or path).replace("\\", "/")).split(".")[0]
        if stem.lower() in stems:
            raise ValueError(f"duplicate source stem would overwrite output: {stem}")
        stems.add(stem.lower())
    return jobs[:max_files] if max_files else jobs


def process_file(path, out_dir, member=None):
    label = member or path
    stem = os.path.basename(label.replace("\\", "/")).split(".")[0]
    handles, w = {}, {}
    for t in TABLES:
        f = open(os.path.join(out_dir, f"{t}__{stem}.csv"), "w",
                 newline="", encoding="utf-8")
        handles[t] = f
        w[t] = csv.writer(f)

    seen_journals, seen_authors = set(), set()
    n = 0
    try:
        with open_source(path, member) as fh:
            ctx = etree.iterparse(fh, events=("end",),
                                  tag=("PubmedArticle", "DeleteCitation"),
                                  load_dtd=False, no_network=True, resolve_entities=False)
            for _, el in ctx:
                if el.tag == "DeleteCitation":
                    for p in el.findall("PMID"):
                        w["deleted_pmids"].writerow([(p.text or "").strip()])
                else:
                    handle_article(el, w, seen_journals, seen_authors)
                    n += 1
                el.clear()
                while el.getprevious() is not None:
                    del el.getparent()[0]
    finally:
        for f in handles.values():
            f.close()
    return label, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help=".xml, .xml.gz, .zip, or a directory of XML files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--max-files", type=int,
                    help="process only the first N XML members (useful for a pilot)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.max_files is not None and args.max_files < 1:
        sys.exit("--max-files must be positive")
    jobs = source_jobs(args.src, args.max_files)
    if not jobs:
        sys.exit(f"no XML records found under {args.src}")

    csv.field_size_limit(10 * 1024 * 1024)
    total = 0
    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_file, path, args.out, member) for path, member in jobs]
            for fut in as_completed(futs):
                p, n = fut.result()
                total += n
                print(f"  {os.path.basename(p):28s} {n:>7,} articles", flush=True)
    else:
        for path, member in jobs:
            p, n = process_file(path, args.out, member)
            total += n
            print(f"  {os.path.basename(p):28s} {n:>7,} articles", flush=True)

    print(f"\n{total:,} articles from {len(jobs)} file(s) -> {args.out}")


if __name__ == "__main__":
    main()
