#!/usr/bin/env python3
"""
Point this at ANY xml / xml.gz file. It tells you what is inside and writes a
candidate mapping YAML you can review in an hour instead of reverse-engineering
a DTD for three days.

This is the tool that makes "drop a new XML type in the folder" tractable:
    detect  -> is this a type we already know?
    profile -> what elements exist, how often, what do they look like?
    propose -> a mapping onto the canonical entity model
    review  -> a human corrects it (this is the only manual step)

Usage:
    python xml_profiler.py data/pubmed26n1443.xml.gz
    python xml_profiler.py data/ctg_studies.xml --records 2000 --out mappings/
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
from collections import Counter, defaultdict

from lxml import etree

# Canonical entities every source is mapped onto. Once a new XML type is
# mapped here, every query already written keeps working - unchanged.
CANONICAL = {
    "document.id":       ["pmid", "id", "nct_id", "accession", "docid", "identifier"],
    "document.title":    ["articletitle", "title", "brief_title", "official_title", "name"],
    "document.text":     ["abstracttext", "abstract", "brief_summary", "description",
                          "detailed_description", "textblock", "summary"],
    "document.date":     ["pubdate", "start_date", "date", "completion_date",
                          "date_revised", "effective_time"],
    "document.language": ["language", "lang"],
    "document.type":     ["publicationtype", "study_type", "doc_type", "type"],
    "person.name":       ["lastname", "forename", "author", "last_name", "first_name",
                          "investigator", "overall_official", "collectivename"],
    "person.identifier": ["orcid", "identifier"],
    "org.name":          ["affiliation", "agency", "sponsor", "facility", "institution",
                          "labeler", "manufacturer"],
    "org.country":       ["country", "nation"],
    "substance.name":    ["nameofsubstance", "intervention_name", "drug", "substance",
                          "activeingredient", "genericname", "brandname"],
    "substance.code":    ["registrynumber", "unii", "rxcui", "cas", "ndc"],
    "condition.name":    ["descriptorname", "condition", "indication", "mesh_term",
                          "keyword"],
    "venue.name":        ["journal", "title", "medlineta", "source", "registry"],
}

SKIP_ATTRS = {"UI", "Version", "ValidYN", "CompleteYN", "MajorTopicYN"}


def open_any(path):
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


# ------------------------------------------------------------------ detect

def detect(path, peek_bytes=200_000):
    """Root element, namespace and DOCTYPE - enough to route a known type."""
    with open_any(path) as fh:
        head = fh.read(peek_bytes)
    txt = head.decode("utf-8", errors="replace")

    doctype = None
    m = re.search(r"<!DOCTYPE\s+([^\s>]+)([^>]*)>", txt)
    if m:
        pub = re.search(r'PUBLIC\s+"([^"]+)"', m.group(2))
        doctype = {"name": m.group(1), "public_id": pub.group(1) if pub else None}

    root, ns = None, None
    for el in re.finditer(r"<([A-Za-z_][\w.:-]*)([^>]*)>", txt):
        tag = el.group(1)
        if tag.startswith(("?", "!")):
            continue
        root = tag
        nsm = re.search(r'xmlns\s*=\s*"([^"]+)"', el.group(2))
        ns = nsm.group(1) if nsm else None
        break

    return {"root": root, "namespace": ns, "doctype": doctype}


def find_record_tag(path, root):
    """
    The repeating unit. Almost every bulk XML is
    <Container><Record/><Record/>...</Container> - we want Record.
    """
    counts = Counter()
    with open_any(path) as fh:
        ctx = etree.iterparse(fh, events=("start",))
        depth = 0
        for _, el in ctx:
            depth += 1
            tree = el.getroottree()
            p = tree.getpath(el)
            lvl = p.count("/")
            if lvl == 2:                      # direct child of the root element
                counts[etree.QName(el).localname] += 1
            if depth > 20_000:
                break
    return counts.most_common(1)[0][0] if counts else root


# ------------------------------------------------------------------ profile

def profile(path, record_tag, max_records, stride=13):
    """
    NOTE: sample with a stride, not from the head of the file. Bulk XML is
    usually ordered, so the first N records are systematically unrepresentative
    - profiling the head of a PubMed file finds zero ORCIDs because the oldest
    citations sort first. Strided sampling costs nothing and avoids the trap.
    """
    paths = Counter()                          # element path -> record count seen in
    multi = Counter()                          # path -> times it repeats within a record
    samples = defaultdict(list)
    text_len = defaultdict(list)
    n = 0
    seen = 0

    with open_any(path) as fh:
        ctx = etree.iterparse(fh, events=("end",), tag=record_tag)
        for _, rec in ctx:
            seen += 1
            if seen % stride:                  # cheap uniform-ish spread
                rec.clear()
                while rec.getprevious() is not None:
                    del rec.getparent()[0]
                continue
            n += 1
            here = Counter()
            for el in rec.iter():
                if not isinstance(el.tag, str):
                    continue
                rel = _relpath(rec, el)
                if rel is None:
                    continue
                here[rel] += 1
                val = (el.text or "").strip()
                if val and len(samples[rel]) < 3:
                    samples[rel].append(val[:110])
                if val:
                    text_len[rel].append(len(val))
                for a, av in el.attrib.items():
                    if a in SKIP_ATTRS:
                        continue
                    ap = f"{rel}@{a}"
                    here[ap] += 1
                    if len(samples[ap]) < 3:
                        samples[ap].append(str(av)[:60])
            for p, c in here.items():
                paths[p] += 1
                if c > 1:
                    multi[p] += 1
            rec.clear()
            while rec.getprevious() is not None:
                del rec.getparent()[0]
            if n >= max_records:
                break

    fingerprint_src = sorted(p for p, c in paths.items() if c > n * 0.5)[:40]
    fp = hashlib.sha1("|".join(fingerprint_src).encode()).hexdigest()[:16]

    fields = []
    for p, c in paths.most_common():
        lens = text_len.get(p, [])
        fields.append({
            "path": p,
            "coverage": round(100.0 * c / n, 1),
            "repeats": round(100.0 * multi[p] / n, 1),
            "avg_len": round(sum(lens) / len(lens)) if lens else 0,
            "samples": samples.get(p, []),
        })
    return {"records_scanned": n, "fingerprint": fp, "fields": fields}


def _relpath(root, el):
    parts = []
    cur = el
    while cur is not None and cur is not root:
        parts.append(etree.QName(cur).localname)
        cur = cur.getparent()
    if cur is None:
        return None
    return "/".join(reversed(parts)) or None


# ------------------------------------------------------------------ propose

def propose(fields):
    """Suggest canonical slots by leaf-name match. A human confirms these."""
    out, used = {}, set()
    for slot, hints in CANONICAL.items():
        best = None
        for f in fields:
            if f["path"] in used or f["coverage"] < 2:
                continue
            leaf = f["path"].split("/")[-1].split("@")[-1].lower()
            # Hints are ordered most-specific first, so an earlier hint outranks
            # a later one. This is what stops <Journal><Title> from stealing
            # document.title away from <ArticleTitle>.
            rank = next((i for i, h in enumerate(hints) if leaf == h), None)
            if rank is None:
                rank = next((i + 6 for i, h in enumerate(hints)
                             if len(h) > 5 and h in leaf), None)
            if rank is None:
                continue
            # containers with no text of their own are structure, not values
            if f["avg_len"] == 0 and not f["path"].endswith(tuple(hints)):
                continue
            score = (100 - rank * 12) + f["coverage"] / 10.0
            if best is None or score > best[0]:
                best = (score, f)
        if best:
            out[slot] = {"xpath": best[1]["path"],
                         "coverage": best[1]["coverage"],
                         "repeating": best[1]["repeats"] > 5,
                         "sample": (best[1]["samples"] or [None])[0]}
            used.add(best[1]["path"])
    return out


def to_yaml(source, det, rec_tag, prof, mapping):
    L = [f"# auto-generated - REVIEW BEFORE USE",
         f"source: {source}",
         f"fingerprint: {prof['fingerprint']}",
         "detect:",
         f"  root: {det['root']}",
         f"  namespace: {det['namespace'] or 'null'}",
         f"  doctype: {(det['doctype'] or {}).get('name', 'null')}",
         f"record_xpath: {rec_tag}",
         "",
         "# canonical entity mapping - every existing query works once this is filled in",
         "canonical:"]
    for slot, m in mapping.items():
        L.append(f"  {slot}:")
        L.append(f"    xpath: {m['xpath']}")
        L.append(f"    repeating: {str(m['repeating']).lower()}")
        L.append(f"    coverage: {m['coverage']}%")
        if m["sample"]:
            L.append(f"    # e.g. {m['sample']}")
    L += ["", "# unmapped fields with >20% coverage - decide keep or drop", "unmapped:"]
    mapped = {m["xpath"] for m in mapping.values()}
    for f in prof["fields"]:
        if f["path"] not in mapped and f["coverage"] >= 20:
            L.append(f"  - {f['path']}   # {f['coverage']}%"
                     + (f"  e.g. {f['samples'][0]}" if f["samples"] else ""))
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--records", type=int, default=1000)
    ap.add_argument("--out", default=".")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    det = detect(args.path)
    rec_tag = find_record_tag(args.path, det["root"])
    prof = profile(args.path, rec_tag, args.records)
    mapping = propose(prof["fields"])

    print(f"\nFILE        {os.path.basename(args.path)}")
    print(f"root        <{det['root']}>   namespace={det['namespace']}")
    print(f"doctype     {(det['doctype'] or {}).get('name')}")
    print(f"record unit <{rec_tag}>   scanned {prof['records_scanned']:,}")
    print(f"fingerprint {prof['fingerprint']}   <- match this to route known types\n")

    print(f"{'ELEMENT PATH':52s} {'COV%':>6} {'MULTI':>6}  SAMPLE")
    print("-" * 118)
    for f in prof["fields"][:args.top]:
        s = (f["samples"] or [""])[0][:40]
        print(f"{f['path'][:52]:52s} {f['coverage']:>6.1f} {f['repeats']:>6.1f}  {s}")

    print(f"\nPROPOSED CANONICAL MAPPING  ({len(mapping)} of {len(CANONICAL)} slots)")
    print("-" * 118)
    for slot, m in mapping.items():
        print(f"  {slot:20s} <- {m['xpath'][:44]:44s} {m['coverage']:>5.1f}%"
              f"{'  [repeating]' if m['repeating'] else ''}")
    unmapped = [s for s in CANONICAL if s not in mapping]
    if unmapped:
        print(f"\n  not found: {', '.join(unmapped)}")

    src = re.sub(r"[^a-z0-9]+", "_", os.path.basename(args.path).lower().split(".")[0])
    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, f"{src}.mapping.yaml")
    with open(dest, "w") as fh:
        fh.write(to_yaml(src, det, rec_tag, prof, mapping))
    print(f"\ncandidate mapping -> {dest}   (review, correct, commit)")


if __name__ == "__main__":
    main()
