#!/usr/bin/env python3
"""
The tool layer. This is the contract the local LLM talks to.

Design rule: the model NEVER writes a query. It picks a function and fills in
typed arguments. Everything the user sees as a number comes back from one of
these functions, computed exactly over the whole corpus by Polars.

Typed tool arguments keep question planning separate from deterministic
Parquet/Polars calculations. No SQL database is used or planned by this change.
"""
from __future__ import annotations

import functools
import os
import re
import threading
from typing import Any

import polars as pl

from dataset import paths, read_json

STORE, INDEX, DATASET = paths()


def _t(name: str) -> pl.LazyFrame:
    return pl.scan_parquet(os.path.join(STORE, f"{name}.parquet"))


# ------------------------------------------------------------------ concepts

@functools.lru_cache(maxsize=1)
def _vocab() -> pl.DataFrame:
    return _t("vocabulary").collect()


_COLS = ["name_lc", "concept_id", "concept_name", "kind", "code", "n_papers"]


# These are related but distinct MeSH concepts.  They are deliberately kept
# separate from synonyms: COVID-19 is the disease and SARS-CoV-2 is the virus.
# Presenting the paired concept makes a search for either term discoverable
# without silently changing the user's selected scope to the other concept.
_RELATED_CONCEPTS = {
    "D000086382": ("D000086402",),  # COVID-19 -> SARS-CoV-2
    "D000086402": ("D000086382",),  # SARS-CoV-2 -> COVID-19
}


@functools.lru_cache(maxsize=1)
def _lookup() -> pl.DataFrame:
    """
    Every string a user might type, mapped to a concept in THIS corpus.

    Three sources, in order of authority:
      1. vocabulary.parquet  - MeSH descriptor names, substance names, author
                               keywords actually present in the corpus
      2. synonyms.parquet    - NLM's own entry terms ("ovarian cancer" ->
                               D010051 Ovarian Neoplasms). Restricted by an
                               inner join to descriptors that occur here, so a
                               synonym never resolves to a concept with 0 papers.

    Without (2), the corpus is only reachable by its controlled vocabulary, and
    nobody types controlled vocabulary.
    """
    v = _vocab().select(_COLS)
    syn_path = os.path.join(STORE, "synonyms.parquet")
    if not os.path.exists(syn_path):
        return v

    present = (_vocab().filter(pl.col("kind") == "mesh")
                       .select("concept_id", "n_papers").unique(subset=["concept_id"]))
    syn = (pl.read_parquet(syn_path)
             .rename({"synonym": "name_lc"})
             .join(present, on="concept_id", how="inner")
             .with_columns(pl.lit("mesh-synonym").alias("kind"),
                           pl.lit(None, pl.Utf8).alias("code"))
             .select(_COLS))
    # Deliberately NOT dropping synonyms whose string also exists in the
    # vocabulary. They point at DIFFERENT concepts, and dropping them cost the
    # MeSH concept its exact match: "ovarian cancer" is both an author keyword
    # (316 papers) and an entry term for Ovarian Neoplasms (452, and far more
    # once narrower terms expand). Keep both and let resolve_concept's coverage
    # weighting choose; a duplicate name across two concept ids is legitimate.
    return pl.concat([v, syn.unique(subset=["name_lc", "concept_id"])])


def resolve_concept(text: str, limit: int = 5) -> list[dict]:
    """
    Free text -> curated concept. ALWAYS call this first: the corpus is indexed
    by MeSH descriptor and substance UI, not by whatever word the user typed.
    Skipping it is how "cisplatin" silently returns nothing.
    """
    q = text.strip().lower()
    v = _lookup()

    # People type possessives and plurals; the vocabulary stores neither.
    # "parkinsons" is not a substring of "parkinson disease", so a literal
    # match found only a 3-paper keyword and missed Parkinson Disease on 666.
    # Try a few cheap spelling variants and keep the best hit per concept.
    variants = [q]
    flat = q.replace("\u2019", "").replace("'", "")
    if flat != q:
        variants.append(flat)
    for cand in list(variants):
        if len(cand) > 4 and cand.endswith("s"):
            variants.append(cand[:-1])
    seen_v, queries = set(), []
    for cand in variants:
        if cand and cand not in seen_v:
            seen_v.add(cand)
            queries.append(cand)

    # Match quality precedes frequency; corpus popularity must not change meaning.
    # alone. Tier-first ranking put "heart attack" - an author keyword on 5
    # papers - above the MeSH concept Myocardial Infarction on 614, because the
    # synonym that matched was the plural "heart attacks" and so fell one tier
    # down. A concept with 100x the coverage is almost always the one meant;
    # these weights still keep an exact match ahead of a similar one of
    # comparable size.
    frames = []
    for i, cand in enumerate(queries):
        # A variant is a weaker signal than what the user actually typed, so
        # discount it rather than letting a stripped plural outrank an exact hit.
        penalty = 1.0 if i == 0 else 0.85
        tier = (pl.when(pl.col("name_lc") == cand).then(1.0 * penalty)
                  .when(pl.col("name_lc").str.starts_with(cand)).then(0.6 * penalty)
                  .otherwise(0.3 * penalty))
        frames.append(v.filter(pl.col("name_lc").str.contains(cand, literal=True))
                       .with_columns(tier.alias("_score")))
    if not frames:
        return []
    out = (pl.concat(frames)
            .sort(["_score", "n_papers"], descending=True)
            .unique(subset=["concept_id"], keep="first", maintain_order=True)
            .head(limit))
    return out.select("concept_id", "concept_name", "kind", "code", "n_papers").to_dicts()


def related_concepts(concept_ids: list[str], limit: int = 3) -> list[dict]:
    """Return curated, in-corpus concepts related to already resolved terms.

    This is for discovery only.  Each returned row remains its own selectable
    scope, so a user can deliberately choose COVID-19 or SARS-CoV-2 rather
    than receiving an unexpected combined count.
    """
    wanted = []
    seen = set(concept_ids)
    parents: dict[str, str] = {}
    for concept_id in concept_ids:
        for related_id in _RELATED_CONCEPTS.get(concept_id, ()):
            if related_id not in seen and related_id not in wanted:
                wanted.append(related_id)
                parents[related_id] = concept_id

    if not wanted:
        return []
    rows = (_vocab().filter(pl.col("concept_id").is_in(wanted))
                    .select("concept_id", "concept_name", "kind", "code", "n_papers")
                    .to_dicts())
    by_id = {row["concept_id"]: row for row in rows}
    out = []
    for related_id in wanted:
        row = by_id.get(related_id)
        if row:
            out.append({**row, "kind": f"related {row['kind']}",
                        "related_to": parents[related_id]})
        if len(out) >= limit:
            break
    return out


@functools.lru_cache(maxsize=1)
def _name_index() -> dict[str, dict]:
    idx: dict[str, dict] = {}
    for r in _lookup().select(_COLS).iter_rows(named=True):
        prev = idx.get(r["name_lc"])
        if prev is None or (r["n_papers"] or 0) > (prev["n_papers"] or 0):
            idx[r["name_lc"]] = r
    return idx


_STOP = {"the", "a", "an", "of", "for", "in", "on", "and", "or", "to", "with",
         "which", "what", "who", "list", "top", "how", "many", "me", "are",
         "is", "publish", "publishes", "publishing", "journals", "drugs",
         "papers", "studies", "about", "show", "give", "find"}


def spot_concepts(text: str, limit: int = 5) -> list[dict]:
    """
    Find every curated concept mentioned anywhere in a sentence, longest match
    first. This is entity linking's front door: it turns "top KOLs for
    cisplatin in ovarian cancer" into two concept ids without the model having
    to guess which words are the medical terms.
    """
    idx = _name_index()
    words = [w for w in re.findall(r"[a-z0-9\-]+", text.lower())]
    hits: dict[str, dict] = {}
    for n in (4, 3, 2, 1):
        for i in range(len(words) - n + 1):
            gram = " ".join(words[i:i + n])
            if n == 1 and (gram in _STOP or len(gram) < 4):
                continue
            m = idx.get(gram)
            if m and m["concept_id"] not in hits:
                hits[m["concept_id"]] = {**m, "matched": gram, "words": n}
    out = sorted(hits.values(), key=lambda r: (-r["words"], -r["n_papers"]))
    return [{k: v for k, v in r.items() if k != "name_lc"} for r in out[:limit]]


@functools.lru_cache(maxsize=1024)
def _descendants(concept_id: str) -> tuple[str, ...]:
    """
    A concept plus every narrower descriptor beneath it in the MeSH tree.

    MeSH indexers tag a paper with the MOST SPECIFIC term that applies, so a
    paper about epithelial ovarian carcinoma is filed under 'Carcinoma, Ovarian
    Epithelial' and NOT under 'Ovarian Neoplasms'. Querying the parent alone
    therefore misses most of its own topic. Tree numbers are prefix-encoded
    (C04.588.322.455 sits under C04.588), so descendants are a prefix match.
    """
    p = os.path.join(STORE, "mesh_tree.parquet")
    if not os.path.exists(p) or concept_id.startswith("kw:"):
        return (concept_id,)
    tree = pl.scan_parquet(p)
    roots = (tree.filter(pl.col("concept_id") == concept_id)
                 .select("tree_number").collect()["tree_number"].to_list())
    if not roots:
        return (concept_id,)
    expr = pl.lit(False)
    for r in roots:
        expr = expr | (pl.col("tree_number") == r) | pl.col("tree_number").str.starts_with(r + ".")
    ids = tree.filter(expr).select("concept_id").unique().collect()["concept_id"].to_list()
    return tuple(sorted(set(ids) | {concept_id}))


def _pmids_for(concept_id: str, expand: bool = True) -> pl.LazyFrame:
    """
    All PMIDs carrying this concept - MeSH topic, substance, or author keyword.

    Keyword concepts carry a 'kw:' prefix and match on the lowercased term,
    because author keywords have no controlled UI. They exist here because MeSH
    lags publication: without them the newest papers, which are the bulk of a
    recent baseline slice, are unreachable by topic.
    """
    if concept_id.startswith("kw:"):
        term = concept_id[3:]
        return (_t("keywords")
                .filter(pl.col("term").str.strip_chars().str.to_lowercase() == term)
                .select("pmid").unique())
    ids = list(_descendants(concept_id)) if expand else [concept_id]
    mesh = _t("mesh_headings").filter(pl.col("descriptor_ui").is_in(ids)).select("pmid")
    sub = _t("substances").filter(pl.col("substance_ui").is_in(ids)).select("pmid")
    names = _lookup().filter(pl.col("concept_id").is_in(ids))["name_lc"].unique().to_list()
    keywords = (_t("keywords").filter(pl.col("term").str.strip_chars().str.to_lowercase().is_in(names))
                .select("pmid"))
    return pl.concat([mesh, sub, keywords]).unique()


def concept_scope(concept_id: str, expand: bool = True) -> dict:
    """What a scoped query actually covered - report this, never assume it."""
    ids = _descendants(concept_id) if expand else (concept_id,)
    return {"concept_id": concept_id, "expanded": expand and len(ids) > 1,
            "descriptors_included": len(ids), "match_fields": ["MeSH", "substances", "equivalent author keywords"]}


def validate_years(since_year=None, until_year=None):
    """Validate inclusive publication-year bounds for every query path."""
    for value in (since_year, until_year):
        if value is not None and (type(value) is not int or not 1000 <= value <= 3000):
            raise ValueError("Years must be integers between 1000 and 3000")
    if since_year is not None and until_year is not None and since_year > until_year:
        raise ValueError("since_year must not exceed until_year")


def _scoped(concept_id: str | None, since_year: int | None,
            expand: bool = True, until_year: int | None = None) -> pl.LazyFrame:
    validate_years(since_year, until_year)
    arts = _t("articles")
    if concept_id:
        arts = arts.join(_pmids_for(concept_id, expand), on="pmid", how="inner")
    if since_year is not None:
        arts = arts.filter(pl.col("pub_year") >= since_year)
    if until_year is not None:
        arts = arts.filter(pl.col("pub_year") <= until_year)
    return arts


# Parallel full-corpus analytics can multiply peak allocations. Serialize these
# collections and request streaming execution to reduce memory pressure.
# Streaming is not a memory cap: joins, groups and results may still be large.
# This lock is process-local; it does not coordinate multiple server workers.
_ANALYTICS_LOCK = threading.Lock()


def _collect_heavy(lf: pl.LazyFrame) -> pl.DataFrame:
    """Serialize participating collections; request lower-memory execution."""
    with _ANALYTICS_LOCK:
        return lf.collect(engine="streaming")


# ------------------------------------------------------------------ tools

@functools.lru_cache(maxsize=4096)
def _concept_name(concept_id: str) -> str:
    hit = _vocab().filter(pl.col("concept_id") == concept_id)
    return hit["concept_name"][0] if hit.height else concept_id


@functools.lru_cache(maxsize=1)
def _valid_ids() -> frozenset:
    """Every concept id that actually exists here, for rejecting invented ones."""
    return frozenset(_vocab()["concept_id"].to_list())


# A concept id looks like D010051 (MeSH), C582435 (substance) or kw:<term>.
#
# The digit count is NOT fixed. Older descriptors are D + 6 digits, but terms
# minted in the last few years are D + 9: COVID-19 is D000086382, Semaglutide
# D000099194, Wearable Electronic Devices D000076251. A regex capped at 7
# digits rejected every recent topic - the exact ones anyone demoing this
# system would type first - and the failure looked like "concept not found"
# rather than "bad pattern". Match any run of digits.
_ID_RE = re.compile(r"^(?:kw:.+|[A-Z]\d{4,12})$")


def _coerce_concept(concept_id: str | None):
    """
    Accept a concept id OR raw free text, and always return a real id.

    The prompt tells the model to call resolve_concept first. Small models
    routinely ignore that and pass "ovarian cancer" straight in as concept_id -
    which matched nothing and produced a confident "there are no journals on
    ovarian cancer". A tool that silently returns an empty answer for valid
    input is a broken tool, no matter whose fault the call was. So resolve it
    here and say what was assumed.

    Returns (concept_id, note, error).
    """
    if not concept_id:
        return None, None, None
    cid = concept_id.strip()
    if _ID_RE.match(cid):
        # Format is not existence. A model asked for cisplatin once produced
        # "D010051" - a real, well-formed MeSH id that means Ovarian Neoplasms.
        # It passed the shape check and returned a confidently wrong answer
        # under the right-sounding label. Verify the id is actually in this
        # corpus; an id that is merely plausible is the worst kind of input.
        if cid in _valid_ids():
            # Name the concept even when the id was valid. Existence checking
            # cannot catch a real-but-wrong id: asked for cisplatin, a model
            # produced D010051, which exists and means Ovarian Neoplasms. The
            # only defence left is making the scope visible, so an answer about
            # "cisplatin" that says "scope: Ovarian Neoplasms" is obviously
            # wrong to whoever reads it.
            return cid, f"Scope: {_concept_name(cid)} ({cid}).", None
        return None, None, (f"Concept id {concept_id!r} does not exist in this "
                            f"corpus. Call resolve_concept with the term itself "
                            f"instead of guessing an id.")
    hits = resolve_concept(cid, limit=1)
    if not hits:
        return None, None, (f"No concept in this corpus matches {concept_id!r}. "
                            f"Try resolve_concept to see near matches.")
    h = hits[0]
    return h["concept_id"], (f"Interpreted {concept_id!r} as {h['concept_name']} "
                             f"({h['concept_id']}, {h['n_papers']:,} papers)."), None


def list_journals(concept_id: str | None = None, since_year: int | None = None,
                  limit: int = 25, expand: bool = True,
                  until_year: int | None = None) -> dict:
    """Every journal publishing on a concept, with exact paper counts."""
    concept_id, note, err = _coerce_concept(concept_id)
    if err:
        return {"error": err, "total_journals": 0, "results": []}
    # Carry the full title, not just the abbreviation. "BMC Cardiovasc Disord"
    # is librarian shorthand; "BMC cardiovascular disorders" is what a reader
    # recognises. Both are in the data, so show the readable one and keep the
    # abbreviation as the secondary label people will see in citations.
    # NOTE: articles and journals BOTH have a column called "title". Joining
    # the whole journals table and then asking for pl.col("title") silently
    # returns the ARTICLE title - the journal column gets renamed to
    # title_right and is never seen. Select and rename before the join so the
    # name means one thing.
    jrn = (_t("journals")
           .select("nlm_id", "medline_ta", "country",
                   pl.col("title").alias("journal_title")))
    res = (_scoped(concept_id, since_year, expand, until_year=until_year)
           .join(jrn, on="nlm_id", how="left")
           .group_by("medline_ta", "country")
           .agg(pl.col("pmid").n_unique().alias("papers"),
                pl.col("journal_title").drop_nulls().first())
           .sort("papers", descending=True))
    full = _collect_heavy(res)
    return {"total_journals": full.height,
            "total_papers": int(full["papers"].sum()),
            "scope": concept_scope(concept_id, expand) if concept_id else None,
            "note": note,
            "results": full.head(limit).to_dicts()}


def list_drugs(concept_id: str | None = None, since_year: int | None = None,
               limit: int = 25, named_only: bool = True, expand: bool = True,
               until_year: int | None = None) -> dict:
    """
    Substances studied within a scope, with UNII/CAS registry numbers.
    named_only drops MeSH class terms ('Antioxidants') that carry no registry
    number, leaving actual named compounds.
    """
    concept_id, note, err = _coerce_concept(concept_id)
    if err:
        return {"error": err, "total_substances": 0, "results": []}
    scope = _scoped(concept_id, since_year, expand, until_year=until_year).select("pmid")
    res = (_t("substances").join(scope, on="pmid", how="inner"))
    if named_only:
        res = res.filter((pl.col("registry_number").is_not_null())
                         & (pl.col("registry_number") != "0"))
    if concept_id:
        res = res.filter(pl.col("substance_ui") != concept_id)
    res = (res.group_by("substance_name", "registry_number", "substance_ui")
              .agg(pl.col("pmid").n_unique().alias("papers"))
              .sort("papers", descending=True))
    full = _collect_heavy(res)
    return {"total_substances": full.height,
            "scope": concept_scope(concept_id, expand) if concept_id else None,
            "note": note,
            "results": full.head(limit).to_dicts()}


@functools.lru_cache(maxsize=1)
def _max_year() -> int:
    """
    The newest year the corpus MEANINGFULLY covers - not its maximum.

    Using the raw max let two records with a mistyped <Year>2028</Year> set the
    "recent" cutoff to 2027, so no paper counted as recent and the recency term
    in rank_kols silently contributed zero to every score. A single bad row
    should never be able to switch off a feature, so take a high percentile
    instead: outliers cannot move it, real coverage can.
    """
    yr = _collect_heavy(_t("articles").select("pub_year"))["pub_year"].drop_nulls()
    if not yr.len():
        return 2026
    return int(yr.quantile(0.99))


def _recent_cutoff(recent_since: int | None) -> int:
    """
    "Recent" must never be a literal. A hardcoded 2025 quietly stops meaning
    anything the moment the corpus moves on, and nothing fails loudly when it
    does. Default to the last full year actually present in the data.
    """
    return recent_since if recent_since is not None else _max_year() - 1


def rank_kols(concept_id: str | None = None, country: str | None = None,
              since_year: int | None = None, limit: int = 20,
              w_first: float = 1.5, w_last: float = 2.0, w_recent: float = 1.0,
              expand: bool = True, recent_since: int | None = None,
              until_year: int | None = None) -> dict:
    """
    Rank researchers by a transparent, tunable score.
    Weights are arguments, not constants - the client will want to argue with
    them, and that argument should be a parameter change, not a code change.
    """
    concept_id, note, err = _coerce_concept(concept_id)
    if err:
        return {"error": err, "total_authors": 0, "results": []}
    scope = _scoped(concept_id, since_year, expand, until_year=until_year).select("pmid", "pub_year")
    a = (_t("authorships").join(scope, on="pmid", how="inner")
         .join(_t("authors").filter(~pl.col("is_group")), on="author_key", how="inner"))
    if country:
        a = a.filter(pl.col("country").str.to_lowercase().str.contains(country.lower()))
    res = (a.group_by("author_key")
            .agg(pl.col("display_name").first().alias("name"),
                 pl.col("orcid").first(),
                 pl.col("pmid").n_unique().alias("papers"),
                 pl.col("is_first").sum().alias("first_author"),
                 pl.col("is_last").sum().alias("senior_author"),
                 pl.col("pub_year").max().alias("latest_year"),
                 pl.col("country").mode().first().alias("main_country"),
                 pl.col("affiliation").drop_nulls().first().alias("affiliation"),
                 (pl.col("pub_year") >= _recent_cutoff(recent_since)).sum().alias("recent"))
            .with_columns((pl.col("papers")
                           + pl.col("first_author") * w_first
                           + pl.col("senior_author") * w_last
                           + pl.col("recent") * w_recent).alias("kol_score"))
            .sort("kol_score", descending=True))
    full = _collect_heavy(res)
    rows = full.head(limit).to_dicts()
    # Honesty check surfaced with the answer, not buried in a log.
    warn = None
    if rows and any(not r["orcid"] for r in rows):
        warn = ("Authors without an ORCID are matched using the available name and may "
                "merge distinct people or split one person. Treat researcher identities as provisional.")
    return {"total_authors": full.height, "results": rows,
            "scope": concept_scope(concept_id, expand) if concept_id else None,
            "note": note, "caveat": warn}



def list_papers(concept_id: str | None = None, journal: str | None = None,
                substance_ui: str | None = None, author_key: str | None = None,
                since_year: int | None = None, limit: int = 50,
                expand: bool = True, until_year: int | None = None) -> dict:
    """
    The papers behind a number. Every count in this system is a set of real
    articles, so every count should be openable - otherwise the user is asked
    to trust a total they cannot inspect.

    Narrow by journal, substance or author to answer "which 108 papers?".
    """
    concept_id, note, err = _coerce_concept(concept_id)
    if err:
        return {"error": err, "total": 0, "results": []}

    lf = _scoped(concept_id, since_year, expand, until_year=until_year).join(
        _t("journals").select("nlm_id", "medline_ta"), on="nlm_id", how="left")

    if journal:
        lf = lf.filter(pl.col("medline_ta") == journal)
    if substance_ui:
        subs = _t("substances").filter(pl.col("substance_ui") == substance_ui).select("pmid")
        lf = lf.join(subs.unique(), on="pmid", how="inner")
    if author_key:
        auth = _t("authorships").filter(pl.col("author_key") == author_key).select("pmid")
        lf = lf.join(auth.unique(), on="pmid", how="inner")

    res = (lf.select("pmid", "title", "abstract", "pub_year",
                     pl.col("medline_ta").alias("journal"), "doi")
             .sort("pub_year", descending=True))
    full = _collect_heavy(res)
    rows = full.head(limit).to_dicts()
    for r in rows:
        ab = r.pop("abstract", None) or ""
        r["snippet"] = ab[:300] + ("\u2026" if len(ab) > 300 else "")
    return {"total": full.height, "note": note,
            "filter": {"journal": journal, "substance_ui": substance_ui,
                       "author_key": author_key},
            "results": rows}



def article_detail(pmid: str) -> dict:
    """
    Everything the corpus holds about one paper.

    The search result card shows a truncated abstract and nothing else, which
    makes a click a dead end - you either leave for PubMed or you learn no
    more. This assembles the whole record from the tables that already have it:
    the full abstract, who wrote it and where, what it was indexed under, and
    which substances it mentions.
    """
    pmid = str(pmid).strip()
    art = (_t("articles").filter(pl.col("pmid") == pmid)
           .join(_t("journals").select("nlm_id", "medline_ta",
                                       pl.col("title").alias("journal_title"),
                                       pl.col("country").alias("journal_country")),
                 on="nlm_id", how="left")
           .collect())
    if not art.height:
        return {"error": f"PMID {pmid} is not in this corpus"}
    row = art.row(0, named=True)

    authors = (_t("authorships").filter(pl.col("pmid") == pmid)
               .join(_t("authors"), on="author_key", how="left")
               .sort("position")
               .select("position", "display_name", "orcid", "affiliation",
                       "is_first", "is_last")
               .collect().to_dicts())
    mesh = (_t("mesh_headings").filter(pl.col("pmid") == pmid)
            .sort("is_major", descending=True)
            .select("descriptor_ui", "descriptor_name", "is_major")
            .collect().to_dicts())
    subs = (_t("substances").filter(pl.col("pmid") == pmid)
            .select("substance_ui", "substance_name", "registry_number")
            .collect().to_dicts())
    kw = (_t("keywords").filter(pl.col("pmid") == pmid)
          .select("term").collect()["term"].to_list())
    ptypes = (_t("publication_types").filter(pl.col("pmid") == pmid)
              .select("type_name").collect()["type_name"].to_list())
    trials = (_t("databank_links").filter(pl.col("pmid") == pmid)
              .select("databank", "accession").collect().to_dicts())
    cited_by = (_t("citations").filter(pl.col("cited_pmid") == pmid)
                .select(pl.len()).collect().item())
    refs = (_t("citations").filter(pl.col("citing_pmid") == pmid)
            .select(pl.len()).collect().item())

    return {"pmid": pmid, "title": row["title"], "abstract": row["abstract"],
            "pub_year": row["pub_year"], "pub_month": row["pub_month"],
            "journal": row.get("journal_title") or row.get("medline_ta"),
            "journal_abbrev": row.get("medline_ta"),
            "journal_country": row.get("journal_country"),
            "doi": row["doi"], "pmcid": row["pmcid"], "language": row["language"],
            "authors": authors, "mesh": mesh, "substances": subs,
            "keywords": kw, "publication_types": ptypes, "trials": trials,
            "cited_by_in_corpus": cited_by, "references_in_corpus": refs}


@functools.lru_cache(maxsize=1)
def _cached_corpus_stats() -> dict:
    """What is actually loaded. Call this when asked about coverage or scope."""
    arts = _t("articles").select("pmid", "pub_year", pl.col("abstract").is_not_null().alias("has_abstract")).collect()
    yr = arts["pub_year"].drop_nulls()
    return {"articles": arts.height,
            "with_abstract": int(arts["has_abstract"].sum()),
            "without_abstract": int((~arts["has_abstract"]).sum()),
            "dataset": DATASET,
            "coverage_note": "Selected source records only; not complete PubMed or full-text coverage. Author identities are provisional.",
            "journals": _t("journals").collect().height,
            # Two different numbers that are easy to confuse, so report both.
            # authorship_rows counts one row per author PER PAPER (7.2x articles);
            # authors counts distinct people. Showing the first under the label
            # "authors" overstates the roster by more than double.
            "authorship_rows": _t("authorships").select(pl.len()).collect().item(),
            "authors": _t("authors").select(pl.len()).collect().item(),
            "concepts": _vocab().height,
            # The raw min/max spans 1886-2028, which reads as "142 years of
            # coverage" and is badly misleading: one 1886 paper, two mistyped
            # 2028 ones, and 95% of the corpus in a single decade. Report the
            # span that actually holds the data alongside the true extremes.
            "year_range": [int(yr.min()), int(yr.max())] if yr.len() else None,
            "year_core": [int(yr.quantile(0.025)), int(yr.quantile(0.975))]
                         if yr.len() else None,
            "year_core_pct": 95}


def corpus_stats() -> dict:
    """Return a fresh top-level mapping so API annotations cannot alter the cache."""
    return dict(_cached_corpus_stats())


def get_articles(pmids: list[str], concept_id: str | None = None,
                 since_year: int | None = None, until_year: int | None = None) -> dict:
    """Fetch specific records within the concept and inclusive year scope."""
    concept_id, note, err = _coerce_concept(concept_id)
    if err:
        return {"error": err, "results": []}
    res = (_scoped(concept_id, since_year, until_year=until_year)
           .filter(pl.col("pmid").is_in(pmids))
           .join(_t("journals"), on="nlm_id", how="left")
           .select("pmid", "title", "abstract", "pub_year",
                   pl.col("medline_ta").alias("journal"), "doi")
           .collect())
    return {"note": note, "results": res.to_dicts()}



# ------------------------------------------------------ coverage extensions
#
# "The AI should answer whatever the user asks" is not a model problem - it is
# a coverage problem. A tool-calling model can only answer what the tool list
# lets it reach, so widening the answerable surface means adding tools, not a
# bigger model. Each function below turns a question people actually ask into
# an exact aggregation.

def trend_by_year(concept_id: str | None = None, since_year: int | None = None,
                  until_year: int | None = None) -> dict:
    """Papers per year for a concept. Use for 'is research on X growing', trends."""
    concept_id, note, err = _coerce_concept(concept_id)
    if err:
        return {"error": err, "results": []}
    res = (_scoped(concept_id, since_year, until_year=until_year)
           .filter(pl.col("pub_year").is_not_null())
           .group_by("pub_year").agg(pl.col("pmid").n_unique().alias("papers"))
           .sort("pub_year"))
    full = _collect_heavy(res)
    return {"total_papers": int(full["papers"].sum()) if full.height else 0,
            "note": note,
            "results": full.to_dicts()}


def list_countries(concept_id: str | None = None, since_year: int | None = None,
                   limit: int = 25, until_year: int | None = None) -> dict:
    """Which countries publish on a concept. Counts distinct papers, not authors."""
    concept_id, note, err = _coerce_concept(concept_id)
    if err:
        return {"error": err, "results": []}
    scope = _scoped(concept_id, since_year, until_year=until_year).select("pmid")
    res = (_t("authorships").join(scope, on="pmid", how="inner")
           .filter(pl.col("country").is_not_null() & (pl.col("country") != ""))
           .group_by("country").agg(pl.col("pmid").n_unique().alias("papers"))
           .sort("papers", descending=True))
    full = _collect_heavy(res)
    return {"total_countries": full.height, "note": note,
            "caveat": ("Country is parsed from the affiliation's last comma-separated "
                       "segment; its accuracy has not been validated on this dataset."),
            "results": full.head(limit).to_dicts()}


def list_institutions(concept_id: str | None = None, since_year: int | None = None,
                      limit: int = 25, until_year: int | None = None) -> dict:
    """Which institutions publish on a concept, from author affiliations."""
    concept_id, note, err = _coerce_concept(concept_id)
    if err:
        return {"error": err, "results": []}
    scope = _scoped(concept_id, since_year, until_year=until_year).select("pmid")
    # Affiliations are free text. Take the first segment naming an organisation;
    # crude, but transparent and good enough to rank the big centres.
    org = (pl.col("affiliation").str.split(",").list.eval(
        pl.element().filter(pl.element().str.contains(
            r"(?i)(University|Universit|Institut|Hospital|College|Center|Centre|School|Clinic)"))
    ).list.first().str.strip_chars())
    res = (_t("authorships").join(scope, on="pmid", how="inner")
           .filter(pl.col("affiliation").is_not_null())
           .with_columns(org.alias("institution"))
           .filter(pl.col("institution").is_not_null())
           .group_by("institution").agg(pl.col("pmid").n_unique().alias("papers"))
           .sort("papers", descending=True))
    full = _collect_heavy(res)
    return {"total_institutions": full.height, "note": note,
            "caveat": ("Institution is extracted from free-text affiliation strings "
                       "and is not normalised - the same centre may appear twice."),
            "results": full.head(limit).to_dicts()}


def top_cited(concept_id: str | None = None, since_year: int | None = None,
              limit: int = 20, until_year: int | None = None) -> dict:
    """
    Most-cited papers on a concept, counted WITHIN this corpus.

    PubMed publishes no citation counts, but 36% of records list their own
    references. Inverting those gives an influence signal at no external cost.
    """
    concept_id, note, err = _coerce_concept(concept_id)
    if err:
        return {"error": err, "results": []}
    scope = _scoped(concept_id, since_year, until_year=until_year).select("pmid")
    cited = (_t("citations")
             .join(scope.rename({"pmid": "cited_pmid"}), on="cited_pmid", how="inner")
             .group_by("cited_pmid").agg(pl.len().alias("times_cited"))
             .sort("times_cited", descending=True).head(limit))
    res = (cited.join(_t("articles").rename({"pmid": "cited_pmid"}),
                      on="cited_pmid", how="left")
                .join(_t("journals"), on="nlm_id", how="left")
                .select(pl.col("cited_pmid").alias("pmid"), "title", "pub_year",
                        pl.col("medline_ta").alias("journal"), "times_cited")
                .sort("times_cited", descending=True))
    full = _collect_heavy(res)
    return {"note": note,
            "caveat": ("Counts citations from papers inside this corpus only. A "
                       "landmark paper cited mostly by work outside the selected files "
                       "will score low."),
            "results": full.to_dicts()}


def list_study_types(concept_id: str | None = None, since_year: int | None = None,
                     limit: int = 20, until_year: int | None = None) -> dict:
    """Breakdown by publication type - trials, reviews, meta-analyses, case reports."""
    concept_id, note, err = _coerce_concept(concept_id)
    if err:
        return {"error": err, "results": []}
    scope = _scoped(concept_id, since_year, until_year=until_year).select("pmid")
    res = (_t("publication_types").join(scope, on="pmid", how="inner")
           .group_by("type_name").agg(pl.col("pmid").n_unique().alias("papers"))
           .sort("papers", descending=True))
    full = _collect_heavy(res)
    return {"total_types": full.height, "note": note,
            "results": full.head(limit).to_dicts()}


def find_trials(concept_id: str | None = None, since_year: int | None = None,
                limit: int = 25, until_year: int | None = None) -> dict:
    """Papers linked to a registered clinical trial, with the NCT number."""
    concept_id, note, err = _coerce_concept(concept_id)
    if err:
        return {"error": err, "results": []}
    scope = _scoped(concept_id, since_year, until_year=until_year).select("pmid")
    res = (_t("databank_links").filter(pl.col("accession").str.contains(r"^NCT\d{8}$"))
           .join(scope, on="pmid", how="inner")
           .join(_t("articles"), on="pmid", how="left")
           .select("pmid", "databank", "accession", "title", "pub_year")
           .sort("pub_year", descending=True))
    full = _collect_heavy(res)
    return {"total_links": full.height, "note": note,
            "caveat": ("Only explicitly recorded NCT links are included. "
                       "This is not a substitute for the ClinicalTrials.gov corpus."),
            "results": full.head(limit).to_dicts()}


def compare_concepts(concept_a: str, concept_b: str,
                     since_year: int | None = None, until_year: int | None = None) -> dict:
    """How two topics compare in volume, without materializing PMID sets in Python."""
    a_id, a_note, a_err = _coerce_concept(concept_a)
    b_id, b_note, b_err = _coerce_concept(concept_b)
    if a_err or b_err:
        return {"error": a_err or b_err}
    # A broad MeSH term can contain millions of papers. Keep each PMID set in
    # Polars and ask the query engine for the three counts, rather than copying
    # all identifiers into Python sets at once.
    a = _scoped(a_id, since_year, until_year=until_year).select("pmid").unique()
    b = _scoped(b_id, since_year, until_year=until_year).select("pmid").unique()
    a_count = _collect_heavy(a.select(pl.len().alias("count"))).item()
    b_count = _collect_heavy(b.select(pl.len().alias("count"))).item()
    both_count = _collect_heavy(a.join(b, on="pmid", how="inner").select(pl.len().alias("count"))).item()
    return {"note": " ".join(x for x in (a_note, b_note) if x),
            "a": {"concept": _concept_name(a_id), "concept_id": a_id, "papers": a_count},
            "b": {"concept": _concept_name(b_id), "concept_id": b_id, "papers": b_count},
            "overlap_papers": both_count,
            "overlap_pct_of_a": round(100 * both_count / a_count, 1) if a_count else 0,
            "overlap_pct_of_b": round(100 * both_count / b_count, 1) if b_count else 0}


# ------------------------------------------------------------------ schemas

def _p(**props):
    return {"type": "object", "properties": props}


CID_DESC = {"type": "string", "description": "The drug, disease or topic in the USER'S OWN WORDS, e.g. 'lung cancer'. Resolved automatically. Do NOT invent a code like D008175."}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "resolve_concept",
        "description": ("Map free text like 'cisplatin' or 'carotid stenosis' to a curated "
                        "concept id. Call this FIRST for any question naming a drug, disease "
                        "or topic. Other tools take concept_id, not free text."),
        "parameters": _p(text={"type": "string", "description": "the term the user used"},
                         limit={"type": "integer", "default": 5}) | {"required": ["text"]}}},
    {"type": "function", "function": {
        "name": "spot_concepts",
        "description": ("Find every curated concept mentioned in a whole sentence, longest match "
                        "first. Prefer this over resolve_concept when the question mentions more "
                        "than one term, e.g. 'KOLs for cisplatin in ovarian cancer'."),
        "parameters": _p(text={"type": "string", "description": "the user's full question"},
                         limit={"type": "integer", "default": 5}) | {"required": ["text"]}}},
    {"type": "function", "function": {
        "name": "list_journals",
        "description": ("Exhaustive list of journals publishing on a concept, with exact paper "
                        "counts. Use for 'which journals', 'where is this published', 'how many'."),
        "parameters": _p(concept_id={"type": "string", "description": "The drug, disease or topic in the USER'S OWN WORDS, e.g. 'cisplatin' or 'ovarian cancer'. It is resolved to the right concept automatically. Do NOT pass a code like D010051 - codes you have not seen in a resolve_concept result in this conversation are wrong, and a wrong code silently answers about a different disease."},
                         since_year={"type": "integer"},
                         limit={"type": "integer", "default": 25})}},
    {"type": "function", "function": {
        "name": "list_drugs",
        "description": ("Substances studied within a scope, with UNII/CAS registry numbers. "
                        "Use for 'which drugs', 'what compounds', 'list substances'."),
        "parameters": _p(concept_id={"type": "string", "description": "The drug, disease or topic in the USER'S OWN WORDS, e.g. 'cisplatin' or 'ovarian cancer'. It is resolved to the right concept automatically. Do NOT pass a code like D010051 - codes you have not seen in a resolve_concept result in this conversation are wrong, and a wrong code silently answers about a different disease."},
                         since_year={"type": "integer"},
                         limit={"type": "integer", "default": 25},
                         named_only={"type": "boolean", "default": True})}},
    {"type": "function", "function": {
        "name": "rank_kols",
        "description": ("Rank researchers (key opinion leaders) by publication volume, "
                        "authorship position and recency. Use for 'who are the experts', "
                        "'top researchers', 'KOLs', 'leading investigators'."),
        "parameters": _p(concept_id={"type": "string", "description": "The drug, disease or topic in the USER'S OWN WORDS, e.g. 'cisplatin' or 'ovarian cancer'. It is resolved to the right concept automatically. Do NOT pass a code like D010051 - codes you have not seen in a resolve_concept result in this conversation are wrong, and a wrong code silently answers about a different disease."},
                         country={"type": "string", "description": "e.g. India, China"},
                         since_year={"type": "integer"},
                         limit={"type": "integer", "default": 20})}},
    {"type": "function", "function": {
        "name": "search_literature",
        "description": ("Semantic + keyword search over abstracts. Use ONLY for open questions "
                        "about findings or evidence - never for counting or listing."),
        "parameters": _p(query={"type": "string"},
                         k={"type": "integer", "default": 8},
                         since_year={"type": "integer"}) | {"required": ["query"]}}},
    {"type": "function", "function": {
        "name": "corpus_stats",
        "description": "What data is loaded: article count, year range, coverage.",
        "parameters": _p()}},
    {"type": "function", "function": {
        "name": "get_articles",
        "description": "Fetch full records by PMID within the concept and inclusive year scope so they can be quoted and cited.",
        "parameters": _p(pmids={"type": "array", "items": {"type": "string"}},
                         concept_id=CID_DESC,
                         since_year={"type": "integer"},
                         until_year={"type": "integer"})
                      | {"required": ["pmids"]}}},
    {"type": "function", "function": {
        "name": "trend_by_year",
        "description": ("Papers per year for a concept. Use for 'is research on X "
                        "growing', 'how has X changed over time', 'trend', 'by year'."),
        "parameters": _p(concept_id=CID_DESC, since_year={"type": "integer"})}},
    {"type": "function", "function": {
        "name": "list_countries",
        "description": ("Which countries publish on a concept, with exact paper counts. "
                        "Use for 'which countries', 'where in the world', 'by country'."),
        "parameters": _p(concept_id=CID_DESC, since_year={"type": "integer"},
                         limit={"type": "integer", "default": 25})}},
    {"type": "function", "function": {
        "name": "list_institutions",
        "description": ("Which universities, hospitals and research centres publish on a "
                        "concept. Use for 'which institutions', 'which universities', "
                        "'which hospitals', 'where is this researched'."),
        "parameters": _p(concept_id=CID_DESC, since_year={"type": "integer"},
                         limit={"type": "integer", "default": 25})}},
    {"type": "function", "function": {
        "name": "top_cited",
        "description": ("Most-cited / most influential papers on a concept, counted "
                        "within this corpus. Use for 'most important papers', 'landmark "
                        "studies', 'most cited', 'key papers'."),
        "parameters": _p(concept_id=CID_DESC, since_year={"type": "integer"},
                         limit={"type": "integer", "default": 20})}},
    {"type": "function", "function": {
        "name": "list_study_types",
        "description": ("Breakdown of study designs - clinical trials, reviews, "
                        "meta-analyses, case reports. Use for 'how many trials', "
                        "'what kind of studies', 'any systematic reviews'."),
        "parameters": _p(concept_id=CID_DESC, since_year={"type": "integer"},
                         limit={"type": "integer", "default": 20})}},
    {"type": "function", "function": {
        "name": "find_trials",
        "description": ("Papers linked to a registered clinical trial, with NCT numbers. "
                        "Use for 'registered trials', 'NCT', 'trial registrations'."),
        "parameters": _p(concept_id=CID_DESC, since_year={"type": "integer"},
                         limit={"type": "integer", "default": 25})}},
    {"type": "function", "function": {
        "name": "compare_concepts",
        "description": ("Compare two topics by volume and overlap. Use whenever the "
                        "question contains 'compare', 'versus', 'vs', or names two "
                        "topics to weigh against each other."),
        "parameters": _p(concept_a=CID_DESC, concept_b=CID_DESC,
                         since_year={"type": "integer"})
                      | {"required": ["concept_a", "concept_b"]}}},
    {"type": "function", "function": {
        "name": "list_papers",
        "description": ("The actual papers behind a count. Use when asked to SHOW or LIST "
                        "the papers themselves, or to see which papers a particular "
                        "journal, drug or author contributed."),
        "parameters": _p(concept_id=CID_DESC,
                         journal={"type": "string", "description": "exact journal abbreviation"},
                         substance_ui={"type": "string"},
                         author_key={"type": "string"},
                         since_year={"type": "integer"},
                         limit={"type": "integer", "default": 50})}},
]

@functools.lru_cache(maxsize=1)
def _searcher():
    from retrieval import HybridSearch
    return HybridSearch()


def search_literature(query: str, k: int = 8, since_year: int | None = None,
                      until_year: int | None = None, concept_id: str | None = None) -> dict:
    """Hybrid BM25 + dense search. Open questions only - never counting."""
    allowed = None
    if concept_id:
        cid, note, error = _coerce_concept(concept_id)
        if error:
            return {"error": error, "results": []}
        allowed = _pmids_for(cid).collect()["pmid"].to_list()
    return _searcher().search(query, k=k, since_year=since_year,
                              until_year=until_year, allowed_pmids=allowed)


REGISTRY = {"resolve_concept": resolve_concept, "spot_concepts": spot_concepts,
            "list_journals": list_journals,
            "list_drugs": list_drugs, "rank_kols": rank_kols,
            "search_literature": search_literature,
            "corpus_stats": corpus_stats, "get_articles": get_articles,
            "trend_by_year": trend_by_year, "list_countries": list_countries,
            "list_institutions": list_institutions, "top_cited": top_cited,
            "list_study_types": list_study_types, "find_trials": find_trials,
            "compare_concepts": compare_concepts, "list_papers": list_papers,
            "article_detail": article_detail}

for _schema in TOOL_SCHEMAS:
    # Every tool exposing a lower year bound also accepts an inclusive upper bound.
    _properties = _schema["function"]["parameters"]["properties"]
    if "since_year" in _properties:
        _properties["until_year"] = {
            "type": "integer", "description": "Inclusive upper publication year",
            "minimum": 1000, "maximum": 3000}
    if _schema["function"]["name"] == "search_literature":
        _properties["concept_id"] = CID_DESC


def register(name: str, fn) -> None:
    REGISTRY[name] = fn


def call(name: str, args: dict) -> Any:
    if name not in REGISTRY:
        return {"error": f"unknown tool {name}", "available": sorted(REGISTRY)}
    try:
        import inspect
        signature = inspect.signature(REGISTRY[name])
        signature.bind(**args)
        validate_years(args.get("since_year"), args.get("until_year"))
        for key, value in args.items():
            if key in ("limit", "k") and (type(value) is not int or not 1 <= value <= 100):
                return {"error": f"{key} must be an integer between 1 and 100"}
            if isinstance(value, str) and len(value) > 4000:
                return {"error": "Argument too long"}
        if name == "get_articles" and (not isinstance(args.get("pmids"), list) or
            len(args["pmids"]) > 50 or any(not isinstance(p, str) or not p.isdigit() for p in args["pmids"])):
            return {"error": "Provide at most 50 numeric PMID strings"}
        return REGISTRY[name](**args)
    except (TypeError, ValueError) as e:
        return {"error": f"bad arguments for {name}: {e}"}


if __name__ == "__main__":
    import json
    print(json.dumps(corpus_stats(), indent=2))
    c = resolve_concept("cisplatin")
    print("\nresolve_concept('cisplatin') ->", json.dumps(c, indent=2))
    if c:
        cid = c[0]["concept_id"]
        print("\nlist_journals ->", json.dumps(list_journals(cid, limit=5), indent=2)[:700])
        print("\nrank_kols ->", json.dumps(rank_kols(cid, limit=3), indent=2)[:900])
