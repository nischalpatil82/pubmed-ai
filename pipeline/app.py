#!/usr/bin/env python3
"""
Web front end for the PubMed literature store.

Serves the same tool functions the agent calls, so the UI and the model see
identical numbers. Counting endpoints (journals / drugs / researchers) return
exact totals over the whole corpus; /api/search is the only endpoint that
returns a ranked sample, and it says so.

    pip install fastapi "uvicorn[standard]"
    set PUBMED_STORE=...\store
    set PUBMED_INDEX=...\index
    python app.py            ->  http://127.0.0.1:8000
"""
from __future__ import annotations

import os
import sys
import time
import threading
from typing import Any

# Default the store/index locations before tools.py reads them, so the app
# runs with no environment set up.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
from dataset import pin
pin()

from fastapi import FastAPI, HTTPException, Query          # noqa: E402
from fastapi.responses import FileResponse, JSONResponse   # noqa: E402

import tools                                               # noqa: E402

app = FastAPI(title="PubMed Literature Intelligence", docs_url="/api/docs")

STATIC = os.path.join(_HERE, "static")


# --------------------------------------------------------------- retrieval

_search_state: dict[str, Any] = {
    "engine": None, "error": None, "loaded": False, "warmed": False,
}
_search_lock = threading.Lock()


def _searcher():
    """Load once without making initialization failure fatal to the API."""
    with _search_lock:
        if not _search_state["loaded"]:
            _search_state["warmed"] = False
            try:
                _search_state["engine"] = tools._searcher()
                _search_state["error"] = None
            except Exception as e:
                _search_state["error"] = f"{type(e).__name__}: {e}"
            _search_state["loaded"] = True
    return _search_state["engine"]


def _validate_years(since_year: int | None, until_year: int | None):
    """Reject invalid bounds before invoking a tool or model."""
    try:
        tools.validate_years(since_year, until_year)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


def _readiness() -> dict[str, Any]:
    """Observe loaded state and the manifest; never load or query a model."""
    from dataset import paths, read_json

    state = dict(_search_state)
    eng = state["engine"]
    cfg = None
    manifest_ready = False
    manifest_error = None
    try:
        _, index, cfg = paths()
        lexical = read_json(index / "lexical.json")
        manifest_ready = (lexical.get("complete") is True
                          and lexical.get("snapshot") == cfg["snapshot"])
        if not manifest_ready:
            manifest_error = "Lexical index is incomplete or from another snapshot"
    except Exception as e:
        manifest_error = f"Lexical manifest unavailable: {type(e).__name__}: {e}"

    search_ready = bool(state["loaded"] and eng is not None
                        and getattr(eng, "bm25", None) is not None
                        and state["warmed"] and not state["error"]
                        and manifest_ready)
    vector_status = getattr(eng, "vector_status", "absent")
    vector_error = getattr(eng, "dense_error", None)
    dense_model = getattr(eng, "dense", None)
    dense_ready = bool(search_ready and dense_model
                       and vector_status == "complete" and not vector_error)
    search_error = state["error"] or manifest_error
    if not search_ready and not search_error:
        search_error = ("Search engine is not loaded" if eng is None or not state["loaded"]
                        else "Lexical search has not warmed successfully")
    return {
        "ok": search_ready, "dataset": cfg, "search_ready": search_ready,
        "dense_ready": dense_ready,
        "retrievers": (["bm25"] + (["vector"] if dense_ready else [])) if search_ready else [],
        "degraded": search_ready and not dense_ready,
        "dense_model": dense_model if dense_ready else None,
        "vector_status": vector_status, "vector_error": vector_error,
        "search_error": search_error,
    }


def _timed(fn, *a, **kw):
    t0 = time.time()
    out = fn(*a, **kw)
    return out, round((time.time() - t0) * 1000)


# --------------------------------------------------------------- endpoints

@app.on_event("startup")
def _warm():
    """Warm lexical shards; retain failures so health can report unavailability."""
    _search_state["warmed"] = False
    engine = _searcher()
    if engine is not None:
        try:
            engine.bm25.warm()
        except Exception as e:
            _search_state["error"] = f"{type(e).__name__}: {e}"
        else:
            _search_state["error"] = None
            _search_state["warmed"] = True


@app.get("/api/stats")
def api_stats():
    s = tools.corpus_stats()
    s.update(_readiness())
    try:
        import agent
        s["llm"] = agent.llm_status()
    except Exception as e:
        s["llm"] = {"configured": False, "provider": "unavailable", "model": None,
                    "privacy": "Answer generation is unavailable.",
                    "error": f"{type(e).__name__}: {e}"}
    return s


@app.get("/api/resolve")
def api_resolve(q: str = Query(..., min_length=1)):
    """
    Free text -> curated concepts. spot_concepts reads the whole sentence and
    prefers the longest match, which is what makes 'KOLs for cisplatin in
    ovarian cancer' resolve to two concepts instead of guessing at one.
    """
    # Use BOTH matchers and merge, because each is weak where the other is
    # strong. spot_concepts scans a sentence for several concepts, but it only
    # matches whole n-grams: typing "alzheimer" found an author keyword on 31
    # papers and missed the MeSH concept Alzheimer Disease on 1,351, because
    # the descriptor is named "alzheimer disease" and never matches the single
    # word. resolve_concept does substring matching and finds it immediately.
    spot, ms = _timed(tools.spot_concepts, q, limit=8)
    direct = tools.resolve_concept(q, limit=8)

    merged: dict[str, dict] = {}
    for c in spot:
        merged[c["concept_id"]] = {**c, "_w": min(c.get("words", 1), 2)}
    for c in direct:
        if c["concept_id"] not in merged:
            merged[c["concept_id"]] = {**c, "_w": 1}

    # Literal matching alone cannot infer that a person searching the disease
    # may also want to inspect its causative virus.  Add only vetted, distinct
    # related concepts that exist in this snapshot.  They are labelled in the
    # UI and remain separate selectable scopes.
    for c in tools.related_concepts(list(merged), limit=3):
        parent_id = c.pop("related_to")
        # Keep the concept the user named first, immediately followed by its
        # curated related concept, before less-specific keyword alternatives.
        merged[parent_id]["_relationship_rank"] = 2
        if c["concept_id"] not in merged:
            merged[c["concept_id"]] = {**c, "_w": 1, "_relationship_rank": 1}

    # A multi-word phrase match is precise and stays on top; otherwise the
    # concept with the most coverage wins.
    out = sorted(merged.values(), key=lambda c: (-c.get("_relationship_rank", 0), -c["_w"],
                                                  -(c.get("n_papers") or 0)))
    for c in out:
        c.pop("_w", None)
        c.pop("_relationship_rank", None)
    return {"query": q, "ms": ms, "concepts": out[:8]}


@app.get("/api/journals")
def api_journals(concept_id: str | None = None, since_year: int | None = None,
                 limit: int = 30, expand: bool = True, until_year: int | None = None):
    _validate_years(since_year, until_year)
    out, ms = _timed(tools.list_journals, concept_id, since_year, limit, expand,
                     until_year=until_year)
    out["ms"] = ms
    out["exact"] = True
    return out


@app.get("/api/drugs")
def api_drugs(concept_id: str | None = None, since_year: int | None = None,
              limit: int = 30, named_only: bool = True, expand: bool = True,
              until_year: int | None = None):
    _validate_years(since_year, until_year)
    out, ms = _timed(tools.list_drugs, concept_id, since_year, limit, named_only, expand,
                     until_year=until_year)
    out["ms"] = ms
    out["exact"] = True
    # Be explicit that the default filter hides class terms without a registry
    # number - roughly two thirds of substance rows in this corpus.
    out["filter_note"] = ("Showing named compounds with a UNII/CAS registry number. "
                          "Turn off 'named only' to include MeSH class terms."
                          if named_only else
                          "Including MeSH class terms (e.g. 'Antioxidants').")
    return out


@app.get("/api/kols")
def api_kols(concept_id: str | None = None, country: str | None = None,
             since_year: int | None = None, limit: int = 25, expand: bool = True,
             until_year: int | None = None):
    _validate_years(since_year, until_year)
    out, ms = _timed(tools.rank_kols, concept_id, country, since_year, limit,
                     1.5, 2.0, 1.0, expand, until_year=until_year)
    out["ms"] = ms
    out["exact"] = True
    return out


@app.get("/api/search")
def api_search(q: str = Query(..., min_length=2, max_length=4000), k: int = Query(12, ge=1, le=50),
               since_year: int | None = None, until_year: int | None = None,
               concept_id: str | None = None):
    _validate_years(since_year, until_year)
    status = _readiness()
    if not status["search_ready"]:
        raise HTTPException(503, f"Search unavailable: {status['search_error']}")
    out, ms = _timed(tools.search_literature, q, k, since_year,
                     until_year=until_year, concept_id=concept_id)
    out["ms"] = ms
    out["exact"] = False          # a ranked sample, never a total
    pmids = [r["pmid"] for r in out.get("results", [])]
    if pmids:
        full = {a["pmid"]: a for a in tools.get_articles(
            pmids, concept_id=concept_id, since_year=since_year,
            until_year=until_year)["results"]}
        for r in out["results"]:
            a = full.get(r["pmid"], {})
            abstract = a.get("abstract") or ""
            r["snippet"] = abstract[:320] + ("…" if len(abstract) > 320 else "")
            r["doi"] = a.get("doi")
            r["journal"] = a.get("journal") or r.get("medline_ta")
    return out


# Everything the model can call, the interface can call too. A capability that
# exists only behind the LLM is a capability most users never find.

@app.get("/api/trend")
def api_trend(concept_id: str | None = None, since_year: int | None = None,
              until_year: int | None = None):
    _validate_years(since_year, until_year)
    out, ms = _timed(tools.trend_by_year, concept_id, since_year, until_year=until_year)
    out["ms"] = ms; out["exact"] = True
    return out


@app.get("/api/countries")
def api_countries(concept_id: str | None = None, since_year: int | None = None,
                  limit: int = 30, until_year: int | None = None):
    _validate_years(since_year, until_year)
    out, ms = _timed(tools.list_countries, concept_id, since_year, limit,
                     until_year=until_year)
    out["ms"] = ms; out["exact"] = True
    return out


@app.get("/api/institutions")
def api_institutions(concept_id: str | None = None, since_year: int | None = None,
                     limit: int = 30, until_year: int | None = None):
    _validate_years(since_year, until_year)
    out, ms = _timed(tools.list_institutions, concept_id, since_year, limit,
                     until_year=until_year)
    out["ms"] = ms; out["exact"] = True
    return out


@app.get("/api/cited")
def api_cited(concept_id: str | None = None, since_year: int | None = None,
              limit: int = 25, until_year: int | None = None):
    _validate_years(since_year, until_year)
    out, ms = _timed(tools.top_cited, concept_id, since_year, limit,
                     until_year=until_year)
    out["ms"] = ms; out["exact"] = True
    return out


@app.get("/api/study_types")
def api_study_types(concept_id: str | None = None, since_year: int | None = None,
                    limit: int = 20, until_year: int | None = None):
    _validate_years(since_year, until_year)
    out, ms = _timed(tools.list_study_types, concept_id, since_year, limit,
                     until_year=until_year)
    out["ms"] = ms; out["exact"] = True
    return out


@app.get("/api/trials")
def api_trials(concept_id: str | None = None, since_year: int | None = None,
               limit: int = 30, until_year: int | None = None):
    _validate_years(since_year, until_year)
    out, ms = _timed(tools.find_trials, concept_id, since_year, limit,
                     until_year=until_year)
    out["ms"] = ms; out["exact"] = True
    return out


@app.get("/api/compare")
def api_compare(concept_a: str, concept_b: str, since_year: int | None = None,
                until_year: int | None = None):
    _validate_years(since_year, until_year)
    out, ms = _timed(tools.compare_concepts, concept_a, concept_b, since_year,
                     until_year=until_year)
    out["ms"] = ms; out["exact"] = True
    return out


@app.get("/api/papers")
def api_papers(concept_id: str | None = None, journal: str | None = None,
               substance_ui: str | None = None, author_key: str | None = None,
               since_year: int | None = None, limit: int = 50, expand: bool = True,
               until_year: int | None = None):
    _validate_years(since_year, until_year)
    out, ms = _timed(tools.list_papers, concept_id, journal, substance_ui,
                     author_key, since_year, limit, expand, until_year=until_year)
    out["ms"] = ms
    out["exact"] = True
    return out


@app.get("/api/ask")
def api_ask(q: str = Query(..., min_length=3, max_length=4000), model: str | None = None,
            concept_id: str | None = None, since_year: int | None = None,
            until_year: int | None = None):
    """
    Plain-English question -> a written answer over tool results.

    Deliberately NOT called automatically when someone searches. On this CPU a
    single answer takes minutes, and the tabs beside it already give exact
    results in under a second. Generating prose is the slowest, least reliable
    part of the system, so it is opt-in.
    """
    _validate_years(since_year, until_year)
    filters = {k: v for k, v in (("concept_id", concept_id),
                                 ("since_year", since_year),
                                 ("until_year", until_year)) if v is not None}
    try:
        import agent
    except Exception as e:
        raise HTTPException(503, f"agent unavailable: {e}")
    try:
        out, ms = _timed(agent.run, q, model or agent.MODEL, 6, False,
                         filters=filters or None)
    except Exception as e:
        raise HTTPException(503, f"model call failed: {type(e).__name__}: {e}")
    # The trace holds full tool payloads; the page does not need them.
    out.pop("trace", None)
    out["ms"] = ms
    # agent.run knows which model actually answered - do not overwrite it with
    # the local default, which reported "qwen2.5:7b" for cloud answers.
    out.setdefault("model", model or agent.MODEL)
    return out


@app.get("/api/article/{pmid}")
def api_article(pmid: str):
    out, ms = _timed(tools.article_detail, pmid)
    if "error" in out:
        raise HTTPException(404, out["error"])
    out["ms"] = ms
    return out


# --------------------------------------------------------------- static

@app.get("/")
def index():
    # No-cache, because this page is edited constantly during development and a
    # browser holding an old copy looks exactly like a fix that did not work.
    # Costs nothing here: the file is one small local request.
    return FileResponse(os.path.join(STATIC, "index.html"), headers={
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
    })


@app.get("/health")
def health():
    status = _readiness()
    return JSONResponse(status, status_code=200 if status["search_ready"] else 503)


if __name__ == "__main__":
    import argparse
    import socket

    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    # Port 8000 is a popular default; fall forward rather than dying with a
    # bind error that looks like an application fault.
    port = args.port
    for candidate in range(args.port, args.port + 12):
        with socket.socket() as s:
            if s.connect_ex((args.host, candidate)) != 0:
                port = candidate
                break
    else:
        sys.exit(f"no free port in {args.port}-{args.port + 11}")

    print(f"store  {tools.STORE}")
    print(f"index  {tools.INDEX}")
    print(f"\n  ->  http://{args.host}:{port}\n", flush=True)
    uvicorn.run(app, host=args.host, port=port, log_level="warning")
