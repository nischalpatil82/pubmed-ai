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
from typing import Any

# Default the store/index locations before tools.py reads them, so the app
# runs with no environment set up.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
os.environ.setdefault("PUBMED_STORE", os.path.join(_ROOT, "store"))
os.environ.setdefault("PUBMED_INDEX", os.path.join(_ROOT, "index"))
sys.path.insert(0, _HERE)

from fastapi import FastAPI, HTTPException, Query          # noqa: E402
from fastapi.responses import FileResponse, JSONResponse   # noqa: E402

import tools                                               # noqa: E402

app = FastAPI(title="PubMed Literature Intelligence", docs_url="/api/docs")

STATIC = os.path.join(_HERE, "static")


# --------------------------------------------------------------- retrieval

_search_state: dict[str, Any] = {"engine": None, "error": None, "loaded": False}


def _searcher():
    """
    Lazy, and never fatal. BM25 alone is a working search engine; the dense
    table may still be building. The UI shows which retrievers are live rather
    than pretending the answer is the same either way.
    """
    if not _search_state["loaded"]:
        _search_state["loaded"] = True
        try:
            from retrieval import HybridSearch
            _search_state["engine"] = HybridSearch()
        except Exception as e:                              # index not built yet
            _search_state["error"] = f"{type(e).__name__}: {e}"
    return _search_state["engine"]


def _timed(fn, *a, **kw):
    t0 = time.time()
    out = fn(*a, **kw)
    return out, round((time.time() - t0) * 1000)


# --------------------------------------------------------------- endpoints

@app.on_event("startup")
def _warm():
    """
    Load the search engine before serving anything.

    Lazily loading it meant the first page hit raced the model: /api/stats
    answered "keyword" while the search that ran a second later used
    "bm25 + vector". The header and the results disagreed on screen, which
    looks like a bug in the numbers rather than a loading order.
    """
    import threading
    threading.Thread(target=_searcher, daemon=True).start()


@app.get("/api/stats")
def api_stats():
    s = tools.corpus_stats()
    eng = _searcher()
    s["retrievers"] = (["bm25"] + (["vector"] if eng and eng.dense else [])) if eng else []
    s["dense_model"] = eng.dense if eng else None
    s["search_error"] = _search_state["error"]
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

    # A multi-word phrase match is precise and stays on top; otherwise the
    # concept with the most coverage wins.
    out = sorted(merged.values(), key=lambda c: (-c["_w"], -(c.get("n_papers") or 0)))
    for c in out:
        c.pop("_w", None)
    return {"query": q, "ms": ms, "concepts": out[:8]}


@app.get("/api/journals")
def api_journals(concept_id: str | None = None, since_year: int | None = None,
                 limit: int = 30, expand: bool = True):
    out, ms = _timed(tools.list_journals, concept_id, since_year, limit, expand)
    out["ms"] = ms
    out["exact"] = True
    return out


@app.get("/api/drugs")
def api_drugs(concept_id: str | None = None, since_year: int | None = None,
              limit: int = 30, named_only: bool = True, expand: bool = True):
    out, ms = _timed(tools.list_drugs, concept_id, since_year, limit, named_only, expand)
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
             since_year: int | None = None, limit: int = 25, expand: bool = True):
    out, ms = _timed(tools.rank_kols, concept_id, country, since_year, limit,
                     1.5, 2.0, 1.0, expand)
    out["ms"] = ms
    out["exact"] = True
    return out


@app.get("/api/search")
def api_search(q: str = Query(..., min_length=2), k: int = 12,
               since_year: int | None = None):
    eng = _searcher()
    if eng is None:
        # "not built" sends people looking for missing files. Usually the files
        # are fine and the embedding model is simply still loading, which takes
        # ~30s after a restart. Say which it is.
        if _search_state["error"]:
            raise HTTPException(503, f"Search unavailable: {_search_state['error']}")
        raise HTTPException(503, "Search is still starting up — the model takes "
                                 "about 30 seconds to load. Try again in a moment.")
    out, ms = _timed(eng.search, q, k, since_year)
    out["ms"] = ms
    out["exact"] = False          # a ranked sample, never a total
    pmids = [r["pmid"] for r in out.get("results", [])]
    if pmids:
        full = {a["pmid"]: a for a in tools.get_articles(pmids)["results"]}
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
def api_trend(concept_id: str | None = None, since_year: int | None = None):
    out, ms = _timed(tools.trend_by_year, concept_id, since_year)
    out["ms"] = ms; out["exact"] = True
    return out


@app.get("/api/countries")
def api_countries(concept_id: str | None = None, since_year: int | None = None,
                  limit: int = 30):
    out, ms = _timed(tools.list_countries, concept_id, since_year, limit)
    out["ms"] = ms; out["exact"] = True
    return out


@app.get("/api/institutions")
def api_institutions(concept_id: str | None = None, since_year: int | None = None,
                     limit: int = 30):
    out, ms = _timed(tools.list_institutions, concept_id, since_year, limit)
    out["ms"] = ms; out["exact"] = True
    return out


@app.get("/api/cited")
def api_cited(concept_id: str | None = None, since_year: int | None = None,
              limit: int = 25):
    out, ms = _timed(tools.top_cited, concept_id, since_year, limit)
    out["ms"] = ms; out["exact"] = True
    return out


@app.get("/api/study_types")
def api_study_types(concept_id: str | None = None, since_year: int | None = None,
                    limit: int = 20):
    out, ms = _timed(tools.list_study_types, concept_id, since_year, limit)
    out["ms"] = ms; out["exact"] = True
    return out


@app.get("/api/trials")
def api_trials(concept_id: str | None = None, since_year: int | None = None,
               limit: int = 30):
    out, ms = _timed(tools.find_trials, concept_id, since_year, limit)
    out["ms"] = ms; out["exact"] = True
    return out


@app.get("/api/compare")
def api_compare(concept_a: str, concept_b: str, since_year: int | None = None):
    out, ms = _timed(tools.compare_concepts, concept_a, concept_b, since_year)
    out["ms"] = ms; out["exact"] = True
    return out


@app.get("/api/papers")
def api_papers(concept_id: str | None = None, journal: str | None = None,
               substance_ui: str | None = None, author_key: str | None = None,
               since_year: int | None = None, limit: int = 50, expand: bool = True):
    out, ms = _timed(tools.list_papers, concept_id, journal, substance_ui,
                     author_key, since_year, limit, expand)
    out["ms"] = ms
    out["exact"] = True
    return out


@app.get("/api/ask")
def api_ask(q: str = Query(..., min_length=3), model: str | None = None):
    """
    Plain-English question -> a written answer over tool results.

    Deliberately NOT called automatically when someone searches. On this CPU a
    single answer takes minutes, and the tabs beside it already give exact
    results in under a second. Generating prose is the slowest, least reliable
    part of the system, so it is opt-in.
    """
    try:
        import agent
    except Exception as e:
        raise HTTPException(503, f"agent unavailable: {e}")
    try:
        out, ms = _timed(agent.run, q, model or agent.MODEL, 6, False)
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
    return JSONResponse({"ok": True})


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

    print(f"store  {os.environ['PUBMED_STORE']}")
    print(f"index  {os.environ['PUBMED_INDEX']}")
    print(f"\n  ->  http://{args.host}:{port}\n", flush=True)
    uvicorn.run(app, host=args.host, port=port, log_level="warning")
