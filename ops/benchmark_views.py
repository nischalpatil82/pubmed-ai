"""Measure overview calculations without loading retrieval or calling an LLM."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="Dataset manifest; defaults to the service's normal selection")
    parser.add_argument("--concept-id")
    parser.add_argument("--since-year", type=int, default=2020)
    parser.add_argument("--until-year", type=int, default=2024)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plans-dir", type=Path,
                        help="Save the actual optimized/unoptimized overview query plans")
    parser.add_argument("--compare-raw", action="store_true",
                        help="Also measure uncached topic lists and raw citation calculation")
    args = parser.parse_args()
    if args.dataset:
        os.environ["PUBMED_DATASET"] = str(Path(args.dataset).resolve())
    sys.path.insert(0, str(ROOT / "pipeline"))
    import app
    import tools
    import article_lookup
    from scope_cache import ScopeCache
    tools.validate_years(args.since_year, args.until_year)
    functions = {
        "trend": (tools.trend_by_year, {}),
        "study_types": (tools.list_study_types, {"limit": 20}),
        "cited": (tools.top_cited, {"limit": 20}),
        "trials": (tools.find_trials, {"limit": 20}),
    }
    report = {"snapshot": tools.DATASET["snapshot"], "scope": {
        "concept_id": args.concept_id, "since_year": args.since_year,
        "until_year": args.until_year}, "panels": {}}
    original_collect = tools._collect_heavy
    plan_number = 0
    def collect_with_plans(lf):
        nonlocal plan_number
        if args.plans_dir:
            args.plans_dir.mkdir(parents=True, exist_ok=True)
            plan_number += 1
            (args.plans_dir / f"query-{plan_number:02d}.txt").write_text(
                "UNOPTIMIZED\n" + lf.explain(optimized=False)
                + "\n\nOPTIMIZED\n" + lf.explain(optimized=True), encoding="utf-8")
        return original_collect(lf)
    if args.plans_dir:
        tools._collect_heavy = collect_with_plans
    start = time.perf_counter()
    article_lookup.article_lookup_path(tools.STORE, tools.INDEX, tools.DATASET["snapshot"])
    tools.citation_counts_path(tools.STORE, tools.INDEX, tools.DATASET["snapshot"])
    report["artifact_verification_ms"] = round((time.perf_counter() - start) * 1000, 3)
    for name, (function, extra) in functions.items():
        filters = dict(concept_id=args.concept_id, since_year=args.since_year,
                       until_year=args.until_year, **extra)
        raw = None
        if args.compare_raw:
            with patch.object(tools, "_scope_cache", ScopeCache(max_bytes=0)), \
                    patch.object(tools, "citation_counts_path", return_value=None), \
                    patch.object(article_lookup, "article_lookup_path", return_value=None):
                start = time.perf_counter()
                raw = function(**filters)
                raw_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        first, _ = app._timed(function, **filters)
        first_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        repeated, _ = app._timed(function, **filters)
        repeat_ms = (time.perf_counter() - start) * 1000
        if first != repeated or first.get("error"):
            raise RuntimeError(f"{name}: failed calculation or repeated result mismatch")
        result = {"first_request_ms": round(first_ms, 3),
                  "repeated_request_ms": round(repeat_ms, 3),
                  "rows": len(first["results"]),
                  "totals": {k: v for k, v in first.items() if k.startswith("total_")}}
        if raw is not None:
            if raw != first:
                raise RuntimeError(f"{name}: raw and optimized result mismatch")
            result.update(raw_request_ms=round(raw_ms, 3), raw_results_equal=True)
        report["panels"][name] = result
        print(name, json.dumps(result), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
