import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from response_cache import ResponseCache


class ResponseCacheTest(unittest.TestCase):
    def test_expiry_and_response_annotations_do_not_contaminate_later_users(self):
        now = [0]
        cache = ResponseCache(ttl=5, clock=lambda: now[0])
        calls = []
        def compute():
            calls.append(True)
            return {"results": [{"papers": 12}]}
        first = cache.get_or_compute("scope", compute)
        first["results"][0]["papers"] = 99
        first["ms"] = 1000
        self.assertEqual(cache.get_or_compute("scope", compute), {"results": [{"papers": 12}]})
        self.assertEqual(len(calls), 1)
        now[0] = 5
        cache.get_or_compute("scope", compute)
        self.assertEqual(len(calls), 2)

    def test_concurrent_same_scope_has_one_computation(self):
        cache = ResponseCache()
        started, release = threading.Event(), threading.Event()
        calls = []
        def compute():
            calls.append(True)
            started.set()
            self.assertTrue(release.wait(5))
            return {"papers": 12}
        with ThreadPoolExecutor(max_workers=4) as pool:
            first = pool.submit(cache.get_or_compute, "scope", compute)
            self.assertTrue(started.wait(5))
            others = [pool.submit(cache.get_or_compute, "scope", compute) for _ in range(3)]
            release.set()
            self.assertEqual([f.result(5) for f in [first, *others]], [{"papers": 12}] * 4)
        self.assertEqual(len(calls), 1)

    def test_failure_can_be_retried_and_error_responses_are_not_retained(self):
        cache = ResponseCache()
        def fail():
            raise ValueError("temporarily unavailable")
        with self.assertRaises(ValueError):
            cache.get_or_compute("scope", fail)
        self.assertEqual(cache.get_or_compute("scope", lambda: {"papers": 1}), {"papers": 1})
        cache.get_or_compute("other", lambda: {"error": "unavailable"})
        self.assertEqual(cache.get_or_compute("other", lambda: {"papers": 2}), {"papers": 2})

    def test_memory_budget_and_lru_eviction(self):
        cache = ResponseCache(max_entries=2, max_bytes=10)
        for key in ["a", "b", "a", "c"]:
            cache.get_or_compute(key, lambda: 1)
        self.assertEqual(list(cache._entries), ["a", "c"])
        cache.get_or_compute("large", lambda: "too large for cache")
        self.assertNotIn("large", cache._entries)
        self.assertLessEqual(cache._bytes, 10)
        disabled = ResponseCache(max_bytes=0)
        self.assertEqual(disabled.get_or_compute("a", lambda: 2), 2)
        self.assertFalse(disabled._entries)

    def test_api_cache_separates_filters_snapshots_and_retrieval_settings(self):
        import dataset
        previous_pin = dataset._PINNED
        try:
            import app
        finally:
            dataset._PINNED = previous_pin
        calls = []
        def trend_by_year(concept_id=None, since_year=None, until_year=None):
            calls.append((concept_id, since_year, until_year))
            return {"results": [], "total_papers": len(calls)}
        trend_by_year.__module__ = "tools"
        with patch.object(app, "_response_cache", ResponseCache()), \
                patch.object(app.tools, "DATASET", {"snapshot": "one"}):
            first, _ = app._timed(trend_by_year, "D1", 2020, until_year=2024)
            first["exact"] = True
            same, _ = app._timed(trend_by_year, concept_id="D1", since_year=2020, until_year=2024)
            self.assertEqual(same["total_papers"], 1)
            self.assertNotIn("exact", same)
            for arguments in [("D2", 2020, 2024), ("D1", 2019, 2024), ("D1", 2020, 2023)]:
                app._timed(trend_by_year, *arguments)
            app.tools.DATASET["snapshot"] = "two"
            app._timed(trend_by_year, "D1", 2020, 2024)
            with patch.dict("os.environ", {"PUBMED_USE_ANN": "0"}):
                app._timed(trend_by_year, "D1", 2020, 2024)
            self.assertEqual(len(calls), 6)

    def test_search_endpoint_returns_hydrated_cards_without_second_article_fetch(self):
        import dataset
        previous_pin = dataset._PINNED
        try:
            import app
        finally:
            dataset._PINNED = previous_pin
        card = {"pmid": "1", "snippet": "Source excerpt", "doi": "10.1/example", "journal": "Journal"}
        with patch.object(app.tools, "search_literature", return_value={"results": [card]}), \
                patch.object(app, "_readiness", return_value={"search_ready": True}), \
                patch.object(app.tools, "get_articles", side_effect=AssertionError("duplicate hydration")):
            result = app.api_search(q="topic", k=20)
            self.assertEqual(result["results"], [card])

    def test_optional_lookup_verification_failure_does_not_disable_search_startup(self):
        import dataset
        previous_pin = dataset._PINNED
        try:
            import app
        finally:
            dataset._PINNED = previous_pin
        with patch("article_lookup.article_lookup_path", side_effect=OSError("unavailable")), \
                patch("analytics_index.citation_counts_path", return_value=None) as citations, \
                patch.object(app, "_searcher") as searcher, \
                patch.object(app, "_search_state", {"warmed": False, "error": None}):
            app._warm()
            citations.assert_called_once()
            searcher.return_value.bm25.warm.assert_called_once()
            self.assertTrue(app._search_state["warmed"])
            self.assertIsNone(app._search_state["error"])


if __name__ == "__main__":
    unittest.main()
