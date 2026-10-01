import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import analytics_index
import article_lookup
import tools
from scope_cache import ScopeCache


class AnalyticsIndexTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store, self.index = self.root / "store", self.root / "index"
        self.store.mkdir()
        self.index.mkdir()
        self.cfg = {"snapshot": "test-snapshot"}
        tables = {
            "articles": pl.DataFrame({"pmid": ["1", "2", "3", "4"],
                                      "title": ["One", "Two", "Three", "Four"],
                                      "pub_year": [2019, 2020, 2024, 2025], "nlm_id": ["N"] * 4}),
            "journals": pl.DataFrame({"nlm_id": ["N"], "medline_ta": ["Journal"]}),
            "mesh_headings": pl.DataFrame({"pmid": ["2", "3", "4"],
                                           "descriptor_ui": ["D0100", "D0200", "D0100"]}),
            "substances": pl.DataFrame({"pmid": ["2"], "substance_ui": ["D0100"]}),
            "keywords": pl.DataFrame({"pmid": ["1", "2"], "term": [" Virus ", "virus"]}),
            "citations": pl.DataFrame({"citing_pmid": ["100", "100", "101", "102", "103", "104", "105"],
                                       "cited_pmid": ["1", "2", "2", "3", "3", "3", "999"]}),
        }
        for name, frame in tables.items():
            frame.write_parquet(self.store / f"{name}.parquet")
        self.patches = [
            patch.multiple(tools, STORE=self.store, INDEX=self.index, DATASET=self.cfg),
            patch.object(tools, "_scope_cache", ScopeCache()),
            patch.object(tools, "_lookup", return_value=pl.DataFrame({"concept_id": ["D0100"], "name_lc": ["virus"]})),
            patch.object(tools, "_descendants", side_effect=lambda cid: ("D0100", "D0200") if cid == "D0100" else (cid,)),
            patch.object(tools, "_coerce_concept", side_effect=lambda cid: (cid, None, None)),
            patch.object(analytics_index, "paths", return_value=(self.store, self.index, self.cfg)),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        analytics_index._validated_path.cache_clear()
        article_lookup._validate.cache_clear()
        article_lookup._data_metadata.cache_clear()
        self.temp.cleanup()

    def build(self):
        return analytics_index.build_citation_counts(reserve_gb=0)

    def test_precomputed_counts_equal_raw_counts_for_all_scopes_and_ties(self):
        # Includes references from citing PMIDs outside the article list and a
        # cited PMID outside the corpus. Scope applies to the cited papers.
        filters = [dict(concept_id=cid, since_year=a, until_year=b, limit=k)
                   for cid in [None, "D0100", "kw:virus", "D0999"]
                   for a, b in [(None, None), (2020, 2024)] for k in [1, 20]]
        raw = [tools.top_cited(**f) for f in filters]
        manifest = self.build()
        self.assertEqual(manifest["source_rows"], 7)
        self.assertEqual([tools.top_cited(**f) for f in filters], raw)
        self.assertTrue(self.build()["reused"])

    def test_membership_reuse_preserves_expansion_keywords_and_years(self):
        with patch.object(tools, "_pmids_query", wraps=tools._pmids_query) as query:
            self.assertEqual(set(tools._pmids_for("D0100").collect()["pmid"]), {"1", "2", "3", "4"})
            for years in [(2020, 2024), (2019, 2020)]:
                tools._scoped("D0100", years[0], until_year=years[1]).collect()
            self.assertEqual(query.call_count, 1)
            self.assertEqual(set(tools._pmids_for("D0100", False).collect()["pmid"]), {"1", "2", "4"})
            self.assertEqual(set(tools._pmids_for("kw:virus").collect()["pmid"]), {"1", "2"})
            self.assertEqual(set(tools._scoped("D0100", 2020, until_year=2024).collect()["pmid"]), {"2", "3"})
            self.cfg["snapshot"] = "another-snapshot"
            tools._pmids_for("D0100").collect()
            self.assertEqual(query.call_count, 4)

    def test_search_reuses_topic_membership_and_retains_year_filters(self):
        with patch.object(tools, "_pmids_query", wraps=tools._pmids_query) as query, \
                patch.object(tools, "_searcher") as searcher:
            searcher.return_value.search.return_value = {"results": []}
            for question in ["virus transmission", "virus vaccines"]:
                tools.search_literature(question, concept_id="D0100", since_year=2020, until_year=2024)
            self.assertEqual(query.call_count, 1)
            call = searcher.return_value.search.call_args
            self.assertEqual(set(call.kwargs["allowed_pmids"]), {"1", "2", "3", "4"})
            self.assertEqual(call.kwargs["since_year"], 2020)
            self.assertEqual(call.kwargs["until_year"], 2024)

    def test_missing_wrong_snapshot_and_corrupted_artifact_fall_back(self):
        expected = tools.top_cited()
        self.build()
        marker = self.index / "analytics" / analytics_index.MARKER
        original = json.loads(marker.read_text())
        artifact = marker.parent / original["file"]
        for damaged in ["[]", "null", "{invalid json"]:
            marker.write_text(damaged)
            self.assertEqual(tools.top_cited(), expected)
        for key, value in [("snapshot", "wrong"), ("source_footer_sha256", "wrong"),
                           ("file", "../outside.parquet"), ("sha256", "wrong")]:
            manifest = {**original, key: value}
            marker.write_text(json.dumps(manifest))
            self.assertEqual(tools.top_cited(), expected)
        marker.write_text(json.dumps(original))
        self.assertIsNotNone(analytics_index.citation_counts_path(self.store, self.index, self.cfg["snapshot"]))
        artifact.write_bytes(b"broken parquet")
        self.assertEqual(tools.top_cited(), expected)

    def test_source_change_invalidates_summary(self):
        self.build()
        source = self.store / "citations.parquet"
        old = pl.read_parquet(source)
        pl.concat([old, pl.DataFrame({"citing_pmid": ["106"], "cited_pmid": ["2"]})]).write_parquet(source)
        self.assertIsNone(analytics_index.citation_counts_path(self.store, self.index, self.cfg["snapshot"]))
        with patch.object(tools, "citation_counts_path", return_value=None):
            expected = tools.top_cited()
        self.assertEqual(tools.top_cited(), expected)

    def test_summary_is_portable_when_files_are_copied(self):
        self.build()
        copy = self.root / "copied"
        shutil.copytree(self.store, copy / "store")
        shutil.copytree(self.index, copy / "index")
        found = analytics_index.citation_counts_path(copy / "store", copy / "index", self.cfg["snapshot"])
        self.assertIsNotNone(found)
        self.assertEqual(found.parent, copy / "index" / "analytics")

    def test_empty_citation_table(self):
        pl.DataFrame(schema={"citing_pmid": pl.String, "cited_pmid": pl.String}).write_parquet(self.store / "citations.parquet")
        self.assertEqual(self.build()["rows"], 0)
        self.assertEqual(tools.top_cited()["results"], [])

    def test_valid_summary_is_included_in_release_and_verified_after_copy(self):
        import manage
        from dataset import atomic_json
        from deploy.fetch_data import verify
        self.build()
        with patch.object(article_lookup, "paths", return_value=(self.store, self.index, self.cfg)):
            article_lookup.build_article_lookup(reserve_gb=0)
        for name in set(manage.SCHEMAS) | {"vocabulary"}:
            target = self.store / f"{name}.parquet"
            if not target.exists():
                pl.DataFrame(schema={"placeholder": pl.String}).write_parquet(target)
        atomic_json(self.store / "manifest.json", self.cfg)
        atomic_json(self.index / "lexical.json", {**self.cfg, "complete": True, "shards": []})
        with patch.object(manage, "paths", return_value=(self.store, self.index, self.cfg)):
            release = manage.release_manifest(self.root / "release.json")
        self.assertTrue(any(path.startswith("index/analytics/citation-counts-") for path in release["files"]))
        self.assertIn("index/analytics/citation-counts.json", release["files"])
        self.assertIn("index/analytics/article-lookup.json", release["files"])
        self.assertTrue(any(path.endswith(".sqlite") for path in release["files"]))
        self.assertTrue(any("article-lookup-" in path and path.endswith(".parquet") for path in release["files"]))
        verify(self.root, release)
        marker = self.index / "analytics" / analytics_index.MARKER
        manifest = json.loads(marker.read_text())
        marker.write_text(json.dumps({**manifest, "snapshot": "stale"}))
        article_marker = self.index / "analytics" / article_lookup.MARKER
        article_manifest = json.loads(article_marker.read_text())
        article_marker.write_text(json.dumps({**article_manifest, "snapshot": "stale"}))
        with patch.object(manage, "paths", return_value=(self.store, self.index, self.cfg)):
            without_stale = manage.release_manifest(self.root / "release.json")
        self.assertFalse(any(path.startswith("index/analytics/") for path in without_stale["files"]))

    def test_trial_totals_and_order_equal_original_query_with_title_loading_after_limit(self):
        pl.DataFrame({"pmid": ["1", "2", "2", "3", "4"], "databank": ["Trial"] * 5,
                      "accession": ["NCT00000001", "NCT00000002", "NCT00000003", "other", "NCT00000004"]})\
            .write_parquet(self.store / "databank_links.parquet")
        for cid in [None, "D0100", "D0999"]:
            scope = tools._scoped(cid, 2020, until_year=2024).select("pmid")
            old = (tools._t("databank_links").filter(pl.col("accession").str.contains(r"^NCT\d{8}$"))
                   .join(scope, on="pmid", how="inner").join(tools._t("articles"), on="pmid", how="left")
                   .select("pmid", "databank", "accession", "title", "pub_year")
                   .sort(["pub_year", "pmid", "accession", "databank"], descending=[True, False, False, False])
                   .collect())
            new = tools.find_trials(cid, 2020, until_year=2024, limit=1)
            self.assertEqual(new["total_links"], old.height)
            self.assertEqual(new["results"], old.head(1).to_dicts())

    def test_indexed_get_articles_preserves_concept_year_scope_and_title_only_records(self):
        source = self.store / "articles.parquet"
        pl.read_parquet(source).with_columns(pl.lit(None, dtype=pl.String).alias("abstract"),
                                            pl.lit(None, dtype=pl.String).alias("doi")).write_parquet(source)
        with patch.object(article_lookup, "paths", return_value=(self.store, self.index, self.cfg)):
            article_lookup.build_article_lookup(reserve_gb=0)
        for cid in [None, "D0100", "D0999"]:
            old = (tools._scoped(cid, 2020, until_year=2024).filter(pl.col("pmid").is_in(["1", "2", "3"]))
                   .join(tools._t("journals"), on="nlm_id", how="left")
                   .select("pmid", "title", "abstract", "pub_year", pl.col("medline_ta").alias("journal"), "doi")
                   .collect().sort("pmid").to_dicts())
            new = tools.get_articles(["1", "2", "3"], cid, 2020, until_year=2024)
            self.assertEqual(sorted(new["results"], key=lambda row: row["pmid"]), old)

    def test_failed_build_keeps_previous_manifest_and_cleans_only_its_temporary_file(self):
        self.build()
        marker = self.index / "analytics" / analytics_index.MARKER
        previous = marker.read_bytes()
        self.cfg["snapshot"] = "new-snapshot"
        with patch.object(pl.LazyFrame, "sink_parquet", side_effect=RuntimeError("interrupted")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.build()
        self.assertEqual(marker.read_bytes(), previous)
        self.assertFalse(list(marker.parent.glob("*.tmp.parquet")))


class ScopeCacheTest(unittest.TestCase):
    def test_memory_limits_and_response_mutation(self):
        frame = pl.DataFrame({"pmid": ["1", "2"]})
        cache = ScopeCache(max_entries=2, max_bytes=frame.estimated_size() * 2)
        result = cache.get_or_compute("a", lambda: frame)
        result.replace_column(0, pl.Series("pmid", ["9", "9"]))
        self.assertEqual(cache.get_or_compute("a", lambda: None)["pmid"].to_list(), ["1", "2"])
        for key in ["b", "a", "c"]:
            cache.get_or_compute(key, lambda: frame)
        self.assertEqual(list(cache._entries), ["a", "c"])
        cache.get_or_compute("large", lambda: pl.DataFrame({"pmid": ["x" * 100]}))
        self.assertNotIn("large", cache._entries)
        self.assertLessEqual(cache._bytes, cache.max_bytes)

    def test_concurrent_membership_computation_and_retry(self):
        cache = ScopeCache()
        calls, started, release = [], threading.Event(), threading.Event()
        def compute():
            calls.append(True)
            started.set()
            if not release.wait(5):
                raise RuntimeError("wait timed out")
            return pl.DataFrame({"pmid": ["1"]})
        with ThreadPoolExecutor(max_workers=4) as pool:
            first = pool.submit(cache.get_or_compute, "a", compute)
            self.assertTrue(started.wait(5))
            rest = [pool.submit(cache.get_or_compute, "a", compute) for _ in range(3)]
            release.set()
            self.assertEqual([f.result(5)["pmid"].to_list() for f in [first, *rest]], [["1"]] * 4)
        self.assertEqual(len(calls), 1)
        def fail():
            raise ValueError("temporary failure")
        with self.assertRaises(ValueError):
            cache.get_or_compute("b", fail)
        self.assertEqual(cache.get_or_compute("b", lambda: pl.DataFrame({"pmid": ["2"]}))["pmid"].to_list(), ["2"])


if __name__ == "__main__":
    unittest.main()
