"""Verify indexed hydration and lexical shortcuts preserve records/ranking/counts."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import article_lookup
from retrieval import BM25Search, HybridSearch, _top_indices


class ArticleLookupTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store, self.index = self.root / "store", self.root / "index"
        self.store.mkdir()
        self.index.mkdir()
        self.cfg = {"snapshot": "test-snapshot"}
        self.rows = pl.DataFrame({"pmid": ["3", "1", "2"], "title": ["Unicode α", "Title", "Title only"],
                                  "abstract": ["A quoted ' finding.", "Late result", None],
                                  "pub_year": [2024, 2019, 2020], "nlm_id": ["N"] * 3,
                                  "doi": [None, "10/example", None], "source_member": ["a.xml"] * 3})
        self.rows.write_parquet(self.store / "articles.parquet", row_group_size=1)
        self.selection = patch.object(article_lookup, "paths", return_value=(self.store, self.index, self.cfg))
        self.selection.start()

    def tearDown(self):
        self.selection.stop()
        article_lookup._validate.cache_clear()
        article_lookup._data_metadata.cache_clear()
        self.temp.cleanup()

    def get(self, ids, columns=None):
        return article_lookup.article_records(self.store, self.index, self.cfg["snapshot"], ids, columns)

    def build(self):
        return article_lookup.build_article_lookup(reserve_gb=0)

    def test_lookup_equals_source_for_projection_order_nulls_and_missing_ids(self):
        ids = ["2", "3", "2", "absent", "1", "' OR 1=1 --"]
        raw = self.get(ids)
        self.build()
        with patch.object(pl, "scan_parquet", side_effect=AssertionError("should not scan source")):
            self.assertTrue(self.get(ids).equals(raw))
            self.assertTrue(self.get(ids, ["pmid", "abstract"]).equals(raw.select("pmid", "abstract")))
            self.assertEqual(self.get([]).height, 0)
            self.assertEqual(self.get(["absent"]).height, 0)
        self.assertTrue(self.build()["reused"])

    def test_invalid_manifest_or_artifact_uses_source(self):
        expected = self.get(["1", "2"])
        manifest = self.build()
        marker = self.index / "analytics" / article_lookup.MARKER
        for value in [[], None, {**manifest, "snapshot": "wrong"},
                      {**manifest, "file": "../outside.sqlite"}, {**manifest, "sha256": "bad"}]:
            marker.write_text(json.dumps(value))
            self.assertTrue(self.get(["1", "2"]).equals(expected))
        marker.write_text(json.dumps(manifest))
        target = marker.parent / manifest["file"]
        target.write_bytes(b"bad sqlite")
        self.assertTrue(self.get(["1", "2"]).equals(expected))

    def test_source_change_and_failed_build_do_not_publish_stale_records(self):
        self.build()
        changed = self.rows.with_columns(pl.lit("Changed").alias("title"))
        changed.write_parquet(self.store / "articles.parquet")
        self.assertEqual(self.get(["1"])["title"][0], "Changed")
        marker = self.index / "analytics" / article_lookup.MARKER
        old = marker.read_bytes()
        with patch.object(article_lookup.pq, "ParquetWriter", side_effect=RuntimeError("interrupted")):
            with self.assertRaises(RuntimeError):
                self.build()
        self.assertEqual(marker.read_bytes(), old)
        self.assertFalse(list(marker.parent.glob("*.tmp.sqlite*")))

    def test_bound_variables_are_batched_and_connections_release_windows_handles(self):
        rows = pl.DataFrame({"pmid": [str(i) for i in range(1100)], "title": ["t"] * 1100})
        rows.write_parquet(self.store / "articles.parquet")
        manifest = self.build()
        self.assertTrue(self.get(rows["pmid"].to_list()).equals(rows))
        artifact = self.index / "analytics" / manifest["file"]
        moved = artifact.with_suffix(".moved")
        artifact.rename(moved)
        moved.rename(artifact)

    def test_empty_article_source(self):
        self.rows.head(0).write_parquet(self.store / "articles.parquet")
        self.assertEqual(self.build()["rows"], 0)
        self.assertEqual(self.get(["1"]).height, 0)

    def test_search_hydrates_returned_papers_and_preserves_rerank_and_missing_candidates(self):
        search = HybridSearch.__new__(HybridSearch)
        search.store, search.index, search.dataset = self.store, self.index, self.cfg
        search.reranker = None
        with patch("retrieval.article_records", wraps=article_lookup.article_records) as lookup:
            self.assertEqual(search._article_candidates(["3", "1", "2"], 1)["pmid"].to_list(), ["3"])
            self.assertEqual(lookup.call_args.args[3], ["3"])
            self.assertEqual(set(search._article_candidates(["absent", "1", "2"], 1)["pmid"]), {"1", "2"})
            search.reranker = object()
            self.assertEqual(search._article_candidates(["3", "1", "2"], 1).height, 3)
            self.assertEqual(lookup.call_args.args[3], ["3", "1", "2"])


class LexicalShortcutTest(unittest.TestCase):
    def test_partition_top_k_matches_full_stable_sort_including_boundary_ties(self):
        random = np.random.default_rng(2026)
        for size in [0, 1, 10, 1000]:
            scores = random.integers(0, 5, size).astype(float)
            eligible = np.flatnonzero(scores > 0)
            for k in [0, 1, 7, 50, 2000]:
                expected = eligible[np.argsort(-scores[eligible], kind="stable")[:k]]
                np.testing.assert_array_equal(_top_indices(scores, eligible, k), expected)

    def test_filter_empty_shards_are_not_loaded_or_scored_and_counts_remain_exact(self):
        search = BM25Search.__new__(BM25Search)
        search.cfg = {"shards": ["old", "current", "other"]}
        search.module = SimpleNamespace(tokenize=lambda *a, **kw: [["query"]])
        search.stemmer = None
        search._loaded, search._metadata = {}, {}
        metadata = {
            "old": pl.DataFrame({"pmid": ["1", "2"], "pub_year": [2000, None]}),
            "current": pl.DataFrame({"pmid": ["3", "4", "5"], "pub_year": [2020, 2024, 2025]}),
            "other": pl.DataFrame({"pmid": ["6"], "pub_year": [2022]}),
        }
        scores = np.array([1.0, 1.0, 50.0])
        retriever = SimpleNamespace(get_scores=lambda tokens: scores)
        with patch.object(search, "_load_meta", side_effect=lambda name: metadata[name]), \
                patch.object(search, "_load_shard", return_value=(retriever, metadata["current"])) as load:
            hits, count = search.search_with_count("query", k=1, since_year=2020,
                                                   until_year=2024, allowed_pmids={"3", "4"})
            load.assert_called_once_with("current")
            self.assertEqual(count, 2)
            self.assertEqual([row["pmid"] for row in hits], ["3"])
            load.reset_mock()
            self.assertEqual(search.search_with_count("query", allowed_pmids=[]), ([], 0))
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
