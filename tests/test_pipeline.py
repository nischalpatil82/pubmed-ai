import importlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from stream_store import build, finalize_owner_partitioned
from dataset import paths, atomic_json
from evidence import (RequestState, bounded_payload, render_quotes, render_table,
                      render_grounded_claims)
from retrieval import build_bm25, BM25Search, HybridSearch, token_passages, build_vectors, lexical_passages


def article(pmid, title="Target", abstract="An abstract result.", year=2025, child=""):
    ab = f"<Abstract><AbstractText>{abstract}</AbstractText></Abstract>" if abstract else ""
    return f'<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article><ArticleTitle>{title}</ArticleTitle><Journal><JournalIssue><PubDate><Year>{year}</Year></PubDate></JournalIssue></Journal>{ab}</Article>{child}</MedlineCitation></PubmedArticle>'


class Tokenizer:
    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]}


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.env = patch.dict(os.environ, {"PUBMED_DATASET": str(self.data / "dataset.json")})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def ingest(self, members):
        source = self.root / "source.zip"
        with zipfile.ZipFile(source, "w") as z:
            for name, body in members:
                z.writestr(name, "<PubmedArticleSet>" + body + "</PubmedArticleSet>")
        return build(source, self.data, batch=2, buckets=2, reserve_gb=0)

    def test_partitioned_finalization_deduplicates_without_global_collect(self):
        parts = self.root / "parts"
        parts.mkdir()
        schema = {"citing_pmid": pl.String, "cited_pmid": pl.String}
        pl.DataFrame([("1", "2"), ("1", "2")], schema=schema, orient="row").write_parquet(
            parts / "citations-0000.parquet"
        )
        pl.DataFrame([("3", "4")], schema=schema, orient="row").write_parquet(
            parts / "citations-0001.parquet"
        )
        destination = self.root / "citations.parquet"
        rows = finalize_owner_partitioned(
            "citations", sorted(parts.glob("citations-*.parquet")), destination
        )
        self.assertEqual(rows, 2)
        self.assertEqual(pl.read_parquet(destination).height, 2)

    def test_revisions_replace_children_and_deletions_can_reintroduce(self):
        old = '<KeywordList><Keyword>obsolete</Keyword></KeywordList>'
        self.ingest([("pubmed002.xml", article(1, "Updated") + '<DeleteCitation><PMID>2</PMID></DeleteCitation>' + article(3, "Returns")),
                     ("pubmed001.xml", article(1, "Old", child=old) + article(2) + article(3) + '<DeleteCitation><PMID>3</PMID></DeleteCitation>')])
        store, _, cfg = paths()
        arts = pl.read_parquet(store / "articles.parquet")
        self.assertEqual(set(arts["pmid"]), {"1", "3"})
        self.assertEqual(arts.filter(pl.col("pmid") == "1")["title"][0], "Updated")
        self.assertEqual(pl.read_parquet(store / "keywords.parquet").height, 0)
        self.assertEqual(pl.read_parquet(store / "deleted_pmids.parquet")["pmid"].to_list(), ["2"])
        self.assertIn("source_member", arts.columns)
        before = (store / "articles.parquet").stat().st_mtime_ns
        build(self.root / "source.zip", self.data, batch=2, buckets=2, reserve_gb=0)
        self.assertEqual(before, (store / "articles.parquet").stat().st_mtime_ns)

    def test_invalid_member_not_committed(self):
        with self.assertRaises(Exception):
            self.ingest([("bad.xml", article(1) + "<broken>")])
        self.assertFalse((self.data / "dataset.json").exists())

    def test_title_only_and_filter_before_top_k(self):
        self.ingest([("a.xml", article(1, "Zebrafish zebrafish", year=2000) + article(2, "Zebrafish", abstract=None))])
        build_bm25(shard_size=1)
        searcher = BM25Search()
        rows = searcher.search("zebrafish", k=1, since_year=2020)
        self.assertEqual(rows[0]["pmid"], "2")
        self.assertFalse(rows[0]["has_abstract"])
        counted, total = searcher.search_with_count("zebrafish", k=1, since_year=2020)
        self.assertEqual([row["pmid"] for row in counted], ["2"])
        self.assertEqual(total, 1)
        self.assertEqual(searcher.search("unfindableword"), [])
        _, index, _ = paths()
        atomic_json(index / "vector_model.json", {"complete": False})
        result = HybridSearch().search("zebrafish", since_year=2020)
        self.assertEqual(result["retrievers"], ["bm25"])
        self.assertEqual(result["results"][0]["evidence"][0]["section"], "title")

    def test_partial_lexical_refused(self):
        self.ingest([("a.xml", article(1) + article(2))])
        build_bm25(limit=1)
        with self.assertRaises(RuntimeError):
            BM25Search()

    def test_chunks_keep_tail_and_token_bound(self):
        body = " ".join(f"term{i}" for i in range(1300)) + " Conclusion."
        chunks = token_passages(body, Tokenizer(), 200, 40)
        self.assertTrue(chunks[-1].endswith("Conclusion."))
        self.assertTrue(all(len(c.split()) <= 200 for c in chunks))
        for term in body.split():
            self.assertTrue(any(term in c for c in chunks))

    def test_keyword_evidence_preserves_late_matching_excerpt(self):
        body = "background " * 1500 + "Late conclusion reports hypoglycemia."
        pieces = lexical_passages("hypoglycemia", {"pmid": "1", "abstract": body, "title": "Study"})
        self.assertIn("hypoglycemia", pieces[0]["text"])
        self.assertTrue(all(piece["text"] in body for piece in pieces))

    def test_covid_and_sars_are_related_but_remain_separate_scopes(self):
        import tools
        vocab = pl.DataFrame({
            "concept_id": ["D000086382", "D000086402"],
            "concept_name": ["COVID-19", "SARS-CoV-2"],
            "kind": ["mesh", "mesh"],
            "code": [None, None],
            "n_papers": [100, 75],
            "name_lc": ["covid-19", "sars-cov-2"],
        })
        with patch("tools._vocab", return_value=vocab):
            sars = tools.related_concepts(["D000086382"])
            covid = tools.related_concepts(["D000086402"])
        self.assertEqual(sars[0]["concept_id"], "D000086402")
        self.assertEqual(sars[0]["concept_name"], "SARS-CoV-2")
        self.assertEqual(sars[0]["kind"], "related mesh")
        self.assertEqual(covid[0]["concept_id"], "D000086382")

    def test_release_checksum_and_traversal_rejection(self):
        import hashlib
        import importlib.util
        script = Path(__file__).resolve().parents[1] / "deploy" / "fetch_data.py"
        spec = importlib.util.spec_from_file_location("release_verifier", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        (self.root / "store").mkdir()
        file = self.root / "store" / "test.parquet"
        file.write_bytes(b"original")
        manifest = {"files": {"store/test.parquet": {"bytes": 8, "sha256": hashlib.sha256(b"original").hexdigest()}}}
        module.verify(self.root, manifest)
        file.write_bytes(b"modified")
        with self.assertRaises(ValueError):
            module.verify(self.root, manifest)
        with self.assertRaises(ValueError):
            module.verify(self.root, {"files": {"../outside": {"bytes": 0, "sha256": ""}}})

    def test_source_snapshot_changes(self):
        first = self.ingest([("a.xml", article(1))])
        second = self.ingest([("a.xml", article(1, "Changed"))])
        self.assertNotEqual(first["snapshot"], second["snapshot"])
        self.assertTrue((self.data / first["store"] / "articles.parquet").exists())

    def test_answer_guards_and_zero(self):
        records = {"12345": ["The study reported 7 participants with improved outcomes."]}
        with self.assertRaises(ValueError):
            render_quotes({"evidence": [{"pmid": "12345", "quote": "The study reported 8 participants with improved outcomes."}]}, records)
        self.assertIn("PMID 12345", render_quotes({"evidence": [{"pmid": "12345", "quote": records["12345"][0]}]}, records))
        self.assertIn("0", render_table({"total_papers": 0, "results": []}))
        payload = json.loads(bounded_payload({"total_papers": 12, "results": [{"abstract": "x" * 1000}]}, 100))
        self.assertEqual(payload["total_papers"], 12)
        self.assertEqual(payload["omitted_rows"], 1)
        state = RequestState("snapshot")
        state.execute("count", {}, lambda *args: {"total_papers": 0})
        with self.assertRaises(ValueError):
            state.execute("count", {}, lambda *args: {})
        grounded = render_grounded_claims(
            {"claims": [{"text": "Fatigue persisted for some participants after infection.",
                          "source_ids": [1]}]},
            [{"id": 1, "pmid": "12345",
              "quote": "Participants reported persistent fatigue after COVID-19."}])
        self.assertIn("Fatigue persisted", grounded)
        self.assertIn("PMID 12345", grounded)
        with self.assertRaises(ValueError):
            render_grounded_claims(
                {"claims": [{"text": "Unsupported source must be rejected by the renderer.",
                              "source_ids": [2]}]},
                [{"id": 1, "pmid": "12345", "quote": records["12345"][0]}])

    def test_vector_resume_after_partial_write_is_idempotent(self):
        import numpy as np
        import lancedb
        from types import SimpleNamespace

        class Model:
            tokenizer = Tokenizer()
            max_seq_length = 64
            def __getitem__(self, key):
                return SimpleNamespace(auto_model=SimpleNamespace(config=SimpleNamespace(_commit_hash="test-revision")))
            def encode(self, texts, **kwargs):
                return np.array([[0.6, 0.8, 0.0] for _ in texts])

        self.ingest([("a.xml", article(1) + article(2) + article(3))])
        with patch("retrieval._encoder", return_value=Model()):
            build_vectors("test-model", limit=1, batch=1, ann=False)
            _, index, _ = paths()
            cfg = json.loads((index / "vector_model.json").read_text())
            self.assertFalse(cfg["complete"])
            marker = index / (cfg["table"] + ".json")
            state = json.loads(marker.read_text())
            state["batches"] = {}  # Simulate write succeeding immediately before process failure.
            atomic_json(marker, state)
            build_vectors("test-model", batch=1, ann=False)
            cfg = json.loads((index / "vector_model.json").read_text())
            self.assertTrue(cfg["complete"])
            self.assertEqual(cfg["docs"], 6)
            table = lancedb.connect(str(index)).open_table(cfg["table"])
            self.assertEqual(table.count_rows(), 6)
        build_bm25()
        with patch("retrieval._encoder", side_effect=OSError("Model unavailable")):
            result = HybridSearch().search("target")
            self.assertEqual(result["vector_status"], "unavailable")
            self.assertEqual(result["retrievers"], ["bm25"])
            self.assertTrue(result["results"])

    def test_two_embedding_workers_finalize_external_passages(self):
        import lancedb
        from types import SimpleNamespace
        from embedding_shards import run_worker, verify, finalize

        class Model:
            tokenizer = Tokenizer()
            max_seq_length = 64
            device = "test-gpu"
            def __getitem__(self, key):
                return SimpleNamespace(auto_model=SimpleNamespace(config=SimpleNamespace(_commit_hash="worker-revision")))
            def get_sentence_embedding_dimension(self):
                return 3
            def encode(self, texts, **kwargs):
                return np.array([[0.6, 0.8, 0.0] for _ in texts], dtype=np.float32)

        self.ingest([("a.xml", article(1, abstract="First target evidence") +
                     article(2, abstract="Second target evidence") +
                     article(3, abstract="Third target evidence") +
                     article(4, abstract="Fourth target evidence"))])
        build_bm25()
        with patch("embedding_shards._encoder", return_value=Model()):
            run_worker("test-model", 0, 2, rows_per_part=2, gpu_batch=2, reserve_gb=0)
            run_worker("test-model", 1, 2, rows_per_part=2, gpu_batch=2, reserve_gb=0)
        summary = verify(2, 2)
        self.assertEqual(summary["articles"], 4)
        config = finalize(2, 2, ann=False, reserve_gb=0, consume_vector_parts=True)
        _, index, _ = paths()
        table = lancedb.connect(str(index)).open_table(config["table"])
        self.assertNotIn("text", table.schema.names)
        self.assertNotIn("title", table.schema.names)
        self.assertFalse(any((index / "vector-parts").rglob("*.parquet")))
        self.assertTrue(any((index / "passage-parts").rglob("*.parquet")))
        with patch("retrieval._encoder", return_value=Model()):
            result = HybridSearch().search("target", k=2)
        self.assertEqual(result["vector_status"], "complete")
        self.assertTrue(any("target evidence" in evidence["text"]
                            for row in result["results"] for evidence in row["evidence"]))

    def test_embedding_pool_uses_configured_gpus(self):
        from embedding_shards import encoder_pool

        class Model:
            def start_multi_process_pool(self, target_devices):
                self.target_devices = target_devices
                return {"processes": []}

        model = Model()
        with patch.dict(os.environ, {"PUBMED_EMBED_DEVICES": "cuda:0,cuda:1"}):
            pool, devices = encoder_pool(model)
        self.assertEqual(devices, ["cuda:0", "cuda:1"])
        self.assertEqual(model.target_devices, devices)
        self.assertEqual(pool, {"processes": []})

    def test_agent_rejects_invented_single_digit_and_accepts_source(self):
        self.ingest([("a.xml", article(12345, abstract="The study reported 7 participants with improved outcomes."))])
        import tools
        import agent
        old = (tools.STORE, tools.INDEX, tools.DATASET)
        tools.STORE, tools.INDEX, tools.DATASET = paths()
        result = {"results": [{"pmid": "12345", "abstract": "The study reported 7 participants with improved outcomes."}]}
        tool_call = {"role": "assistant", "tool_calls": [{"function": {"name": "get_articles", "arguments": {"pmids": ["12345"]}}}]}
        try:
            for digit, refused in [("8", True), ("7", False)]:
                response = {"role": "assistant", "content": json.dumps({"evidence": [{"pmid": "12345", "quote": f"The study reported {digit} participants with improved outcomes."}]})}
                with patch("agent.chat", side_effect=[tool_call, response]), patch("tools.call", return_value=result):
                    answer = agent.run("Investigate the available records.", adaptive=False)
                    self.assertEqual(answer["refused"], refused)
                    if refused:
                        self.assertNotIn("8 participants", answer["answer"])
        finally:
            tools.STORE, tools.INDEX, tools.DATASET = old

    def test_groq_tool_request_retries_strict_payload_without_deprecated_tokens(self):
        import agent

        class Response:
            def __init__(self, status, body):
                self.status_code = status
                self._body = body
                self.ok = status == 200
                self.headers = {}

            def json(self):
                return self._body

        rejected = Response(400, {"error": {
            "message": "Failed to call a function. Please adjust your prompt.",
            "failed_generation": {"reason": "invalid tool arguments"},
        }})
        accepted = Response(200, {"choices": [{"message": {
            "role": "assistant", "content": "ready",
        }}], "usage": {"completion_tokens": 1}})
        sent = []

        def post(*args, **kwargs):
            sent.append(kwargs["json"])
            return rejected if len(sent) == 1 else accepted

        config = {
            "PUBMED_LLM": "cloud",
            "PUBMED_ALLOW_CLOUD": "1",
            "PUBMED_API_BASE": "https://api.groq.com/openai/v1",
            "PUBMED_API_KEY": "test-key",
            "PUBMED_CLOUD_MODEL": "openai/gpt-oss-120b",
        }
        with patch.dict(os.environ, config), patch("requests.post", side_effect=post):
            reply = agent.chat([{"role": "user", "content": "question"}],
                               "openai/gpt-oss-120b", 30)
        self.assertEqual(reply["content"], "ready")
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[1]["temperature"], 0)
        self.assertEqual(sent[1]["reasoning_format"], "hidden")
        self.assertEqual(sent[1]["max_completion_tokens"], 900)
        self.assertEqual(sent[1]["reasoning_effort"], "low")
        self.assertNotIn("max_tokens", sent[1])
        with patch.dict(os.environ, config), patch("requests.post", side_effect=post):
            agent.chat([{"role": "user", "content": "write JSON"}],
                       "openai/gpt-oss-120b", 30, allow_tools=False)
        self.assertEqual(sent[2]["response_format"], {"type": "json_object"})
        self.assertNotIn("tools", sent[2])
        self.assertEqual(agent._json_answer('```json\n{"evidence": []}\n```'),
                         {"evidence": []})

    def test_findings_question_retrieves_before_one_model_call(self):
        self.ingest([("a.xml", article(
            12345, abstract="Participants reported persistent fatigue after COVID-19."))])
        import tools
        import agent
        old = (tools.STORE, tools.INDEX, tools.DATASET)
        tools.STORE, tools.INDEX, tools.DATASET = paths()
        result = {"results": [{
            "pmid": "12345", "has_abstract": True,
            "abstract": "Participants reported persistent fatigue after COVID-19.",
            "evidence": [{"section": "abstract",
                          "text": "Participants reported persistent fatigue after COVID-19."}],
        }]}
        response = {"role": "assistant", "content": json.dumps({"claims": [{
            "text": "The retrieved study reports persistent fatigue after COVID-19.",
            "source_ids": [1],
        }]})}
        try:
            with patch("agent.chat", return_value=response) as model, \
                    patch("tools.call", return_value=result):
                answer = agent.run(
                    "What do papers report about long COVID fatigue?", adaptive=False)
            self.assertIn("persistent fatigue", answer["answer"])
            self.assertEqual(model.call_count, 1)
            self.assertFalse(model.call_args.kwargs["allow_tools"])
            self.assertEqual(answer["calls"][0]["tool"], "search_literature")
        finally:
            tools.STORE, tools.INDEX, tools.DATASET = old


if __name__ == "__main__":
    unittest.main()
