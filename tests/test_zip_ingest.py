import gzip
import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile


MODULE = Path(__file__).resolve().parents[1] / "pipeline" / "pubmed_ingest.py"
spec = importlib.util.spec_from_file_location("pubmed_ingest", MODULE)
ingest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingest)

XML = b'''<?xml version="1.0"?>
<PubmedArticleSet><PubmedArticle><MedlineCitation Status="MEDLINE">
<PMID>12345</PMID><Article><ArticleTitle>A <i>nested</i> title</ArticleTitle>
<Abstract><AbstractText Label="RESULTS">Sample evidence.</AbstractText></Abstract>
</Article></MedlineCitation></PubmedArticle>
<DeleteCitation><PMID>99999</PMID></DeleteCitation></PubmedArticleSet>'''


class ZipIngestTests(unittest.TestCase):
    def test_equivalent_rows_for_xml_gzip_and_zip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            xml = root / "sample.xml"
            gz = root / "sample.xml.gz"
            archive = root / "sample.zip"
            xml.write_bytes(XML)
            gz.write_bytes(gzip.compress(XML))
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("nested/sample.xml", XML)
            results = []
            for number, source in enumerate((xml, gz, archive)):
                dest = root / f"output{number}"
                dest.mkdir()
                [(path, member)] = ingest.source_jobs(str(source))
                _, count = ingest.process_file(path, str(dest), member)
                self.assertEqual(count, 1)
                result = {f.name: f.read_bytes() for f in dest.iterdir()}
                self.assertIn(b"A nested title", result["articles__sample.csv"])
                self.assertEqual(result["deleted_pmids__sample.csv"].strip(), b"99999")
                results.append(result)
            self.assertEqual(results[0], results[1])
            self.assertEqual(results[0], results[2])
            self.assertFalse((root / "nested").exists())

    def test_gzipped_member_and_source_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "sample.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("pubmed25n1509.xml.gz", gzip.compress(XML))
                z.writestr("pubmed25n1011.xml", XML)
                z.writestr("notes.txt", "Not XML")
            jobs = ingest.source_jobs(str(archive))
            self.assertEqual([m for _, m in jobs],
                             ["pubmed25n1011.xml", "pubmed25n1509.xml.gz"])
            dest = root / "output"
            dest.mkdir()
            self.assertEqual(ingest.process_file(jobs[1][0], str(dest), jobs[1][1])[1], 1)

    def test_duplicate_member_stems_cannot_overwrite_results(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "duplicate.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("one/sample.xml", XML)
                z.writestr("two/sample.xml", XML)
            with self.assertRaisesRegex(ValueError, "duplicate source stem"):
                ingest.source_jobs(str(archive))

    def test_incomplete_zip_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "broken.zip"
            archive.write_bytes(b"PK\x03\x04incomplete")
            with self.assertRaises(zipfile.BadZipFile):
                ingest.source_jobs(str(archive))


if __name__ == "__main__":
    unittest.main()
