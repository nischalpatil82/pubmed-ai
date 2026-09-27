"""Check copied-release verification with a small synthetic inventory."""
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops.verify_snapshot import verify


class VerifySnapshotTest(unittest.TestCase):
    def test_checks_files_and_detects_corruption(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "stores" / "snapshot").mkdir(parents=True)
            (root / "indexes" / "snapshot").mkdir(parents=True)
            article = root / "stores" / "snapshot" / "articles.parquet"
            article.write_bytes(b"test article")
            dataset = {"snapshot": "snapshot", "status": "ready", "pilot": False,
                       "store": "stores/snapshot", "index": "indexes/snapshot"}
            inventory = {"dataset": {"snapshot": "snapshot"},
                         "files": {"store/articles.parquet": {
                             "bytes": article.stat().st_size,
                             "sha256": hashlib.sha256(article.read_bytes()).hexdigest()}}}
            manifest = root / "dataset.json"
            release = root / "release.json"
            manifest.write_text(json.dumps(dataset), encoding="utf-8")
            release.write_text(json.dumps(inventory), encoding="utf-8")
            self.assertEqual(verify(manifest, release), (1, 12))
            article.write_bytes(b"bad!")
            with self.assertRaisesRegex(ValueError, "Size mismatch"):
                verify(manifest, release)


if __name__ == "__main__":
    unittest.main()
