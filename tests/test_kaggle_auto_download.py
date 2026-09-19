import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


SCRIPT = Path(__file__).parents[1] / "kaggle" / "auto_download.py"
SPEC = importlib.util.spec_from_file_location("auto_download", SCRIPT)
auto_download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(auto_download)


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


class KaggleAutoDownloadTests(unittest.TestCase):
    def test_safe_extract_rejects_parent_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as target:
                target.writestr("../outside.txt", "unsafe")
            with self.assertRaisesRegex(RuntimeError, "Unsafe archive member"):
                auto_download.safe_extract(archive, root / "output")

    def test_validate_restored_index_checks_counts_and_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            index = Path(folder)
            passage = index / "passage-parts" / "w1-0" / "part-000000.parquet"
            vector = index / "vector-parts" / "w1-0" / "part-000000.parquet"
            checkpoint = index / "embedding-checkpoints" / "w1-0" / "part-000000.json"
            for path, value in ((passage, b"passages"), (vector, b"vectors")):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(
                json.dumps(
                    {
                        "articles": 2,
                        "passages": 3,
                        "passage_file": passage.relative_to(index).as_posix(),
                        "passage_file_bytes": passage.stat().st_size,
                        "passage_file_sha256": digest(passage),
                        "vector_file": vector.relative_to(index).as_posix(),
                        "vector_file_bytes": vector.stat().st_size,
                        "vector_file_sha256": digest(vector),
                    }
                ),
                encoding="utf-8",
            )
            result = auto_download.validate_restored_index(
                index, {"parts": 1, "articles": 2, "passages": 3}
            )
            self.assertEqual(result, {"articles": 2, "passages": 3, "parts": 1})
            vector.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                auto_download.validate_restored_index(
                    index, {"parts": 1, "articles": 2, "passages": 3}
                )


if __name__ == "__main__":
    unittest.main()
