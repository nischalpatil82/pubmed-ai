"""Synthetic ANN integration benchmark. This does not measure biomedical relevance."""
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
import lancedb
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from dataset import atomic_json


def run(output):
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(4096, 32)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    with tempfile.TemporaryDirectory() as tmp:
        db = lancedb.connect(tmp)
        table = db.create_table("ann_test", [{"id": str(i), "vector": vector.tolist()} for i, vector in enumerate(vectors)])
        table.create_index(metric="cosine", index_type="IVF_FLAT", num_partitions=16)
        rows = []
        for vector in vectors[::200][:20]:
            start = time.perf_counter()
            exact = table.search(vector).distance_type("cosine").bypass_vector_index().limit(10).to_list()
            exact_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            ann = table.search(vector).distance_type("cosine").limit(10).to_list()
            gold = {r["id"] for r in exact}
            rows.append({"recall_at_10": len(gold & {r["id"] for r in ann}) / len(gold),
                         "exact_ms": exact_ms, "ann_ms": (time.perf_counter() - start) * 1000})
        atomic_json(output, {"kind": "synthetic integration only", "vectors": 4096, "queries": len(rows),
            "mean_recall_at_10": float(np.mean([r["recall_at_10"] for r in rows])), "rows": rows,
            "limitation": "Not representative of PubMed embeddings or production scale; ANN remains opt-in."})


if __name__ == "__main__":
    run(sys.argv[1])
