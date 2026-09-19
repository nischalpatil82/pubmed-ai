"""Run against a built snapshot, without network calls or a generation provider."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from fastapi.testclient import TestClient
from app import app
from dataset import atomic_json


with TestClient(app) as client:
    health = client.get("/health")
    assert health.status_code == 200, health.text
    stats = client.get("/api/stats")
    assert stats.status_code == 200, stats.text
    search = client.get("/api/search", params={"q": "diabetes", "since_year": 2020, "k": 5})
    assert search.status_code == 200, search.text
    assert all(r["pub_year"] >= 2020 for r in search.json()["results"])
    assert client.get("/api/search", params={"q": "test", "k": 500}).status_code == 422
    answer = client.get("/api/ask", params={"q": "how many articles?"})
    assert answer.status_code == 200, answer.text
    assert str(stats.json()["articles"]) in answer.json()["answer"].replace(",", "")
    atomic_json(sys.argv[1], {"health": health.json(), "stats": stats.json(), "search": search.json(),
                             "direct_answer": answer.json(), "provider_called": False})
