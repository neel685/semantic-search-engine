from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from src.api.server import app
from src.data.preprocessing import clean_text
from src.embeddings.encoder import EmbeddingEncoder
from src.evaluation.metrics import (
    answer_relevance_proxy,
    context_precision_at_k,
    faithfulness_proxy,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever


def test_preprocessing_clean_text() -> None:
    raw = "  Hello,   WORLD!\nThis is a test.  "
    assert clean_text(raw) == "hello, world! this is a test."


def test_bm25_retrieval_works() -> None:
    documents = [
        {"id": "d1", "text": "Apple is a fruit with red or green skin."},
        {"id": "d2", "text": "The car is parked in the garage."},
        {"id": "d3", "text": "Banana is a yellow tropical fruit."},
    ]
    retriever = BM25Retriever(documents, k1=1.5, b=0.75)
    results = retriever.search("fruit apple", top_k=2)
    assert results[0]["document_id"] == "d1"


def test_embedding_encoder_produces_vectors() -> None:
    encoder = EmbeddingEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu", normalize=True, batch_size=4)
    vectors = encoder.encode(["apple fruit", "banana fruit", "car garage"])
    assert vectors.shape[0] == 3
    assert vectors.shape[1] > 0
    assert np.isfinite(vectors).all()


def test_dense_search_returns_ranking() -> None:
    documents = [
        {"id": "d1", "text": "The dog chased the ball in the park."},
        {"id": "d2", "text": "A cat sat on a warm window sill."},
        {"id": "d3", "text": "A bicycle is parked beside the fence."},
    ]
    retriever = DenseRetriever(documents, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu", normalize=True, batch_size=4)
    results = retriever.search("dog playing in park", top_k=2)
    assert results[0]["document_id"] == "d1"


def test_hybrid_search_combines_scores() -> None:
    class FakeBM25:
        def search(self, query: str, top_k: int = 10):
            return [
                {"document_id": "d1", "score": 7.0, "text": "apple fruit"},
                {"document_id": "d2", "score": 3.0, "text": "car garage"},
            ]

    class FakeDense:
        def search(self, query: str, top_k: int = 10):
            return [
                {"document_id": "d2", "score": 0.93, "text": "car garage"},
                {"document_id": "d1", "score": 0.89, "text": "apple fruit"},
            ]

    documents = [
        {"id": "d1", "text": "apple fruit"},
        {"id": "d2", "text": "car garage"},
    ]
    hybrid = HybridRetriever(documents, bm25_retriever=FakeBM25(), dense_retriever=FakeDense(), alpha=0.5)
    results = hybrid.search("fruit", top_k=2)
    assert results[0]["document_id"] in {"d1", "d2"}
    assert len(results) == 2


def test_evaluation_metrics() -> None:
    relevant = {"d1", "d3"}
    retrieved = ["d2", "d1", "d4"]
    assert recall_at_k(relevant, retrieved, 3) == 0.5
    assert mrr_at_k(relevant, retrieved, 3) == 1 / 2
    assert ndcg_at_k(relevant, retrieved, 3) > 0.0
    assert context_precision_at_k(relevant, retrieved, 3) == 1 / 3
    assert faithfulness_proxy("Apple is a fruit.", "Apple is a fruit with red skin.") == 1.0
    assert answer_relevance_proxy("apple fruit", "Apple is a fruit.") == 1.0


def test_api_validation() -> None:
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    preflight = client.options(
        "/search",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers.get("access-control-allow-origin") == "*"

    response = client.post("/search", json={"query": "sample query", "top_k": 5, "method": "semantic"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["method"] == "semantic"
    assert len(payload["results"]) > 0

    invalid = client.post("/search", json={"query": "", "top_k": 0, "method": "unknown"})
    assert invalid.status_code in {400, 422}
