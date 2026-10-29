from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import get_project_root, load_config
from src.data.loader import load_processed_data
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.generation.context import build_context
from src.generation.ollama import OllamaGenerator


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query to search")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of search results")
    method: Literal["bm25", "semantic", "hybrid"] = Field(default="hybrid", description="Retrieval method")


class SearchResult(BaseModel):
    rank: int
    document_id: str
    score: float
    text: str


class SearchResponse(BaseModel):
    query: str
    method: str
    results: list[SearchResult]


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question to answer")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of source passages")


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[SearchResult]


class SearchEngine:
    def __init__(self, config_path: str | Path = "config.yaml") -> None:
        self.config = load_config(config_path)
        dataset_cfg = self.config.get("dataset", {})
        embedding_cfg = self.config.get("embedding", {})
        retrieval_cfg = self.config.get("retrieval", {})
        processed_dir = Path(dataset_cfg.get("processed_path", "data/processed"))
        if not processed_dir.is_absolute():
            processed_dir = get_project_root() / processed_dir

        self.documents, self.queries, self.qrels = load_processed_data(processed_dir)
        if not self.documents:
            self.documents = [
                {"id": "fallback_1", "text": "Semantic search retrieves relevant passages by comparing meaning, not just exact keywords."},
                {"id": "fallback_2", "text": "BM25 is a lexical retrieval model that scores documents based on term frequency and document length."},
                {"id": "fallback_3", "text": "FAISS is a vector index used to perform efficient nearest-neighbor search in embedding space."},
                {"id": "fallback_4", "text": "Hybrid retrieval combines BM25 lexical relevance with semantic similarity for better ranking."},
            ]

        self.bm25 = BM25Retriever(
            self.documents,
            k1=float(retrieval_cfg.get("bm25_k1", 1.5)),
            b=float(retrieval_cfg.get("bm25_b", 0.75)),
        ) if self.documents else None
        self.dense = DenseRetriever(
            self.documents,
            model_name=embedding_cfg.get("model_name", "BAAI/bge-small-en-v1.5"),
            device=embedding_cfg.get("device", "auto"),
            normalize=bool(embedding_cfg.get("normalize", True)),
            batch_size=int(embedding_cfg.get("batch_size", 32)),
            index_type=retrieval_cfg.get("index_type", "IndexFlatIP"),
        ) if self.documents else None
        self.hybrid = HybridRetriever(
            self.documents,
            bm25_retriever=self.bm25,
            dense_retriever=self.dense,
            alpha=float(retrieval_cfg.get("hybrid_alpha", 0.5)),
        ) if self.documents and self.bm25 and self.dense else None
        reranking_cfg = self.config.get("reranking", {})
        self.reranker = (
            CrossEncoderReranker(
                model_name=reranking_cfg.get("model_name", "BAAI/bge-reranker-v2-m3"),
                device=embedding_cfg.get("device", "auto"),
                batch_size=int(reranking_cfg.get("batch_size", 16)),
            )
            if bool(reranking_cfg.get("enabled", True))
            else None
        )
        generation_cfg = self.config.get("generation", {})
        self.generator = (
            OllamaGenerator(
                model_name=generation_cfg.get("model_name", "qwen3:8b"),
                base_url=generation_cfg.get("base_url", "http://localhost:11434"),
                temperature=float(generation_cfg.get("temperature", 0.0)),
                timeout=float(generation_cfg.get("timeout_seconds", 120)),
            )
            if generation_cfg.get("provider", "ollama") == "ollama"
            else None
        )

    def search(self, query: str, method: str = "hybrid", top_k: int = 10) -> list[dict[str, float | int | str]]:
        if not self.documents:
            raise HTTPException(status_code=503, detail="No processed documents found. Run preprocessing and indexing first.")

        if method == "bm25":
            results = self.bm25.search(query, top_k=top_k) if self.bm25 is not None else []
        elif method == "semantic":
            results = self.dense.search(query, top_k=top_k) if self.dense is not None else []
        elif method == "hybrid":
            results = self.hybrid.search(query, top_k=top_k) if self.hybrid is not None else []
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported retrieval method: {method}")

        return [
            {
                "rank": item["rank"],
                "document_id": str(item["document_id"]),
                "score": float(item["score"]),
                "text": str(item["text"]),
            }
            for item in results
        ]

    def answer(self, query: str, top_k: int = 5) -> tuple[str, list[dict[str, float | int | str]]]:
        candidate_k = max(top_k * 6, 20)
        candidates = self.search(query, method="hybrid", top_k=candidate_k)
        if self.reranker is not None:
            candidates = self.reranker.rank(query, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]
        sources = [
            {
                "rank": index,
                "document_id": str(item["document_id"]),
                "score": float(item["score"]),
                "text": str(item["text"]),
            }
            for index, item in enumerate(candidates, start=1)
        ]
        if self.generator is None:
            raise HTTPException(status_code=503, detail="No answer generator is configured.")
        context_cfg = self.config.get("generation", {})
        context = build_context(
            sources,
            max_characters=int(context_cfg.get("max_context_characters", 12000)),
        )
        return self.generator.generate(query, context), sources


def create_app(config_path: str | Path = "config.yaml") -> FastAPI:
    app = FastAPI(title="Semantic Search Engine API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.search_engine = SearchEngine(config_path=config_path)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")
        results = app.state.search_engine.search(request.query, method=request.method, top_k=request.top_k)
        return SearchResponse(query=request.query, method=request.method, results=[SearchResult(**result) for result in results])

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
        answer, sources = app.state.search_engine.answer(request.query, top_k=request.top_k)
        return AskResponse(
            query=request.query,
            answer=answer,
            sources=[SearchResult(**source) for source in sources],
        )

    return app


app = create_app()
