from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi


def tokenize(text: str | None) -> list[str]:
    if text is None:
        return []
    return re.findall(r"\b\w+\b", str(text).lower())


class BM25Retriever:
    def __init__(self, documents: list[str] | list[dict[str, Any]] | None = None, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents: list[str] = []
        self.document_ids: list[str] = []
        self.bm25: BM25Okapi | None = None
        if documents is not None:
            self.fit(documents)

    def _normalize_documents(self, documents: list[str] | list[dict[str, Any]]) -> tuple[list[str], list[str]]:
        texts: list[str] = []
        ids: list[str] = []
        for document in documents:
            if isinstance(document, dict):
                texts.append(str(document.get("text", "")))
                ids.append(str(document.get("id", document.get("document_id", ""))))
            else:
                texts.append(str(document))
                ids.append("")
        return texts, ids

    def fit(self, documents: list[str] | list[dict[str, Any]]) -> None:
        texts, ids = self._normalize_documents(documents)
        self.documents = texts
        self.document_ids = ids
        tokenized = [tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        if self.bm25 is None:
            raise ValueError("BM25Retriever must be fitted before searching.")

        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]

        results: list[dict[str, Any]] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            if scores[idx] <= 0:
                continue
            results.append({
                "rank": rank,
                "document_id": self.document_ids[idx] if self.document_ids[idx] else str(idx),
                "score": float(scores[idx]),
                "text": self.documents[idx],
            })
        return results


def retrieve_bm25(
    query: str,
    documents: list[str] | list[dict[str, Any]],
    *,
    top_k: int = 10,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict[str, Any]]:
    retriever = BM25Retriever(documents, k1=k1, b=b)
    return retriever.search(query, top_k=top_k)
