from __future__ import annotations

from typing import Any


class HybridRetriever:
    def __init__(
        self,
        documents: list[dict[str, Any]],
        *,
        bm25_retriever: Any | None = None,
        dense_retriever: Any | None = None,
        alpha: float = 0.5,
    ) -> None:
        self.documents = documents
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.alpha = alpha

    @staticmethod
    def _normalize_scores(scores: list[float]) -> list[float]:
        if not scores:
            return []
        minimum = min(scores)
        maximum = max(scores)
        if maximum == minimum:
            return [0.0 for _ in scores]
        return [(score - minimum) / (maximum - minimum) for score in scores]

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        if self.bm25_retriever is None or self.dense_retriever is None:
            raise ValueError("Hybrid retrieval requires both BM25 and dense retrievers.")

        bm25_results = self.bm25_retriever.search(query, top_k=max(top_k * 5, top_k))
        dense_results = self.dense_retriever.search(query, top_k=max(top_k * 5, top_k))

        merged: dict[str, dict[str, float]] = {}
        for result in bm25_results:
            doc_id = str(result["document_id"])
            merged.setdefault(doc_id, {"bm25": 0.0, "semantic": 0.0})
            merged[doc_id]["bm25"] = float(result["score"])
        for result in dense_results:
            doc_id = str(result["document_id"])
            merged.setdefault(doc_id, {"bm25": 0.0, "semantic": 0.0})
            merged[doc_id]["semantic"] = float(result["score"])

        bm25_values = [payload["bm25"] for payload in merged.values()]
        semantic_values = [payload["semantic"] for payload in merged.values()]
        normalized_bm25 = self._normalize_scores(bm25_values)
        normalized_semantic = self._normalize_scores(semantic_values)

        combined: list[dict[str, Any]] = []
        for idx, (doc_id, payload) in enumerate(merged.items()):
            bm25_score = payload["bm25"]
            semantic_score = payload["semantic"]
            combined_score = (
                self.alpha * normalized_bm25[idx] + (1 - self.alpha) * normalized_semantic[idx]
            )
            text = ""
            for result in bm25_results + dense_results:
                if str(result["document_id"]) == doc_id:
                    text = result.get("text", "")
                    break
            combined.append({
                "document_id": doc_id,
                "score": combined_score,
                "text": text,
                "bm25_score": bm25_score,
                "semantic_score": semantic_score,
            })

        combined.sort(key=lambda item: item["score"], reverse=True)
        output: list[dict[str, Any]] = []
        for rank, item in enumerate(combined[:top_k], start=1):
            output.append({
                "rank": rank,
                "document_id": item["document_id"],
                "score": float(item["score"]),
                "text": item["text"],
            })
        return output


def hybrid_search(
    query: str,
    documents: list[dict[str, Any]],
    *,
    bm25_retriever: Any,
    dense_retriever: Any,
    top_k: int = 10,
    alpha: float = 0.5,
) -> list[dict[str, Any]]:
    retriever = HybridRetriever(documents, bm25_retriever=bm25_retriever, dense_retriever=dense_retriever, alpha=alpha)
    return retriever.search(query, top_k=top_k)
