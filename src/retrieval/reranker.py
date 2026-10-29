from __future__ import annotations

from typing import Any


class CrossEncoderReranker:
    """Rerank a small candidate set with a query-document cross-encoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            kwargs = {"device": self.device} if self.device and self.device != "auto" else {}
            self._model = CrossEncoder(self.model_name, **kwargs)
        return self._model

    def rank(self, query: str, candidates: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        if not candidates:
            return []
        pairs = [(query, str(candidate.get("text", ""))) for candidate in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        ranked = []
        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["score"] = float(score)
            ranked.append(item)
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return [
            {**item, "rank": rank}
            for rank, item in enumerate(ranked[:top_k], start=1)
        ]
