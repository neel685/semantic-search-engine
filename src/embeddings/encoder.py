from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

if torch.cuda.is_available():
    torch.set_num_threads(1)
else:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def get_device(device: str | None = None) -> str:
    if device is None:
        device = "auto"
    if device == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class EmbeddingEncoder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        device: str | None = None,
        normalize: bool = True,
        batch_size: int = 32,
    ) -> None:
        selected_device = get_device(device)
        self.model_name = model_name
        self.device = selected_device
        self.normalize = normalize
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=selected_device)

    def encode(self, texts: list[str], *, batch_size: int | None = None) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        actual_batch_size = batch_size or self.batch_size
        embeddings = self.model.encode(
            texts,
            batch_size=actual_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_documents(self, documents: list[str] | list[dict[str, Any]], *, batch_size: int | None = None) -> np.ndarray:
        texts = []
        for document in documents:
            if isinstance(document, dict):
                texts.append(str(document.get("text", "")))
            else:
                texts.append(str(document))
        return self.encode(texts, batch_size=batch_size)

    def encode_query(self, query: str, *, batch_size: int | None = None) -> np.ndarray:
        encoded = self.encode([query], batch_size=batch_size)
        return encoded[0].astype(np.float32)
