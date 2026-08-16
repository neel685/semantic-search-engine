from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np

faiss.omp_set_num_threads(1)

from src.config import get_project_root
from src.embeddings.encoder import EmbeddingEncoder


class DenseRetriever:
    def __init__(
        self,
        documents: list[dict[str, Any]] | None = None,
        *,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
        normalize: bool = True,
        batch_size: int = 32,
        index_type: str = "IndexFlatIP",
    ) -> None:
        self.documents: list[dict[str, Any]] = []
        self.document_ids: list[str] = []
        self.document_texts: list[str] = []
        self.encoder = EmbeddingEncoder(model_name=model_name, device=device, normalize=normalize, batch_size=batch_size)
        self.index_type = index_type
        self.index: faiss.Index | None = None
        self.embeddings: np.ndarray | None = None
        if documents is not None:
            self.fit(documents)

    def fit(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents
        self.document_ids = [str(document.get("id", "")) for document in documents]
        self.document_texts = [str(document.get("text", "")) for document in documents]
        embeddings = self.encoder.encode_documents(self.document_texts)
        if embeddings.size == 0:
            raise ValueError("Cannot build a dense index from an empty document collection.")

        if self.index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
        elif self.index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        else:
            raise ValueError(f"Unsupported FAISS index type: {self.index_type}")

        if self.index_type == "IndexFlatIP":
            faiss.normalize_L2(embeddings)

        self.embeddings = embeddings.astype(np.float32)
        self.index.add(self.embeddings)

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.index is None:
            raise ValueError("No index available to save.")

        faiss.write_index(self.index, str(output_path))
        mapping_path = output_path.with_name("document_mapping.json")
        with mapping_path.open("w", encoding="utf-8") as file:
            json.dump({"document_ids": self.document_ids, "documents": self.documents}, file, ensure_ascii=False, indent=2)
        return output_path

    def load(self, path: str | Path, *, doc_mapping_path: str | Path | None = None) -> None:
        index_path = Path(path)
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        self.index = faiss.read_index(str(index_path))
        mapping_file = Path(doc_mapping_path) if doc_mapping_path else index_path.with_name("document_mapping.json")
        if mapping_file.exists():
            with mapping_file.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            self.document_ids = payload.get("document_ids", [])
            self.documents = payload.get("documents", [])
            self.document_texts = [str(document.get("text", "")) for document in self.documents]

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        if self.index is None:
            raise ValueError("Dense retriever must be fitted or loaded before searching.")

        query_vector = self.encoder.encode_query(query).astype(np.float32)
        if self.index_type == "IndexFlatIP":
            faiss.normalize_L2(query_vector.reshape(1, -1))

        scores, indices = self.index.search(query_vector.reshape(1, -1), min(top_k, self.index.ntotal))
        results: list[dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1:
                continue
            doc_id = self.document_ids[int(idx)] if int(idx) < len(self.document_ids) else str(int(idx))
            text = self.document_texts[int(idx)] if int(idx) < len(self.document_texts) else ""
            results.append({
                "rank": rank,
                "document_id": doc_id,
                "score": float(score),
                "text": text,
            })
        return results


def retrieve_semantic(
    query: str,
    documents: list[dict[str, Any]],
    *,
    top_k: int = 10,
    model_name: str = "BAAI/bge-small-en-v1.5",
    device: str | None = None,
    normalize: bool = True,
    batch_size: int = 32,
    index_type: str = "IndexFlatIP",
) -> list[dict[str, Any]]:
    retriever = DenseRetriever(
        documents,
        model_name=model_name,
        device=device,
        normalize=normalize,
        batch_size=batch_size,
        index_type=index_type,
    )
    return retriever.search(query, top_k=top_k)


def load_dense_index(index_path: str | Path, doc_mapping_path: str | Path | None = None) -> DenseRetriever:
    retriever = DenseRetriever()
    retriever.load(index_path, doc_mapping_path=doc_mapping_path)
    return retriever
