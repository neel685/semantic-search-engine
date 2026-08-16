from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_project_root, load_config
from src.data.loader import load_processed_data
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever


def main() -> None:
    config = load_config()
    dataset_cfg = config.get("dataset", {})
    processed_dir = get_project_root() / dataset_cfg.get("processed_path", "data/processed")
    documents, queries, _ = load_processed_data(processed_dir)
    if not documents or not queries:
        raise ValueError("No processed dataset found. Run preprocessing first.")

    start = time.perf_counter()
    bm25 = BM25Retriever(documents, k1=1.5, b=0.75)
    bm25_time = time.perf_counter() - start

    start = time.perf_counter()
    dense = DenseRetriever(
        documents,
        model_name=config.get("embedding", {}).get("model_name", "BAAI/bge-small-en-v1.5"),
        device=config.get("embedding", {}).get("device", "auto"),
        normalize=True,
        batch_size=int(config.get("embedding", {}).get("batch_size", 32)),
    )
    dense_time = time.perf_counter() - start

    query = queries[0]["text"]
    start = time.perf_counter()
    _ = bm25.search(query, top_k=10)
    bm25_latency = time.perf_counter() - start

    start = time.perf_counter()
    _ = dense.search(query, top_k=10)
    dense_latency = time.perf_counter() - start

    print("Performance Summary")
    print(f"BM25 setup time: {bm25_time:.4f}s")
    print(f"Dense setup time: {dense_time:.4f}s")
    print(f"BM25 retrieval latency: {bm25_latency:.6f}s")
    print(f"Dense retrieval latency: {dense_latency:.6f}s")

    output_dir = get_project_root() / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "benchmark.txt"
    with summary_path.open("w", encoding="utf-8") as file:
        file.write(
            f"BM25 setup time: {bm25_time:.4f}s\n"
            f"Dense setup time: {dense_time:.4f}s\n"
            f"BM25 retrieval latency: {bm25_latency:.6f}s\n"
            f"Dense retrieval latency: {dense_latency:.6f}s\n"
        )
    print(f"Saved benchmark summary to {summary_path}")


if __name__ == "__main__":
    main()
