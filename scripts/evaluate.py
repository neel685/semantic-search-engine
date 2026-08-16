from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import get_project_root, load_config
from src.data.loader import load_processed_data
from src.evaluation.evaluator import compare_methods
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever


def make_methods(documents, config):
    retrieval_cfg = config.get("retrieval", {})
    embedding_cfg = config.get("embedding", {})
    bm25 = BM25Retriever(documents, k1=float(retrieval_cfg.get("bm25_k1", 1.5)), b=float(retrieval_cfg.get("bm25_b", 0.75)))
    dense = DenseRetriever(
        documents,
        model_name=embedding_cfg.get("model_name", "BAAI/bge-small-en-v1.5"),
        device=embedding_cfg.get("device", "auto"),
        normalize=bool(embedding_cfg.get("normalize", True)),
        batch_size=int(embedding_cfg.get("batch_size", 32)),
        index_type=retrieval_cfg.get("index_type", "IndexFlatIP"),
    )
    hybrid = HybridRetriever(
        documents,
        bm25_retriever=bm25,
        dense_retriever=dense,
        alpha=float(retrieval_cfg.get("hybrid_alpha", 0.5)),
    )
    return {
        "BM25": lambda query, top_k: bm25.search(query, top_k=top_k),
        "Dense": lambda query, top_k: dense.search(query, top_k=top_k),
        "Hybrid": lambda query, top_k: hybrid.search(query, top_k=top_k),
    }


def main() -> None:
    config = load_config()
    dataset_cfg = config.get("dataset", {})
    processed_dir = get_project_root() / dataset_cfg.get("processed_path", "data/processed")
    documents, queries, qrels = load_processed_data(processed_dir)
    if not queries:
        raise ValueError("No queries available in processed dataset. Run preprocessing first.")

    methods = make_methods(documents, config)
    results = compare_methods(queries, methods, top_k_values=(1, 5, 10))

    print("Method              Recall@10    MRR@10    NDCG@10")
    print("---------------------------------------------------")
    for method_name, metrics in results.items():
        print(
            f"{method_name:<16} "
            f"{metrics.get('recall@10', 0.0):>10.4f}    "
            f"{metrics.get('mrr@10', 0.0):>7.4f}    "
            f"{metrics.get('ndcg@10', 0.0):>7.4f}"
        )

    results_path = get_project_root() / "results" / "evaluation_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    metric_names = ["recall@10", "mrr@10", "ndcg@10"]
    labels = list(results.keys())
    for metric_name in metric_names:
        values = [results[method].get(metric_name, 0.0) for method in labels]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(labels, values, color=["steelblue", "darkorange", "forestgreen"])
        ax.set_title(f"{metric_name} by Retrieval Method")
        ax.set_xlabel("Method")
        ax.set_ylabel(metric_name)
        ax.set_ylim(0, 1.1)
        for tick in ax.get_xticklabels():
            tick.set_rotation(15)
        plt.tight_layout()
        plt.savefig(get_project_root() / "results" / "figures" / f"{metric_name.replace('@', '_')}.png")
        plt.close(fig)

    print(f"Saved evaluation results to {results_path}")


if __name__ == "__main__":
    main()
