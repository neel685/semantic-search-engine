from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from src.evaluation.metrics import mrr_at_k, ndcg_at_k, recall_at_k


def evaluate_query(
    query: dict[str, Any],
    retrieval_function: Callable[[str, int], list[dict[str, Any]]],
    *,
    top_k_values: tuple[int, ...] = (1, 5, 10),
) -> dict[str, dict[str, float]]:
    relevant_ids = set(query.get("relevant_doc_ids", []))
    result_by_k: dict[str, dict[str, float]] = {}

    for k in top_k_values:
        retrieved = retrieval_function(query["text"], k)
        retrieved_ids = [str(result["document_id"]) for result in retrieved]
        result_by_k[f"@{k}"] = {
            "recall": recall_at_k(relevant_ids, retrieved_ids, k),
            "mrr": mrr_at_k(relevant_ids, retrieved_ids, k),
            "ndcg": ndcg_at_k(relevant_ids, retrieved_ids, k),
        }

    return result_by_k


def evaluate_method(
    queries: list[dict[str, Any]],
    retrieval_function: Callable[[str, int], list[dict[str, Any]]],
    *,
    top_k_values: tuple[int, ...] = (1, 5, 10),
) -> dict[str, dict[str, float]]:
    aggregated = defaultdict(float)
    counts = defaultdict(int)

    for query in queries:
        metric_by_k = evaluate_query(query, retrieval_function, top_k_values=top_k_values)
        for k_label, metrics in metric_by_k.items():
            for metric_name, score in metrics.items():
                aggregated[f"{metric_name}{k_label}"] += score
                counts[f"{metric_name}{k_label}"] += 1

    final: dict[str, float] = {}
    for metric_name, total in aggregated.items():
        final[metric_name] = total / max(1, counts[metric_name])
    return final


def compare_methods(
    queries: list[dict[str, Any]],
    methods: dict[str, Callable[[str, int], list[dict[str, Any]]]],
    *,
    top_k_values: tuple[int, ...] = (1, 5, 10),
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for method_name, retrieval_function in methods.items():
        results[method_name] = evaluate_method(queries, retrieval_function, top_k_values=top_k_values)
    return results
