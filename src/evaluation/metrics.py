from __future__ import annotations

import math
from typing import Iterable, Sequence


def recall_at_k(relevant: Iterable[str] | set[str], retrieved: Sequence[str], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = set(retrieved[:k])
    return len(relevant_set & top_k) / len(relevant_set)


def mrr_at_k(relevant: Iterable[str] | set[str], retrieved: Sequence[str], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant: Iterable[str] | set[str], retrieved: Sequence[str], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0

    gains = [1.0 if doc_id in relevant_set else 0.0 for doc_id in retrieved[:k]]
    dcg = 0.0
    for rank, gain in enumerate(gains, start=1):
        dcg += (2**gain - 1) / math.log2(rank + 1)

    ideal_hits = min(k, len(relevant_set))
    idcg = 0.0
    for rank in range(1, ideal_hits + 1):
        idcg += (2**1 - 1) / math.log2(rank + 1)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg
