from __future__ import annotations

import math
import re
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


def context_precision_at_k(
    relevant: Iterable[str] | set[str],
    retrieved: Sequence[str],
    k: int,
) -> float:
    """Share of the top-k context passages judged relevant by the qrels."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    relevant_set = set(relevant)
    return sum(doc_id in relevant_set for doc_id in top_k) / len(top_k)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9'-]*\b", text.lower()))


def faithfulness_proxy(answer: str, context: str) -> float:
    """Estimate supported answer sentences using lexical context overlap.

    This is deliberately named a proxy: use a human or model judge for a
    semantic faithfulness score in a production evaluation.
    """
    sentences = [sentence.strip() for sentence in re.split(r"[.!?]+", answer) if sentence.strip()]
    context_tokens = _tokens(context)
    if not sentences:
        return 0.0
    supported = sum(bool(_tokens(sentence) & context_tokens) for sentence in sentences)
    return supported / len(sentences)


def answer_relevance_proxy(query: str, answer: str) -> float:
    """Estimate query-answer topical overlap as a lightweight proxy."""
    query_tokens = _tokens(query)
    answer_tokens = _tokens(answer)
    if not query_tokens or not answer_tokens:
        return 0.0
    return len(query_tokens & answer_tokens) / len(query_tokens)
