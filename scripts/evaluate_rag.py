from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_project_root, load_config
from src.data.loader import load_processed_data
from src.evaluation.metrics import (
    answer_relevance_proxy,
    context_precision_at_k,
    faithfulness_proxy,
    recall_at_k,
)
from src.generation.context import build_context
from src.generation.ollama import OllamaGenerator
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker


def build_pipeline(documents: list[dict[str, Any]], config: dict[str, Any]) -> tuple[HybridRetriever, CrossEncoderReranker | None]:
    retrieval_cfg = config.get("retrieval", {})
    embedding_cfg = config.get("embedding", {})
    bm25 = BM25Retriever(
        documents,
        k1=float(retrieval_cfg.get("bm25_k1", 1.5)),
        b=float(retrieval_cfg.get("bm25_b", 0.75)),
    )
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
    reranking_cfg = config.get("reranking", {})
    reranker = (
        CrossEncoderReranker(
            model_name=reranking_cfg.get("model_name", "BAAI/bge-reranker-v2-m3"),
            device=embedding_cfg.get("device", "auto"),
            batch_size=int(reranking_cfg.get("batch_size", 16)),
        )
        if bool(reranking_cfg.get("enabled", True))
        else None
    )
    return hybrid, reranker


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG context quality and optional answer proxies.")
    parser.add_argument("--generate", action="store_true", help="Call Ollama and evaluate generated-answer proxies.")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N queries.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of reranked context passages.")
    args = parser.parse_args()

    config = load_config()
    dataset_cfg = config.get("dataset", {})
    processed_dir = get_project_root() / dataset_cfg.get("processed_path", "data/processed")
    documents, queries, _ = load_processed_data(processed_dir)
    if not queries:
        raise ValueError("No queries available. Run preprocessing first.")
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    hybrid, reranker = build_pipeline(documents, config)
    generation_cfg = config.get("generation", {})
    generator = None
    if args.generate:
        if generation_cfg.get("provider", "ollama") != "ollama":
            raise ValueError("Answer evaluation currently supports only the Ollama provider.")
        generator = OllamaGenerator(
            model_name=generation_cfg.get("model_name", "qwen3:8b"),
            base_url=generation_cfg.get("base_url", "http://localhost:11434"),
            temperature=float(generation_cfg.get("temperature", 0.0)),
            timeout=float(generation_cfg.get("timeout_seconds", 120)),
        )

    selected_queries = queries[:args.limit] if args.limit > 0 else queries
    rows: list[dict[str, Any]] = []
    for query in selected_queries:
        relevant = set(query.get("relevant_doc_ids", []))
        retrieval_started = time.perf_counter()
        candidates = hybrid.search(query["text"], top_k=max(args.top_k * 6, 20))
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        rerank_ms = 0.0
        if reranker is not None:
            rerank_started = time.perf_counter()
            context_results = reranker.rank(query["text"], candidates, top_k=args.top_k)
            rerank_ms = (time.perf_counter() - rerank_started) * 1000
        else:
            context_results = candidates[:args.top_k]

        context_ids = [str(item["document_id"]) for item in context_results]
        row: dict[str, Any] = {
            "query_id": str(query.get("id", "")),
            "context_recall": recall_at_k(relevant, context_ids, args.top_k),
            "context_precision": context_precision_at_k(relevant, context_ids, args.top_k),
            "retrieval_latency_ms": retrieval_ms,
            "reranking_latency_ms": rerank_ms,
        }
        if generator is not None:
            context = build_context(
                context_results,
                max_characters=int(generation_cfg.get("max_context_characters", 12000)),
            )
            generation_started = time.perf_counter()
            answer = generator.generate(query["text"], context)
            row.update({
                "answer": answer,
                "faithfulness_proxy": faithfulness_proxy(answer, context),
                "answer_relevance_proxy": answer_relevance_proxy(query["text"], answer),
                "generation_latency_ms": (time.perf_counter() - generation_started) * 1000,
            })
        rows.append(row)

    metric_names = ["context_recall", "context_precision", "retrieval_latency_ms", "reranking_latency_ms"]
    if args.generate:
        metric_names += ["faithfulness_proxy", "answer_relevance_proxy", "generation_latency_ms"]
    summary = {name: mean(row[name] for row in rows) for name in metric_names}
    report = {
        "configuration": {
            "queries_evaluated": len(rows),
            "top_k": args.top_k,
            "answer_generation": args.generate,
            "faithfulness_and_relevance": "lexical proxies; not semantic judge scores",
        },
        "summary": summary,
        "per_query": rows,
    }
    output_path = get_project_root() / "results" / "rag_evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("RAG evaluation summary")
    print(f"Queries: {len(rows)} | Context k: {args.top_k}")
    print(f"Context Recall@{args.top_k}: {summary['context_recall']:.4f}")
    print(f"Context Precision@{args.top_k}: {summary['context_precision']:.4f}")
    print(f"Retrieval latency: {summary['retrieval_latency_ms']:.2f} ms")
    print(f"Reranking latency: {summary['reranking_latency_ms']:.2f} ms")
    if args.generate:
        print(f"Faithfulness proxy: {summary['faithfulness_proxy']:.4f}")
        print(f"Answer relevance proxy: {summary['answer_relevance_proxy']:.4f}")
        print(f"Generation latency: {summary['generation_latency_ms']:.2f} ms")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
