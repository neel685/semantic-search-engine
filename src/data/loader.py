from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from src.config import get_project_root, load_config
from src.data.preprocessing import clean_text, filter_valid_documents


def ensure_directory(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_msmarco_subset(
    config: dict[str, Any] | None = None,
    *,
    max_documents: int | None = None,
    max_queries: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    cfg = config or load_config()
    dataset_cfg = cfg.get("dataset", {})
    max_docs = max_documents if max_documents is not None else int(dataset_cfg.get("max_documents", 5000))
    max_qs = max_queries if max_queries is not None else int(dataset_cfg.get("max_queries", 1000))

    dataset = load_dataset("microsoft/ms_marco", "v1.1", split=f"train[:{max_docs}]")

    documents: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    qrels: dict[str, list[str]] = {}

    for row in dataset:
        query_id = str(row.get("query_id", ""))
        query_text = clean_text(row.get("query", ""))
        passages: dict[str, Any] = row.get("passages", {})
        passage_texts: list[str] = passages.get("passage_text", [])
        selected_flags: list[int] = passages.get("is_selected", [])

        if not passage_texts:
            continue

        query_doc_ids: list[str] = []
        for idx, passage in enumerate(passage_texts):
            doc_id = f"{query_id}_{idx}"
            cleaned_text = clean_text(passage)
            if not cleaned_text:
                continue
            documents.append({"id": doc_id, "text": cleaned_text})
            query_doc_ids.append(doc_id)

        relevant_ids = [
            f"{query_id}_{idx}"
            for idx, is_selected in enumerate(selected_flags)
            if idx < len(passage_texts) and is_selected == 1
        ]
        queries.append({
            "id": query_id,
            "text": query_text,
            "relevant_doc_ids": relevant_ids,
        })
        qrels[query_id] = relevant_ids

        if len(queries) >= max_qs:
            break

    documents = filter_valid_documents(documents)
    return documents, queries, qrels


def save_processed_data(documents: list[dict[str, Any]], queries: list[dict[str, Any]], qrels: dict[str, list[str]], output_dir: str | Path | None = None) -> Path:
    root = Path(output_dir) if output_dir else get_project_root() / "data" / "processed"
    ensure_directory(root)

    documents_path = root / "documents.jsonl"
    with documents_path.open("w", encoding="utf-8") as file:
        for doc in documents:
            file.write(json.dumps(doc, ensure_ascii=False) + "\n")

    queries_path = root / "queries.jsonl"
    with queries_path.open("w", encoding="utf-8") as file:
        for query in queries:
            file.write(json.dumps(query, ensure_ascii=False) + "\n")

    qrels_path = root / "qrels.json"
    with qrels_path.open("w", encoding="utf-8") as file:
        json.dump(qrels, file, ensure_ascii=False, indent=2)

    return root


def load_processed_data(processed_dir: str | Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    root = Path(processed_dir) if processed_dir else get_project_root() / "data" / "processed"
    docs_path = root / "documents.jsonl"
    queries_path = root / "queries.jsonl"
    qrels_path = root / "qrels.json"

    documents: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    qrels: dict[str, list[str]] = {}

    if docs_path.exists():
        with docs_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    documents.append(json.loads(line))

    if queries_path.exists():
        with queries_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    queries.append(json.loads(line))

    if qrels_path.exists():
        with qrels_path.open("r", encoding="utf-8") as file:
            qrels = json.load(file)

    return documents, queries, qrels
