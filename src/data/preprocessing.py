from __future__ import annotations

import re
from typing import Any


def normalize_whitespace(text: str | None) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def clean_text(
    text: Any,
    *,
    lowercase: bool = True,
    remove_punctuation: bool = False,
    normalize_whitespace_flag: bool = True,
) -> str:
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\xa0", " ")
    text = text.replace("\r", " ").replace("\n", " ")

    if remove_punctuation:
        text = re.sub(r"[^\w\s]", " ", text)

    if lowercase:
        text = text.lower()

    if normalize_whitespace_flag:
        text = normalize_whitespace(text)

    return text.strip()


def preprocess_document(document: dict[str, Any], *, lowercase: bool = True) -> dict[str, Any]:
    document_id = str(document.get("id", document.get("document_id", "")))
    text = document.get("text")
    cleaned_text = clean_text(text, lowercase=lowercase)
    return {
        "id": document_id,
        "text": cleaned_text,
    }


def preprocess_query(query: str | None, *, lowercase: bool = True) -> str:
    return clean_text(query, lowercase=lowercase)


def filter_valid_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for document in documents:
        if not isinstance(document, dict):
            continue
        text = str(document.get("text", ""))
        if not text.strip():
            continue
        cleaned.append(preprocess_document(document))
    return cleaned
