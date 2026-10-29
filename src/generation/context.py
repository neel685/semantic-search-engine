from __future__ import annotations

from typing import Any


def build_context(results: list[dict[str, Any]], max_characters: int = 12000) -> str:
    chunks: list[str] = []
    total = 0
    for result in results:
        chunk = f"[{result['document_id']}]\n{result.get('text', '')}\n"
        if total + len(chunk) > max_characters:
            break
        chunks.append(chunk)
        total += len(chunk)
    return "\n".join(chunks)
