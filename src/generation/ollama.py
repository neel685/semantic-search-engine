from __future__ import annotations

import json
from urllib import error, request


class OllamaGenerator:
    def __init__(
        self,
        model_name: str = "qwen3:8b",
        *,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        self.model_name = model_name
        self.endpoint = f"{base_url.rstrip('/')}/api/generate"
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, query: str, context: str) -> str:
        if not context.strip():
            return "I don't have enough information in the retrieved documents."

        prompt = (
            "You answer questions using only the supplied context. "
            "Do not use outside knowledge or invent facts. "
            "If the context is insufficient, say exactly: "
            '"I don\'t have enough information in the retrieved documents." '
            "Include the relevant source document IDs in square brackets.\n\n"
            f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
        )
        payload = json.dumps({
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }).encode("utf-8")
        http_request = request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unable to generate an answer with Ollama: {exc}") from exc

        answer = body.get("response")
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Ollama returned an empty answer.")
        return answer.strip()
