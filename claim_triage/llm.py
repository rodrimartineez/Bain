"""Thin client for the local Ollama server (chat with JSON-schema output, vision, embeddings)."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Protocol

import httpx

from .config import settings


class LLMError(RuntimeError):
    """Raised when the model server is unreachable or returns unusable output."""


class LLMClient(Protocol):
    def chat_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        *,
        images: list[Path] | None = None,
        model: str | None = None,
        max_tokens: int = 256,
        timeout_s: float | None = None,
    ) -> dict[str, Any]: ...


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaClient:
    """Implements both LLMClient and Embedder against Ollama's HTTP API."""

    def __init__(self, base_url: str | None = None, timeout_s: float | None = None) -> None:
        self.base_url = (base_url or settings.ollama_url).rstrip("/")
        self.model_name = settings.embedding_model
        self._http = httpx.Client(timeout=timeout_s or settings.llm_timeout_s)

    # ------------------------------------------------------------------ health
    def available_models(self) -> list[str]:
        try:
            resp = self._http.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama is not reachable at {self.base_url}: {exc}") from exc
        return [m["name"] for m in resp.json().get("models", [])]

    # ------------------------------------------------------------------ chat
    def chat_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        *,
        images: list[Path] | None = None,
        model: str | None = None,
        max_tokens: int = 256,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "user", "content": user}
        if images:
            message["images"] = [base64.b64encode(Path(p).read_bytes()).decode("ascii") for p in images]
        payload = {
            "model": model or settings.text_model,
            "stream": False,
            "format": schema,  # Ollama constrains decoding to this JSON schema
            "keep_alive": settings.llm_keep_alive,
            # Smaller context for vision calls: less KV-cache memory, so more layers fit on a small GPU.
            "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 2048 if images else 4096},
            "messages": [{"role": "system", "content": system}, message],
        }
        if images and settings.vision_num_gpu.strip():
            payload["options"]["num_gpu"] = int(settings.vision_num_gpu)
        try:
            resp = self._http.post(f"{self.base_url}/api/chat", json=payload, timeout=timeout_s or settings.llm_timeout_s)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM call failed ({payload['model']}): {exc}") from exc
        content = resp.json().get("message", {}).get("content", "")
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            # Usually means generation was truncated by max_tokens.
            raise LLMError(f"Model returned invalid JSON: {content[:200]!r}") from exc
        if not isinstance(data, dict):
            raise LLMError("Model returned JSON that is not an object")
        return data

    # ------------------------------------------------------------------ embeddings
    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            resp = self._http.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model_name, "input": texts, "keep_alive": settings.llm_keep_alive},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Embedding call failed ({self.model_name}): {exc}") from exc
        vectors = resp.json().get("embeddings")
        if not vectors or len(vectors) != len(texts):
            raise LLMError("Embedding server returned an unexpected number of vectors")
        return vectors
