"""Test doubles. These are ONLY used by the automated tests; the application always uses Ollama.

FakeEmbedder is a deterministic bag-of-words hashing embedder, so that vector-search tests run without
a model server. It is not semantic, which is why integration tests (marked `ollama`) also exist.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from typing import Any

import numpy as np

from claim_triage.llm import LLMError


class FakeEmbedder:
    model_name = "fake-hash-embedder"

    def __init__(self, dims: int = 256, fail: bool = False) -> None:
        self.dims = dims
        self.fail = fail
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.fail:
            raise LLMError("embedding server down")
        out = []
        for t in texts:
            v = np.zeros(self.dims, dtype=np.float32)
            for w in re.findall(r"[a-z]{3,}", t.lower().replace("search_query", "").replace("search_document", "")):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dims] += 1.0
            out.append(v.tolist())
        return out


Responder = Callable[[str, str, dict[str, Any]], dict[str, Any]]


class FakeLLM:
    """Routes each call to a canned response based on which agent's system prompt is used."""

    def __init__(self, overrides: dict[str, Responder | Exception] | None = None) -> None:
        self.overrides = overrides or {}
        self.calls: list[str] = []

    @staticmethod
    def kind(system: str, images: list | None) -> str:
        if images:
            return "image"
        if "coordinate a team" in system:
            return "planner"
        if "extract facts" in system:
            return "description"
        if "supporting document" in system:
            return "document"
        if "triage assistant" in system:
            return "narrative"
        return "unknown"

    def chat_json(
        self, system: str, user: str, schema: dict[str, Any], *, images=None, model=None, max_tokens=256, timeout_s=None
    ):
        kind = self.kind(system, images)
        self.calls.append(kind)
        override = self.overrides.get(kind)
        if isinstance(override, Exception):
            raise override
        if override is not None:
            return override(system, user, schema)
        return DEFAULTS[kind](system, user, schema)


def _planner_first_allowed(_s: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
    # Picks the first line of the ALLOWED ACTIONS menu, like a cooperative model would.
    first = user.split("ALLOWED ACTIONS:\n", 1)[1].splitlines()[0]
    m = re.match(r"- (\w+) target=(.*)", first)
    target = m.group(2).strip('"')
    return {"reason": "test planner picks the first allowed action", "action": m.group(1), "target": target}


DEFAULTS: dict[str, Responder] = {
    "planner": _planner_first_allowed,
    "description": lambda *_: {
        "damage_types": ["water_leak"],
        "cause": "burst pipe in bathroom above",
        "affected_areas": ["bedroom ceiling"],
        "incident_date": "2026-10-08",
        "safety_hazards": ["water in ceiling light"],
        "habitability": "partially_habitable",
        "ongoing_damage": False,
        "estimated_loss_eur": None,
        "vulnerable_occupants": None,
        "key_facts": ["Water came through the bedroom ceiling"],
    },
    "image": lambda *_: {
        "damage_visible": True,
        "damage_types": ["water_leak"],
        "observations": ["Brown stain on ceiling"],
        "severity": "moderate",
        "hazards_visible": [],
        "image_kind": "photograph",
    },
    "document": lambda *_: {"document_type": "repair_estimate", "key_facts": ["Total 4,350.00 EUR"]},
    "narrative": lambda *_: {
        "executive_summary": "The claimant states a burst pipe upstairs caused water damage to the bedroom ceiling.",
        "preliminary_assessment": "Photos show a ceiling stain consistent with the declared escape of water.",
        "recommended_actions": ["Arrange electrical safety check", "Confirm source of leak with neighbour"],
        "information_to_request": ["Neighbour insurer details"],
    },
}
