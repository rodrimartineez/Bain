"""Vector knowledge base of historical claims.

Design: each claim is rendered to a canonical text, embedded with nomic-embed-text and stored as an
L2-normalised float32 matrix (``vectors.npy``) plus a manifest. Search is exact cosine similarity
(a single matrix-vector product). For a few hundred / thousand records this is faster and more
reproducible than an ANN index; the ``VectorIndex`` interface is where FAISS/pgvector would plug in.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import settings
from .llm import Embedder

# nomic-embed-text is trained with task prefixes; using them noticeably improves retrieval.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def claim_to_text(claim: dict[str, Any]) -> str:
    """Canonical text representation that is embedded for each historical claim."""
    return (
        f"Damage type: {claim['damage_type'].replace('_', ' ')}. Cause: {claim['cause']}. "
        f"Property: {claim['property_type']}. Description: {claim['description']} "
        f"Evidence: {'; '.join(claim['evidence'])}. Severity: {claim['severity']}."
    )


def load_claims(path: Path | None = None) -> list[dict[str, Any]]:
    data = json.loads((path or settings.kb_path).read_text(encoding="utf-8"))
    claims = data["claims"]
    ids = [c["claim_id"] for c in claims]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate claim_id in knowledge base")
    return claims


def _dataset_hash(claims: list[dict[str, Any]]) -> str:
    blob = json.dumps(claims, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class SearchHit:
    claim: dict[str, Any]
    similarity: float


class VectorIndex:
    def __init__(self, claims: list[dict[str, Any]], vectors: np.ndarray, embedder: Embedder) -> None:
        if len(claims) != vectors.shape[0]:
            raise ValueError("Claims and vectors are misaligned")
        self.claims = claims
        self.vectors = vectors
        self.embedder = embedder

    # ------------------------------------------------------------------ build / load
    @classmethod
    def build(cls, claims: list[dict[str, Any]], embedder: Embedder, batch_size: int = 16) -> VectorIndex:
        texts = [DOC_PREFIX + claim_to_text(c) for c in claims]
        rows: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            rows.extend(embedder.embed(texts[i : i + batch_size]))
        return cls(claims, _normalise(np.asarray(rows, dtype=np.float32)), embedder)

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "vectors.npy", self.vectors)
        manifest = {
            "embedding_model": self.embedder.model_name,
            "dimensions": int(self.vectors.shape[1]),
            "count": len(self.claims),
            "dataset_hash": _dataset_hash(self.claims),
            "claim_ids": [c["claim_id"] for c in self.claims],
        }
        (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load_or_build(cls, embedder: Embedder, kb_path: Path | None = None, index_dir: Path | None = None) -> VectorIndex:
        """Load the persisted index, rebuilding it if the dataset or embedding model changed."""
        claims = load_claims(kb_path)
        index_dir = index_dir or settings.index_dir
        manifest_path = index_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("dataset_hash") == _dataset_hash(claims) and manifest.get("embedding_model") == embedder.model_name:
                return cls(claims, np.load(index_dir / "vectors.npy"), embedder)
        index = cls.build(claims, embedder)
        index.save(index_dir)
        return index

    # ------------------------------------------------------------------ search
    def search(
        self,
        query: str,
        top_k: int = 5,
        damage_types: list[str] | None = None,
        min_similarity: float = 0.0,
    ) -> list[SearchHit]:
        query_vec = _normalise(np.asarray(self.embedder.embed([QUERY_PREFIX + query]), dtype=np.float32))[0]
        scores = self.vectors @ query_vec  # cosine similarity (rows are unit vectors)
        order = np.argsort(-scores)
        hits: list[SearchHit] = []
        for idx in order:
            claim = self.claims[int(idx)]
            if damage_types and claim["damage_type"] not in damage_types:
                continue
            score = float(scores[idx])
            if score < min_similarity:
                break  # sorted descending: nothing further can pass
            hits.append(SearchHit(claim=claim, similarity=round(score, 4)))
            if len(hits) >= top_k:
                break
        return hits


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
