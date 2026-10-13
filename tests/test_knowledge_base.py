import json

import numpy as np
import pytest

from claim_triage.knowledge_base import VectorIndex, claim_to_text, load_claims
from tests.fakes import FakeEmbedder


def test_dataset_is_marked_synthetic_and_well_formed():
    claims = load_claims()
    assert len(claims) >= 30
    assert all(c["synthetic"] is True for c in claims)
    required = {"claim_id", "description", "damage_type", "evidence", "severity", "outcome", "cause", "property_type"}
    assert all(required <= c.keys() for c in claims)
    assert len({c["damage_type"] for c in claims}) >= 8


def test_vectors_are_unit_normalised(index):
    norms = np.linalg.norm(index.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_search_ranks_by_cosine_similarity(index):
    target = next(c for c in index.claims if c["claim_id"] == "HC-2023-0018")
    hits = index.search(claim_to_text(target), top_k=3)
    assert hits[0].claim["claim_id"] == "HC-2023-0018"
    assert hits[0].similarity >= hits[1].similarity >= hits[2].similarity


def test_threshold_returns_no_results_for_unrelated_query(index):
    assert index.search("quantum chromodynamics lecture notes", top_k=5, min_similarity=0.5) == []


def test_damage_type_filter(index):
    hits = index.search("water damage", top_k=5, damage_types=["fire"])
    assert hits and all(h.claim["damage_type"] == "fire" for h in hits)


def test_persisted_index_is_reused_and_rebuilt_on_change(tmp_path):
    kb = tmp_path / "kb.json"
    claims = load_claims()[:5]
    kb.write_text(json.dumps({"claims": claims}), encoding="utf-8")
    emb = FakeEmbedder()
    VectorIndex.load_or_build(emb, kb_path=kb, index_dir=tmp_path / "idx")
    calls = emb.calls
    VectorIndex.load_or_build(emb, kb_path=kb, index_dir=tmp_path / "idx")
    assert emb.calls == calls, "unchanged dataset must load from disk"
    kb.write_text(json.dumps({"claims": claims[:4]}), encoding="utf-8")
    idx = VectorIndex.load_or_build(emb, kb_path=kb, index_dir=tmp_path / "idx")
    assert emb.calls > calls and len(idx.claims) == 4


def test_duplicate_ids_rejected(tmp_path):
    c = load_claims()[0]
    kb = tmp_path / "kb.json"
    kb.write_text(json.dumps({"claims": [c, c]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_claims(kb)
