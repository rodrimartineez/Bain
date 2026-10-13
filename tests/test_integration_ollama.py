"""Integration tests against the real local models. Excluded by default; run with:

pytest -m ollama
"""

import pytest

from claim_triage.config import settings
from claim_triage.knowledge_base import VectorIndex
from claim_triage.llm import LLMError, OllamaClient

pytestmark = pytest.mark.ollama


@pytest.fixture(scope="module")
def client():
    c = OllamaClient()
    try:
        c.available_models()
    except LLMError as exc:
        pytest.skip(f"Ollama not available: {exc}")
    return c


@pytest.fixture(scope="module")
def real_index(client):
    return VectorIndex.load_or_build(client)


@pytest.mark.parametrize(
    "query,expected_type",
    [
        ("water is dripping through my ceiling from the flat upstairs", "water_leak"),
        ("the wind storm blew tiles off the roof", "storm_wind"),
        ("frying pan fire burnt the kitchen", "fire"),
        ("river burst its banks and the ground floor is under water", "flood"),
    ],
)
def test_real_semantic_retrieval_top1_type(real_index, query, expected_type):
    hits = real_index.search(query, top_k=3)
    # storm-damaged roofs are legitimately filed as either storm_wind or roof in the knowledge base
    allowed = {expected_type, "roof"} if expected_type == "storm_wind" else {expected_type}
    assert hits[0].claim["damage_type"] in allowed


def test_off_topic_query_is_below_threshold(real_index):
    assert real_index.search("I lost my phone on the train", top_k=1, min_similarity=settings.min_similarity) == []


def test_real_description_extraction_is_schema_valid(client):
    from claim_triage.agents.intake import DESCRIPTION_SCHEMA, DESCRIPTION_SYSTEM
    from claim_triage.schemas import DescriptionFacts

    raw = client.chat_json(
        DESCRIPTION_SYSTEM, 'Claim description:\n"""A kitchen fire on 2026-10-01 burnt the cupboards."""', DESCRIPTION_SCHEMA
    )
    facts = DescriptionFacts.model_validate(raw)
    assert "fire" in [d.value for d in facts.damage_types]
