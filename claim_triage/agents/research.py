"""Knowledge Researcher agent: semantic search over the vector knowledge base of historical claims."""

from __future__ import annotations

import re
from collections import Counter
from statistics import median

from ..config import settings
from ..knowledge_base import VectorIndex
from ..llm import LLMError
from ..schemas import RetrievedClaim
from ..state import ClaimState, SearchRecord
from .base import Agent, Tool, ToolError

_STOP = set(
    "the a an and or of to in on at for with from by was were is are be been it its this that after before "
    "into onto during over under about my our we i they he she no not has have had there their as all".split()
)


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _STOP}


def build_query(state: ClaimState, focus: str | None = None) -> str:
    """Builds the semantic query from extracted facts (not just the raw text) so that visual
    evidence and documents also steer retrieval."""
    parts: list[str] = []
    facts = state.description_facts
    damage = state.all_damage_types()
    if damage:
        parts.append("Damage type: " + ", ".join(d.replace("_", " ") for d in damage) + ".")
    if facts and facts.cause:
        parts.append(f"Cause: {facts.cause}.")
    if facts and facts.affected_areas:
        parts.append("Affected: " + ", ".join(facts.affected_areas) + ".")
    if state.submission.metadata.property_type:
        parts.append(f"Property: {state.submission.metadata.property_type}.")
    parts.append(state.submission.description[:600])
    observations = [o for f in state.image_findings.values() for o in f.observations][:4]
    if observations:
        parts.append("Visible: " + "; ".join(observations) + ".")
    if focus:
        parts.append(f"Focus: {focus}.")
    return " ".join(parts)


def explain_similarity(state: ClaimState, claim: dict, query: str) -> list[str]:
    """Human-readable reasons shown next to each hit. This is an explanation layer only;
    ranking is purely by embedding cosine similarity."""
    reasons: list[str] = []
    damage = state.all_damage_types()
    if damage and claim["damage_type"] == damage[0]:
        reasons.append(f"same primary damage type ({claim['damage_type']})")
    elif claim["damage_type"] in damage:
        reasons.append(f"matches a secondary damage type ({claim['damage_type']})")
    prop = (state.submission.metadata.property_type or "").lower()
    if prop and prop == claim["property_type"]:
        reasons.append(f"same property type ({prop})")
    shared = sorted(_terms(query) & _terms(claim["description"] + " " + claim["cause"]))[:6]
    if shared:
        reasons.append("shared concepts: " + ", ".join(shared))
    return reasons


def historical_insights(hits: list[RetrievedClaim]) -> list[str]:
    """Aggregates outcomes of the retrieved precedents into statements an adjuster can verify."""
    if not hits:
        return ["No sufficiently similar historical claims were found; no precedent-based guidance."]
    insights = []
    outcomes = Counter(h.outcome for h in hits)
    ids = ", ".join(h.claim_id for h in hits)
    insights.append(
        f"Outcomes of {len(hits)} similar claims ({ids}): "
        + ", ".join(f"{k.replace('_', ' ')} x{v}" for k, v in outcomes.most_common())
    )
    flags = Counter(f for h in hits for f in h.flags)
    for flag, count in flags.most_common(3):
        examples = ", ".join(h.claim_id for h in hits if flag in h.flags)
        insights.append(f"{count}/{len(hits)} similar claims were flagged '{flag.replace('_', ' ')}' ({examples}).")
    settled = [h.settled_amount_eur for h in hits if h.settled_amount_eur]
    if settled:
        insights.append(f"Settled amounts in similar paid claims ranged {min(settled):,.0f}-{max(settled):,.0f} EUR.")
    days = [h.days_to_close for h in hits if h.days_to_close]
    if days:
        insights.append(f"Median time to close for similar claims: {median(days):.0f} days.")
    return insights


class KnowledgeResearcherAgent(Agent):
    name = "knowledge_researcher"

    def __init__(self, index: VectorIndex) -> None:
        self.index = index

    def tools(self) -> list[Tool]:
        return [
            Tool(
                "search_similar_claims",
                self.name,
                "Semantic (embedding) search of the historical claims knowledge base using all evidence gathered "
                "so far. Can be repeated once if new evidence arrived after the first search.",
                self._targets,
                self.search,
            )
        ]

    def _targets(self, state: ClaimState) -> list[str | None]:
        if state.description_facts is None and state.description_error is None:
            return []  # need at least the description analysed to form a good query
        if state.search_failures >= 2:
            return []
        if not state.searches:
            return [None]
        last = state.searches[-1]
        if len(state.searches) < settings.max_searches and last.evidence_version < state.evidence_version:
            return [None]
        return []

    def search(self, state: ClaimState, focus: str | None) -> str:
        query = build_query(state, focus)
        try:
            hits = self.index.search(query, top_k=settings.top_k, min_similarity=settings.min_similarity)
        except LLMError as exc:
            state.search_failures += 1
            if state.search_failures >= 2:
                state.add_limitation("Vector search failed repeatedly; no historical precedents were used.")
            raise ToolError(f"vector search failed: {exc}") from exc
        retrieved = [
            RetrievedClaim(
                claim_id=h.claim["claim_id"],
                similarity=h.similarity,
                damage_type=h.claim["damage_type"],
                severity=h.claim["severity"],
                outcome=h.claim["outcome"],
                settled_amount_eur=h.claim.get("settled_amount_eur"),
                days_to_close=h.claim.get("days_to_close"),
                summary=h.claim["description"][:220],
                flags=h.claim.get("flags", []),
                why_similar=explain_similarity(state, h.claim, query),
                synthetic=h.claim.get("synthetic", True),
            )
            for h in hits
        ]
        # Precedents are evidence too: bump first so checks/assessment become stale, but this search is not.
        state.bump_evidence()
        state.searches.append(SearchRecord(query=query, hits=retrieved, evidence_version=state.evidence_version))
        if not retrieved:
            state.add_limitation(
                f"No historical claim reached the similarity threshold ({settings.min_similarity}); "
                "triage is not supported by precedents."
            )
            return f"search #{len(state.searches)}: 0 claims above similarity {settings.min_similarity}"
        top = ", ".join(f"{r.claim_id}({r.similarity:.2f},{r.damage_type})" for r in retrieved)
        return f"search #{len(state.searches)}: {len(retrieved)} similar claims: {top}"
