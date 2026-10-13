"""Triage Assessor agent.

Split of responsibilities (key design decision):
* Priority and confidence are computed by a transparent, deterministic rubric, so the same evidence
  always gives the same priority and every point can be traced to a source.
* The LLM only writes the narrative (summary, assessment, actions, information to request), which is
  schema-validated and screened for coverage/indemnity decisions; on failure a template is used.
"""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from ..llm import LLMClient, LLMError
from ..schemas import AssessmentNarrative, Confidence, Priority, PriorityFactor
from ..state import Assessment, ClaimState
from .base import Agent, Tool

SEVERITY_POINTS = {"critical": 3, "severe": 2, "moderate": 1}
LEVELS = [(6, "P1", "Critical"), (4, "P2", "High"), (2, "P3", "Medium"), (0, "P4", "Low")]

# Statements the assistant must never make: it supports triage, it does not decide claims.
FORBIDDEN = re.compile(
    r"\b(claim (is|will be) (approved|denied|accepted|rejected)|coverage (is )?(confirmed|granted|denied)|"
    r"(we|insurer) will (pay|reimburse)|is (fully )?covered|is not covered|liability (is )?(accepted|confirmed))\b",
    re.I,
)

NARRATIVE_SYSTEM = (
    "You are a claims triage assistant writing for a professional insurance adjuster. Use ONLY the JSON "
    "evidence provided. Say 'the claimant states' for declared facts and 'photos show' for observed facts. "
    "Reference historical claims by their ID when you use them. The priority is already decided: explain it, "
    "do not change it. Never decide coverage, liability or payment; recommend checks instead. Be concise: "
    "executive_summary max 3 sentences, preliminary_assessment max 4 sentences, "
    "each list item one short sentence. Do not request documents or photos that were already provided."
)

NARRATIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "executive_summary": {"type": "string"},
        "preliminary_assessment": {"type": "string"},
        "recommended_actions": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5},
        "information_to_request": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
    },
    "required": ["executive_summary", "preliminary_assessment", "recommended_actions", "information_to_request"],
}


def score_priority(state: ClaimState) -> Priority:
    factors: list[PriorityFactor] = []
    facts = state.description_facts
    images = [f for f in state.image_findings.values() if f.processed]

    hazards = list(facts.safety_hazards) if facts else []
    img_hazards = [h for f in images for h in f.hazards_visible]
    if hazards or img_hazards:
        src = "declared" if hazards else "observed in photos"
        factors.append(PriorityFactor(factor=f"Safety hazard: {', '.join((hazards + img_hazards)[:3])}", points=3, source=src))

    if facts and facts.habitability == "uninhabitable":
        factors.append(PriorityFactor(factor="Property declared uninhabitable", points=3, source="declared"))
    elif facts and facts.habitability == "partially_habitable":
        factors.append(PriorityFactor(factor="Property partially habitable", points=1, source="declared"))

    if facts and facts.ongoing_damage:
        factors.append(PriorityFactor(factor="Damage is ongoing / property exposed", points=2, source="declared"))

    sev = [f.severity for f in images if f.severity in SEVERITY_POINTS]
    if sev:
        worst = max(sev, key=lambda s: SEVERITY_POINTS[s])
        factors.append(
            PriorityFactor(factor=f"Visible severity: {worst}", points=SEVERITY_POINTS[worst], source="observed in photos")
        )
    elif state.latest_hits:
        # No usable photo: fall back to the severity of the closest precedent, clearly labelled.
        top = state.latest_hits[0]
        if top.severity in SEVERITY_POINTS:
            factors.append(
                PriorityFactor(
                    factor=f"Severity of closest precedent {top.claim_id}: {top.severity}",
                    points=max(SEVERITY_POINTS[top.severity] - 1, 0),
                    source="historical (inferred)",
                )
            )

    amounts = [state.submission.metadata.claimed_amount_eur, facts.estimated_loss_eur if facts else None]
    amounts += [max(d.amounts_eur) for d in state.document_findings.values() if d.processed and d.amounts_eur]
    loss = max((a for a in amounts if a), default=None)
    if loss and loss >= 25000:
        factors.append(PriorityFactor(factor=f"Estimated loss {loss:,.0f} EUR (>= 25k)", points=2, source="declared/document"))
    elif loss and loss >= 5000:
        factors.append(PriorityFactor(factor=f"Estimated loss {loss:,.0f} EUR (5k-25k)", points=1, source="declared/document"))

    if facts and facts.vulnerable_occupants:
        factors.append(PriorityFactor(factor="Vulnerable occupants", points=1, source="declared"))

    score = sum(f.points for f in factors)
    level, label = next((lvl, lab) for threshold, lvl, lab in LEVELS if score >= threshold)
    return Priority(level=level, label=label, score=score, factors=factors)


def score_confidence(state: ClaimState) -> Confidence:
    total_inputs = 1 + len(state.submission.images) + len(state.submission.documents)
    processed = 1 if state.description_facts else 0
    processed += sum(1 for f in state.image_findings.values() if f.processed)
    processed += sum(1 for d in state.document_findings.values() if d.processed)
    coverage = processed / total_inputs

    applicable = [c for c in state.consistency_checks if c.status != "not_applicable"]
    agreement = (sum(1 for c in applicable if c.status == "pass") / len(applicable)) if applicable else 0.5

    best = max((h.similarity for h in state.latest_hits), default=0.0)
    precedent = min(max((best - 0.55) / 0.25, 0.0), 1.0)

    facts = state.description_facts
    known = 0
    if facts:
        known = sum(
            [
                bool(facts.damage_types),
                bool(facts.cause),
                facts.habitability != "unknown",
                facts.estimated_loss_eur is not None or state.submission.metadata.claimed_amount_eur is not None,
                bool(facts.incident_date or state.submission.metadata.incident_date),
            ]
        )
    completeness = known / 5
    has_photo = 1.0 if any(f.processed for f in state.image_findings.values()) else 0.0

    components = {
        "input_coverage": round(coverage, 2),
        "evidence_agreement": round(agreement, 2),
        "precedent_support": round(precedent, 2),
        "fact_completeness": round(completeness, 2),
        "visual_evidence": has_photo,
    }
    score = round(sum(components.values()) / len(components), 2)
    level = "high" if score >= 0.75 else "medium" if score >= 0.5 else "low"
    # Contradictory evidence caps the level regardless of how complete the inputs are.
    if agreement < 0.5:
        level = "low"
    elif level == "high" and any(c.status == "fail" for c in applicable):
        level = "medium"
    return Confidence(level=level, score=score, components=components)


def human_review_reasons(state: ClaimState, priority: Priority, confidence: Confidence) -> list[str]:
    reasons = []
    if priority.level == "P1":
        reasons.append("P1 claim: immediate adjuster ownership required.")
    for c in state.consistency_checks:
        if c.status == "fail" or (
            c.status == "warn"
            and c.check
            in (
                "declared_vs_visual_damage",
                "claimed_amount_vs_documents",
                "precedent_support",
                "document_dates",
                "photo_authenticity",
                "inputs_processed",
            )
        ):
            reasons.append(f"{c.check}: {c.detail}")
    if confidence.level == "low":
        reasons.append("Low confidence: evidence is incomplete or contradictory.")
    return reasons


def template_narrative(state: ClaimState, priority: Priority) -> AssessmentNarrative:
    facts = state.description_facts
    damage = ", ".join(state.all_damage_types()) or "undetermined damage"
    hits = state.latest_hits
    precedent = (
        f" Closest precedent: {hits[0].claim_id} ({hits[0].outcome.replace('_', ' ')})."
        if hits
        else " No close historical precedent was found."
    )
    actions = [f"Assign to the {priority.label.lower()}-priority queue ({priority.level})."]
    if any(c.status in ("warn", "fail") for c in state.consistency_checks):
        actions.append("Review the evidence consistency warnings before progressing the claim.")
    if not state.image_findings:
        actions.append("Request photos of the damage.")
    requests = []
    if not state.submission.documents:
        requests.append("Repair estimate or invoice")
    if facts and not facts.incident_date and not state.submission.metadata.incident_date:
        requests.append("Exact date of the incident")
    return AssessmentNarrative(
        executive_summary=f"Claim reporting {damage}. Assigned {priority.level} ({priority.label}) by the rubric "
        f"with score {priority.score}.{precedent}",
        preliminary_assessment="Automatically generated summary (LLM narrative unavailable). See facts, checks "
        "and priority factors for the evidence.",
        recommended_actions=actions,
        information_to_request=requests,
    )


def _evidence_packet(state: ClaimState, priority: Priority) -> str:
    facts = state.description_facts
    packet = {
        "claimant_description": state.submission.description[:800],
        "form_metadata": state.submission.metadata.model_dump(exclude_none=True),
        "declared_facts": facts.model_dump(exclude={"dropped_unsupported"}, mode="json") if facts else None,
        "photo_findings": [
            f.model_dump(include={"filename", "damage_types", "observations", "severity", "hazards_visible"}, mode="json")
            for f in state.image_findings.values()
            if f.processed
        ],
        "documents": [
            d.model_dump(include={"filename", "document_type", "amounts_eur", "key_facts"})
            for d in state.document_findings.values()
            if d.processed
        ],
        "similar_historical_claims": [
            h.model_dump(include={"claim_id", "similarity", "damage_type", "outcome", "flags"}) for h in state.latest_hits[:4]
        ],
        "consistency_issues": [c.detail for c in state.consistency_checks if c.status in ("warn", "fail")],
        "priority": {"level": priority.level, "label": priority.label, "factors": [f.factor for f in priority.factors]},
        "limitations": state.limitations,
    }
    return json.dumps(packet, ensure_ascii=False)


def _screen(narrative: AssessmentNarrative, state: ClaimState) -> None:
    text = " ".join([narrative.executive_summary, narrative.preliminary_assessment, *narrative.recommended_actions])
    if FORBIDDEN.search(text):
        raise ValueError("narrative contains a coverage/payment decision")
    known = {h.claim_id for s in state.searches for h in s.hits}
    cited = set(re.findall(r"HC-\d{4}-\d{4}", text))
    if cited - known:
        raise ValueError(f"narrative cites claims that were not retrieved: {sorted(cited - known)}")


class TriageAssessorAgent(Agent):
    name = "triage_assessor"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def tools(self) -> list[Tool]:
        return [
            Tool(
                "assess_claim",
                self.name,
                "Score priority with the rubric, compute the confidence indicator and write the triage narrative. "
                "Requires at least one search and an up-to-date cross-check.",
                lambda s: (
                    [None]
                    if (s.description_facts or s.description_error)
                    and (s.searches or s.search_failures >= 2)
                    and s.assessments_run < 2
                    and s.checks_version == s.evidence_version
                    and (s.assessment is None or s.assessment.evidence_version != s.evidence_version)
                    else []
                ),
                self.assess,
            )
        ]

    def assess(self, state: ClaimState, _target: str | None) -> str:
        priority = score_priority(state)
        confidence = score_confidence(state)
        narrative, source, error = None, "llm", ""
        prompt = _evidence_packet(state, priority)
        for _attempt in range(2):  # one retry with the validation error fed back
            try:
                raw = self.llm.chat_json(NARRATIVE_SYSTEM, prompt + error, NARRATIVE_SCHEMA, max_tokens=700)
                candidate = AssessmentNarrative.model_validate(raw)
                _screen(candidate, state)
                narrative = candidate
                break
            except (LLMError, ValidationError, ValueError) as exc:
                error = f"\n\nYour previous answer was rejected: {str(exc)[:300]}. Fix it."
        if narrative is None:
            narrative, source = template_narrative(state, priority), "template_fallback"
            reason = error.split("rejected: ", 1)[-1].removesuffix(". Fix it.")[:160]
            state.add_limitation(f"LLM narrative failed validation twice ({reason}); a deterministic template was used.")
        state.assessments_run += 1
        state.assessment = Assessment(
            priority=priority,
            confidence=confidence,
            narrative=narrative,
            narrative_source=source,
            human_review_reasons=human_review_reasons(state, priority, confidence),
            evidence_version=state.evidence_version,
        )
        return (
            f"priority={priority.level} (score {priority.score}), confidence={confidence.level} "
            f"({confidence.score}), narrative={source}, human_review={bool(state.assessment.human_review_reasons)}"
        )
