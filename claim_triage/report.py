"""Compiles the validated TriageReport from the claim state."""

from __future__ import annotations

import uuid

from .agents.assessor import human_review_reasons, score_confidence, score_priority, template_narrative
from .agents.research import historical_insights
from .schemas import DamageType, Fact, TriageReport
from .state import Assessment, ClaimState


def _facts(state: ClaimState) -> list[Fact]:
    facts: list[Fact] = []
    d = state.description_facts
    if d:
        if d.cause:
            facts.append(Fact(statement=f"Cause stated: {d.cause}", category="declared", source="claim description"))
        if d.affected_areas:
            facts.append(
                Fact(statement="Affected areas: " + ", ".join(d.affected_areas), category="declared", source="claim description")
            )
        for h in d.safety_hazards:
            facts.append(Fact(statement=f"Safety hazard reported: {h}", category="declared", source="claim description"))
        if d.habitability != "unknown":
            facts.append(
                Fact(
                    statement=f"Habitability: {d.habitability.replace('_', ' ')}", category="declared", source="claim description"
                )
            )
        if d.ongoing_damage is not None:
            facts.append(
                Fact(
                    statement=f"Damage ongoing: {'yes' if d.ongoing_damage else 'no'}",
                    category="declared",
                    source="claim description",
                )
            )
        if d.estimated_loss_eur is not None:
            facts.append(
                Fact(
                    statement=f"Estimated loss stated: {d.estimated_loss_eur:,.2f} EUR",
                    category="declared",
                    source="claim description",
                )
            )
        for k in d.key_facts:
            facts.append(Fact(statement=k, category="declared", source="claim description"))
    m = state.submission.metadata
    if m.claimed_amount_eur is not None:
        facts.append(
            Fact(statement=f"Claimed amount: {m.claimed_amount_eur:,.2f} EUR", category="declared", source="intake form")
        )
    if m.incident_date:
        facts.append(Fact(statement=f"Incident date: {m.incident_date}", category="declared", source="intake form"))
    for f in state.image_findings.values():
        for obs in f.observations if f.processed else []:
            facts.append(Fact(statement=obs, category="observed", source=f"photo {f.filename} (vision model)"))
    for doc in state.document_findings.values():
        if doc.processed:
            for k in doc.key_facts:
                facts.append(Fact(statement=k, category="document", source=f"document {doc.filename}"))
            if doc.amounts_eur:
                facts.append(
                    Fact(
                        statement="Amounts found: " + ", ".join(f"{a:,.2f} EUR" for a in doc.amounts_eur[:5]),
                        category="document",
                        source=f"document {doc.filename} (regex)",
                    )
                )
    for h in state.latest_hits[:3]:
        facts.append(
            Fact(
                statement=f"Similar claim {h.claim_id} ({h.damage_type}) ended {h.outcome.replace('_', ' ')}",
                category="historical",
                source=f"knowledge base, similarity {h.similarity:.2f}",
            )
        )
    return facts


def build_report(state: ClaimState, termination: str) -> TriageReport:
    assessment = state.assessment
    if assessment is None:
        # Controlled termination (e.g. step limit reached): deterministic parts only.
        priority = score_priority(state)
        confidence = score_confidence(state)
        assessment = Assessment(
            priority=priority,
            confidence=confidence,
            narrative=template_narrative(state, priority),
            narrative_source="template_fallback",
            human_review_reasons=human_review_reasons(state, priority, confidence),
            evidence_version=state.evidence_version,
        )
    reasons = list(assessment.human_review_reasons)
    if termination != "completed":
        reasons.append(f"Run ended early ({termination}); analysis incomplete.")

    damage = state.all_damage_types()
    uncertainties = list(state.limitations)
    if state.description_facts and state.description_facts.dropped_unsupported:
        uncertainties.append(
            "Values proposed by the model but not found in the text were discarded: "
            + ", ".join(state.description_facts.dropped_unsupported)
        )
    if not state.submission.images:
        uncertainties.append("No photos submitted: damage extent is unverified.")
    if not state.submission.documents:
        uncertainties.append("No supporting documents submitted: amounts are unverified.")
    uncertainties.append("Historical claims are SYNTHETIC demo data; precedent statistics are illustrative only.")

    n = assessment.narrative
    return TriageReport(
        report_id=f"TR-{uuid.uuid4().hex[:8].upper()}",
        claim_id=state.claim_id,
        executive_summary=n.executive_summary,
        primary_damage_type=DamageType(damage[0]) if damage else None,
        secondary_damage_types=[DamageType(d) for d in damage[1:]],
        facts=_facts(state),
        image_findings=list(state.image_findings.values()),
        document_findings=list(state.document_findings.values()),
        similar_claims=state.latest_hits,
        historical_insights=historical_insights(state.latest_hits),
        consistency_checks=state.consistency_checks,
        preliminary_assessment=n.preliminary_assessment,
        priority=assessment.priority,
        recommended_actions=n.recommended_actions,
        information_to_request=n.information_to_request,
        uncertainties=uncertainties,
        confidence=assessment.confidence,
        needs_human_review=bool(reasons),
        human_review_reasons=reasons,
        narrative_source=assessment.narrative_source,
        termination=termination,
        trace=state.trace,
    )
