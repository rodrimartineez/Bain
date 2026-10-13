"""Evidence Auditor agent: deterministic cross-checks between declared, observed and documented evidence.

Deliberately rule-based: consistency checks must be reproducible and explainable, which an LLM
judgement is not.
"""

from __future__ import annotations

import re
from datetime import date

from ..config import settings
from ..schemas import ConsistencyCheck
from ..state import ClaimState
from .base import Agent, Tool

# Damage types that are commonly reported together; used to avoid false "mismatch" warnings.
RELATED = {
    "water_leak": {"flood", "mold", "roof"},
    "flood": {"water_leak", "mold", "structural"},
    "storm_wind": {"roof", "hail", "water_leak", "structural"},
    "roof": {"storm_wind", "hail", "water_leak"},
    "hail": {"roof", "storm_wind"},
    "fire": {"smoke", "structural"},
    "smoke": {"fire"},
    "structural": {"roof", "flood", "storm_wind", "fire"},
    "mold": {"water_leak", "flood"},
}
RED_FLAGS = {"fraud_indicators", "evidence_inconsistency", "pre_existing_damage", "gradual_damage", "wear_and_tear"}


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", value)
    if m:
        y, mo, d = map(int, m.groups())
    else:
        m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", value)
        if not m:
            return None
        d, mo, y = map(int, m.groups())
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def run_checks(state: ClaimState, today: date | None = None) -> list[ConsistencyCheck]:
    today = today or date.today()
    checks: list[ConsistencyCheck] = []
    facts = state.description_facts
    declared = {d.value for d in facts.damage_types} if facts else set()
    processed_imgs = [f for f in state.image_findings.values() if f.processed]

    # 1. Declared vs visually observed damage type
    observed = {d.value for f in processed_imgs for d in f.damage_types}
    if not processed_imgs or not declared:
        checks.append(
            ConsistencyCheck(
                check="declared_vs_visual_damage",
                status="not_applicable",
                detail="Needs both an analysed description and at least one analysed photo.",
            )
        )
    elif declared & observed or any(observed & RELATED.get(d, set()) for d in declared):
        checks.append(
            ConsistencyCheck(
                check="declared_vs_visual_damage",
                status="pass",
                detail=f"Declared {sorted(declared)} consistent with observed {sorted(observed)}.",
            )
        )
    else:
        checks.append(
            ConsistencyCheck(
                check="declared_vs_visual_damage",
                status="warn",
                detail=f"Declared {sorted(declared)} but photos show {sorted(observed) or 'no classifiable damage'}.",
            )
        )

    # 2. Photos actually show damage
    if processed_imgs:
        no_damage = [f.filename for f in processed_imgs if f.damage_visible is False]
        if len(no_damage) == len(processed_imgs):
            checks.append(
                ConsistencyCheck(
                    check="photos_show_damage", status="fail", detail="None of the analysed photos shows visible damage."
                )
            )
        elif no_damage:
            checks.append(
                ConsistencyCheck(
                    check="photos_show_damage", status="warn", detail=f"No visible damage in: {', '.join(no_damage)}."
                )
            )
        else:
            checks.append(
                ConsistencyCheck(check="photos_show_damage", status="pass", detail="All analysed photos show visible damage.")
            )
        synthetic = [f.filename for f in processed_imgs if f.image_kind == "illustration_or_synthetic"]
        if synthetic:
            checks.append(
                ConsistencyCheck(
                    check="photo_authenticity",
                    status="warn",
                    detail=f"Vision model flagged as drawing/synthetic: {', '.join(synthetic)}.",
                )
            )

    # 3. Claimed amount vs document amounts
    claimed = state.submission.metadata.claimed_amount_eur or (facts.estimated_loss_eur if facts else None)
    doc_amounts = [a for d in state.document_findings.values() if d.processed for a in d.amounts_eur]
    if claimed and doc_amounts:
        closest = max(doc_amounts)  # totals are usually the largest figure in an estimate/invoice
        gap = abs(closest - claimed) / claimed
        status = "pass" if gap <= 0.25 else "warn"
        checks.append(
            ConsistencyCheck(
                check="claimed_amount_vs_documents",
                status=status,
                detail=f"Claimed {claimed:,.2f} EUR vs largest document amount {closest:,.2f} EUR ({gap:.0%} difference).",
            )
        )
    else:
        checks.append(
            ConsistencyCheck(
                check="claimed_amount_vs_documents",
                status="not_applicable",
                detail="Needs a claimed amount and a document with amounts.",
            )
        )

    # 4. Incident date sanity and agreement
    form_date = _parse_date(state.submission.metadata.incident_date)
    text_date = _parse_date(facts.incident_date if facts else None)
    incident = form_date or text_date
    if incident is None:
        checks.append(ConsistencyCheck(check="incident_date", status="warn", detail="Incident date unknown."))
    elif incident > today:
        checks.append(
            ConsistencyCheck(check="incident_date", status="fail", detail=f"Incident date {incident} is in the future.")
        )
    elif form_date and text_date and form_date != text_date:
        checks.append(
            ConsistencyCheck(
                check="incident_date", status="warn", detail=f"Form date {form_date} differs from description date {text_date}."
            )
        )
    else:
        checks.append(ConsistencyCheck(check="incident_date", status="pass", detail=f"Incident date {incident}."))
        doc_dates = [_parse_date(x) for d in state.document_findings.values() for x in d.dates]
        early = sorted({d for d in doc_dates if d and (incident - d).days > 1})
        if early:
            checks.append(
                ConsistencyCheck(
                    check="document_dates",
                    status="warn",
                    detail=f"Document(s) dated before the incident: {', '.join(map(str, early))}.",
                )
            )

    # 5. Precedent support
    hits = state.latest_hits
    if not state.searches:
        checks.append(ConsistencyCheck(check="precedent_support", status="not_applicable", detail="No search run yet."))
    elif not hits:
        checks.append(
            ConsistencyCheck(
                check="precedent_support",
                status="warn",
                detail=f"No historical claim above similarity {settings.min_similarity}.",
            )
        )
    else:
        flagged = sorted({f"{h.claim_id}:{f}" for h in hits[:3] for f in h.flags if f in RED_FLAGS})
        status = "warn" if flagged else "pass"
        detail = f"Top match {hits[0].claim_id} (similarity {hits[0].similarity:.2f})."
        if flagged:
            detail += " Close precedents carry red flags: " + ", ".join(flagged) + "."
        checks.append(ConsistencyCheck(check="precedent_support", status=status, detail=detail))

    # 6. Input processing completeness
    failed = [f.filename for f in state.image_findings.values() if not f.processed]
    failed += [d.filename for d in state.document_findings.values() if not d.processed]
    pending = state.pending_images + state.pending_documents
    if failed or pending:
        checks.append(
            ConsistencyCheck(
                check="inputs_processed", status="warn", detail=f"Unprocessed inputs: {', '.join(failed + pending)}."
            )
        )
    else:
        checks.append(ConsistencyCheck(check="inputs_processed", status="pass", detail="All submitted inputs were processed."))
    return checks


class EvidenceAuditorAgent(Agent):
    name = "evidence_auditor"

    def tools(self) -> list[Tool]:
        return [
            Tool(
                "cross_check_evidence",
                self.name,
                "Deterministic consistency checks: declared vs photo damage, amounts vs documents, dates, "
                "precedent red flags, unprocessed inputs.",
                lambda s: (
                    [None] if (s.description_facts or s.description_error) and s.checks_version != s.evidence_version else []
                ),
                self.cross_check,
            )
        ]

    def cross_check(self, state: ClaimState, _target: str | None) -> str:
        state.consistency_checks = run_checks(state)
        state.checks_version = state.evidence_version
        issues = [f"{c.check}={c.status}" for c in state.consistency_checks if c.status in ("warn", "fail")]
        return f"{len(state.consistency_checks)} checks; issues: {', '.join(issues) or 'none'}"
