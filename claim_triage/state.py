"""Mutable working memory shared by the coordinator and the specialist agents for one claim."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schemas import (
    AssessmentNarrative,
    ClaimSubmission,
    Confidence,
    ConsistencyCheck,
    DescriptionFacts,
    DocumentFinding,
    ImageFinding,
    Priority,
    RetrievedClaim,
    TraceStep,
)


@dataclass
class SearchRecord:
    query: str
    hits: list[RetrievedClaim]
    evidence_version: int


@dataclass
class Assessment:
    priority: Priority
    confidence: Confidence
    narrative: AssessmentNarrative
    narrative_source: str
    human_review_reasons: list[str]
    evidence_version: int


@dataclass
class ClaimState:
    claim_id: str
    submission: ClaimSubmission
    description_facts: DescriptionFacts | None = None
    description_error: str | None = None
    image_findings: dict[str, ImageFinding] = field(default_factory=dict)
    document_findings: dict[str, DocumentFinding] = field(default_factory=dict)
    searches: list[SearchRecord] = field(default_factory=list)
    search_failures: int = 0
    consistency_checks: list[ConsistencyCheck] = field(default_factory=list)
    checks_version: int = -1
    assessment: Assessment | None = None
    assessments_run: int = 0
    trace: list[TraceStep] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    # Incremented every time new evidence is added; lets the coordinator know when a
    # search / cross-check / assessment is stale and worth re-running.
    evidence_version: int = 0
    finished: bool = False

    # ------------------------------------------------------------------ helpers
    @property
    def pending_images(self) -> list[str]:
        return [i.filename for i in self.submission.images if i.filename not in self.image_findings]

    @property
    def pending_documents(self) -> list[str]:
        return [d.filename for d in self.submission.documents if d.filename not in self.document_findings]

    @property
    def latest_hits(self) -> list[RetrievedClaim]:
        return self.searches[-1].hits if self.searches else []

    def add_limitation(self, text: str) -> None:
        if text not in self.limitations:
            self.limitations.append(text)

    def bump_evidence(self) -> None:
        self.evidence_version += 1

    def all_damage_types(self) -> list[str]:
        """Damage types in order of appearance: declared first, then visually observed."""
        seen: list[str] = []
        if self.description_facts:
            seen += [d.value for d in self.description_facts.damage_types]
        for f in self.image_findings.values():
            seen += [d.value for d in f.damage_types]
        return list(dict.fromkeys(seen))

    def summary_for_planner(self) -> dict[str, Any]:
        """Compact view of the state given to the LLM coordinator (keeps prompts short)."""
        facts = self.description_facts
        return {
            "description_analyzed": facts is not None or self.description_error is not None,
            "declared_damage_types": [d.value for d in facts.damage_types] if facts else None,
            "images_total": len(self.submission.images),
            "images_pending": self.pending_images,
            "image_damage_types": sorted({d.value for f in self.image_findings.values() for d in f.damage_types}),
            "documents_total": len(self.submission.documents),
            "documents_pending": self.pending_documents,
            "searches_done": len(self.searches),
            "best_similarity": max((h.similarity for h in self.latest_hits), default=None),
            "new_evidence_since_last_search": bool(self.searches) and self.searches[-1].evidence_version < self.evidence_version,
            "cross_check_up_to_date": self.checks_version == self.evidence_version,
            "check_warnings": [c.check for c in self.consistency_checks if c.status in ("warn", "fail")],
            "assessment_up_to_date": self.assessment is not None and self.assessment.evidence_version == self.evidence_version,
            "tool_errors": [t.action for t in self.trace if t.status == "error"],
        }
