"""Pydantic data contracts shared by tools, agents, the API and the UI."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "1.0"
# Values small models emit instead of an empty list / null.
_PLACEHOLDERS = {"", "unknown", "none", "n/a", "na", "null", "not stated", "not mentioned", "not specified"}


def _drop_placeholders(items: list[str]) -> list[str]:
    return [i.strip() for i in items if i and i.strip().lower() not in _PLACEHOLDERS]


def _clean_prose(text: str, max_sentences: int) -> str:
    """Removes repeated sentences (a known small-model failure) and caps the length."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    unique = list(dict.fromkeys(s for s in sentences if s))
    return " ".join(unique[:max_sentences])


class DamageType(StrEnum):
    WATER_LEAK = "water_leak"
    FLOOD = "flood"
    STORM_WIND = "storm_wind"
    HAIL = "hail"
    ROOF = "roof"
    FIRE = "fire"
    SMOKE = "smoke"
    STRUCTURAL = "structural"
    MOLD = "mold"
    VANDALISM = "vandalism"
    OTHER = "other"


DAMAGE_TYPE_VALUES = [d.value for d in DamageType]

Severity = Literal["minor", "moderate", "severe", "critical", "unknown"]
Habitability = Literal["habitable", "partially_habitable", "uninhabitable", "unknown"]


# --------------------------------------------------------------------------- intake


class ClaimMetadata(BaseModel):
    claim_id: str | None = None
    policy_number: str | None = None
    incident_date: str | None = None
    property_type: str | None = None
    location: str | None = None
    claimed_amount_eur: float | None = Field(default=None, ge=0)


class UploadedFile(BaseModel):
    filename: str
    path: str
    kind: Literal["image", "document"]
    size_bytes: int
    warning: str | None = None  # set by intake validation (e.g. unreadable image)


class ClaimSubmission(BaseModel):
    description: str
    metadata: ClaimMetadata = Field(default_factory=ClaimMetadata)
    images: list[UploadedFile] = Field(default_factory=list)
    documents: list[UploadedFile] = Field(default_factory=list)


# --------------------------------------------------------------------------- tool outputs


class DescriptionFacts(BaseModel):
    """Facts extracted from the claimant's own words (category: declared)."""

    damage_types: list[DamageType] = Field(default_factory=list)
    cause: str | None = None
    affected_areas: list[str] = Field(default_factory=list)
    incident_date: str | None = None
    safety_hazards: list[str] = Field(default_factory=list)
    habitability: Habitability = "unknown"
    ongoing_damage: bool | None = None
    estimated_loss_eur: float | None = None
    vulnerable_occupants: bool | None = None
    key_facts: list[str] = Field(default_factory=list)
    dropped_unsupported: list[str] = Field(
        default_factory=list, description="Extracted values removed because they were not found in the text"
    )

    @field_validator("affected_areas", "safety_hazards", "key_facts")
    @classmethod
    def _no_placeholders(cls, items: list[str]) -> list[str]:
        return _drop_placeholders(items)

    @field_validator("cause", "incident_date")
    @classmethod
    def _placeholder_to_none(cls, value: str | None) -> str | None:
        return None if value is None or value.strip().lower() in _PLACEHOLDERS else value.strip()


class ImageFinding(BaseModel):
    filename: str
    processed: bool
    damage_visible: bool | None = None
    damage_types: list[DamageType] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    severity: Severity = "unknown"
    hazards_visible: list[str] = Field(default_factory=list)
    image_kind: str = "unclear"
    error: str | None = None

    @field_validator("observations", "hazards_visible")
    @classmethod
    def _no_placeholders(cls, items: list[str]) -> list[str]:
        return _drop_placeholders(items)


class DocumentFinding(BaseModel):
    filename: str
    processed: bool
    document_type: str = "unknown"
    char_count: int = 0
    amounts_eur: list[float] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    excerpt: str = ""
    error: str | None = None


class RetrievedClaim(BaseModel):
    claim_id: str
    similarity: float
    damage_type: str
    severity: str
    outcome: str
    settled_amount_eur: float | None = None
    days_to_close: int | None = None
    summary: str
    flags: list[str] = Field(default_factory=list)
    why_similar: list[str] = Field(default_factory=list)
    synthetic: bool = True


class ConsistencyCheck(BaseModel):
    check: str
    status: Literal["pass", "warn", "fail", "not_applicable"]
    detail: str


# --------------------------------------------------------------------------- report


class Fact(BaseModel):
    statement: str
    category: Literal["observed", "declared", "document", "historical", "inferred"]
    source: str


class PriorityFactor(BaseModel):
    factor: str
    points: int
    source: str


class Priority(BaseModel):
    level: Literal["P1", "P2", "P3", "P4"]
    label: str
    score: int
    factors: list[PriorityFactor]
    scale: str = (
        "P1 Critical (score>=6): emergency response within 24h; P2 High (4-5): adjuster contact within 48h; "
        "P3 Medium (2-3): standard queue; P4 Low (0-1): fast-track / desk review"
    )


class Confidence(BaseModel):
    level: Literal["low", "medium", "high"]
    score: float = Field(ge=0, le=1)
    components: dict[str, float]
    note: str = (
        "Heuristic evidence-coverage indicator, NOT a calibrated probability. "
        "Capped at low if most checks disagree, at medium if any check fails."
    )


class AssessmentNarrative(BaseModel):
    """The only part of the report written by the LLM; validated and length-bounded."""

    executive_summary: str = Field(min_length=20, max_length=1200)
    preliminary_assessment: str = Field(min_length=20, max_length=1500)
    recommended_actions: list[str] = Field(min_length=1, max_length=8)
    information_to_request: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("recommended_actions", "information_to_request")
    @classmethod
    def _strip_items(cls, items: list[str]) -> list[str]:
        return list(dict.fromkeys(_drop_placeholders(items)))

    @field_validator("executive_summary", mode="before")
    @classmethod
    def _clean_summary(cls, text: str) -> str:
        return _clean_prose(text, 4) if isinstance(text, str) else text

    @field_validator("preliminary_assessment", mode="before")
    @classmethod
    def _clean_assessment(cls, text: str) -> str:
        return _clean_prose(text, 6) if isinstance(text, str) else text


class TraceStep(BaseModel):
    step: int
    agent: str
    action: str
    target: str | None = None
    reason: str
    decided_by: Literal["llm", "rules", "fallback_rules", "guardrail"]
    status: Literal["ok", "error", "skipped"]
    observation: str
    duration_ms: int


class TriageReport(BaseModel):
    schema_version: str = SCHEMA_VERSION
    report_id: str
    claim_id: str
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    executive_summary: str
    primary_damage_type: DamageType | None
    secondary_damage_types: list[DamageType] = Field(default_factory=list)
    facts: list[Fact]
    image_findings: list[ImageFinding]
    document_findings: list[DocumentFinding]
    similar_claims: list[RetrievedClaim]
    historical_insights: list[str]
    consistency_checks: list[ConsistencyCheck]
    preliminary_assessment: str
    priority: Priority
    recommended_actions: list[str]
    information_to_request: list[str]
    uncertainties: list[str]
    confidence: Confidence
    needs_human_review: bool
    human_review_reasons: list[str]
    narrative_source: Literal["llm", "template_fallback"]
    termination: Literal["completed", "max_steps_reached", "aborted"]
    trace: list[TraceStep]
    disclaimer: str = (
        "Decision-support output only. This report does not decide coverage, liability or indemnity; "
        "the responsible adjuster makes the final decision."
    )
