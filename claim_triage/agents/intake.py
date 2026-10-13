"""Intake Analyst agent: turns raw description, photos and documents into structured, sourced facts."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import ValidationError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from ..config import settings
from ..llm import LLMClient, LLMError
from ..schemas import DAMAGE_TYPE_VALUES, DescriptionFacts, DocumentFinding, ImageFinding
from ..state import ClaimState
from .base import Agent, Tool, ToolError

DESCRIPTION_SYSTEM = (
    "You extract facts from an insurance property-damage claim description. "
    "Only report what the text states. Use null or 'unknown' when something is not stated. "
    "Never guess amounts or dates. Keep strings short.\n"
    "Definitions:\n"
    "- habitability: 'uninhabitable' only if the whole home cannot be lived in; 'partially_habitable' if some "
    "rooms are unusable; 'habitable' if people live normally.\n"
    "- ongoing_damage: true only if damage is still happening now (active leak, rain still entering, open roof); "
    "false if the source was stopped.\n"
    "- safety_hazards: risks to people (water near electrics, exposed wiring, gas, structural collapse, fire).\n"
    "- vulnerable_occupants: true only if elderly, disabled, ill people or babies are mentioned.\n"
    "- damage_types: water_leak = escape of water from pipes/appliances; flood = water entering from outside; "
    "storm_wind = wind damage; roof = roof failure; fire; smoke; structural; mold; hail; vandalism; other."
)

DESCRIPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "damage_types": {"type": "array", "items": {"type": "string", "enum": DAMAGE_TYPE_VALUES}, "maxItems": 3},
        "cause": {"type": ["string", "null"]},
        "affected_areas": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "incident_date": {"type": ["string", "null"]},
        "safety_hazards": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        "habitability": {"type": "string", "enum": ["habitable", "partially_habitable", "uninhabitable", "unknown"]},
        "ongoing_damage": {"type": ["boolean", "null"]},
        "estimated_loss_eur": {"type": ["number", "null"]},
        "vulnerable_occupants": {"type": ["boolean", "null"]},
        "key_facts": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
    },
    "required": [
        "damage_types",
        "cause",
        "affected_areas",
        "incident_date",
        "safety_hazards",
        "habitability",
        "ongoing_damage",
        "estimated_loss_eur",
        "vulnerable_occupants",
        "key_facts",
    ],
}

IMAGE_SYSTEM = (
    "You are a property-damage claims inspector reviewing one photo submitted with an insurance claim. "
    "List the damage you can see (stains, discoloration, cracks, missing or broken parts, burn marks, water, debris) "
    "and any hazards. Only describe what is visible; do not guess hidden causes. If nothing is damaged, set "
    "damage_visible=false. Damage type hints: brown stains, rings, bubbling plaster or dripping on ceilings/walls "
    "-> water_leak; missing, displaced or broken roof tiles -> roof or storm_wind; char or flames -> fire; soot -> "
    "smoke; standing water or waterlines -> flood; large cracks or collapse -> structural; black or green growth -> "
    "mold. Separately, set image_kind to illustration_or_synthetic if the picture looks drawn or computer-generated."
)

IMAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "damage_visible": {"type": "boolean"},
        "damage_types": {"type": "array", "items": {"type": "string", "enum": DAMAGE_TYPE_VALUES}, "maxItems": 3},
        "observations": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        "severity": {"type": "string", "enum": ["minor", "moderate", "severe", "critical", "unknown"]},
        "hazards_visible": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "image_kind": {"type": "string", "enum": ["photograph", "illustration_or_synthetic", "document", "unclear"]},
    },
    "required": ["damage_visible", "damage_types", "observations", "severity", "hazards_visible", "image_kind"],
}

DOCUMENT_SYSTEM = (
    "You read a supporting document attached to a property-damage insurance claim. Classify it and "
    "list up to 4 short key facts that are written in it. Do not add anything that is not in the text."
)

DOCUMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "document_type": {
            "type": "string",
            "enum": [
                "repair_estimate",
                "invoice",
                "police_report",
                "fire_brigade_report",
                "weather_report",
                "expert_report",
                "policy_document",
                "other",
            ],
        },
        "key_facts": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
    },
    "required": ["document_type", "key_facts"],
}

_AMOUNT_RE = re.compile(
    r"(?:€|EUR)\s?(\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{2})?|\d+(?:[.,]\d{2})?)"
    r"|(\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{2})?|\d+(?:[.,]\d{2})?)\s?(?:€|EUR|euros?)",
    re.IGNORECASE,
)
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})\b")


def parse_amount(raw: str) -> float | None:
    """Parse '4,350.00', '4.350,00', '4 350' or '4350' into a float."""
    s = raw.replace(" ", "")
    if re.search(r"[.,]\d{2}$", s):
        whole, cents = s[:-3], s[-2:]
        whole = re.sub(r"[.,]", "", whole)
        s = f"{whole}.{cents}"
    else:
        s = re.sub(r"[.,]", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def find_amounts(text: str) -> list[float]:
    values = []
    for m in _AMOUNT_RE.finditer(text):
        value = parse_amount(m.group(1) or m.group(2))
        if value is not None and value > 0:
            values.append(value)
    return list(dict.fromkeys(values))


def _number_in_text(value: float, text: str) -> bool:
    """Grounding check: the extracted amount must literally appear in the source text."""
    candidates = {parse_amount(m) for m in re.findall(r"\d[\d.,\s]*\d|\d", text)}
    return any(c is not None and abs(c - value) < 0.5 for c in candidates)


def _date_in_text(value: str, text: str) -> bool:
    parts = re.findall(r"\d+", value)
    return bool(parts) and all(p.lstrip("0") in text or p in text for p in parts)


def read_document_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            reader = PdfReader(str(path))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except (PdfReadError, OSError, ValueError) as exc:
            raise ToolError(f"PDF could not be parsed ({exc.__class__.__name__})") from exc
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


class IntakeAnalystAgent(Agent):
    name = "intake_analyst"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def tools(self) -> list[Tool]:
        return [
            Tool(
                "analyze_description",
                self.name,
                "Extract declared facts (damage type, cause, hazards, habitability, loss) from the claimant's text.",
                lambda s: [None] if s.description_facts is None and s.description_error is None else [],
                self.analyze_description,
            ),
            Tool(
                "analyze_image",
                self.name,
                "Run the vision model on one damage photo to record visible damage and hazards.",
                lambda s: list(s.pending_images),
                self.analyze_image,
            ),
            Tool(
                "extract_document",
                self.name,
                "Extract text, amounts, dates and key facts from one supporting document.",
                lambda s: list(s.pending_documents),
                self.extract_document,
            ),
        ]

    # ------------------------------------------------------------------ description
    def analyze_description(self, state: ClaimState, _target: str | None) -> str:
        text = state.submission.description
        try:
            raw = self.llm.chat_json(DESCRIPTION_SYSTEM, f'Claim description:\n"""{text}"""', DESCRIPTION_SCHEMA, max_tokens=320)
            facts = DescriptionFacts.model_validate(raw)
        except (LLMError, ValidationError) as exc:
            state.description_error = str(exc)[:200]
            state.add_limitation("Description could not be analysed by the language model; facts rely on other inputs.")
            raise ToolError(f"description analysis failed: {exc}") from exc

        # Grounding: drop numeric values the model produced that are not in the text.
        if facts.estimated_loss_eur is not None and not _number_in_text(facts.estimated_loss_eur, text):
            facts.dropped_unsupported.append(f"estimated_loss_eur={facts.estimated_loss_eur}")
            facts.estimated_loss_eur = None
        if facts.incident_date and not _date_in_text(facts.incident_date, text):
            facts.dropped_unsupported.append(f"incident_date={facts.incident_date}")
            facts.incident_date = None
        state.description_facts = facts
        state.bump_evidence()
        dropped = f"; dropped unsupported: {', '.join(facts.dropped_unsupported)}" if facts.dropped_unsupported else ""
        return (
            f"damage={[d.value for d in facts.damage_types]}, cause={facts.cause!r}, "
            f"hazards={facts.safety_hazards}, habitability={facts.habitability}, "
            f"ongoing={facts.ongoing_damage}, loss={facts.estimated_loss_eur}{dropped}"
        )

    # ------------------------------------------------------------------ images
    def analyze_image(self, state: ClaimState, target: str | None) -> str:
        image = next((i for i in state.submission.images if i.filename == target), None)
        if image is None:
            raise ToolError(f"unknown image {target!r}")
        if image.warning:
            state.image_findings[image.filename] = ImageFinding(filename=image.filename, processed=False, error=image.warning)
            state.add_limitation(f"Image '{image.filename}' could not be analysed: {image.warning}.")
            state.bump_evidence()
            raise ToolError(f"{image.filename}: {image.warning}")
        try:
            raw = self.llm.chat_json(
                IMAGE_SYSTEM,
                "Inspect this claim photo.",
                IMAGE_SCHEMA,
                images=[Path(image.path)],
                model=settings.vision_model,
                max_tokens=220,
                timeout_s=settings.vision_timeout_s,
            )
            finding = ImageFinding(filename=image.filename, processed=True, **raw)
        except (LLMError, ValidationError, TypeError) as exc:
            state.image_findings[image.filename] = ImageFinding(filename=image.filename, processed=False, error=str(exc)[:200])
            state.add_limitation(f"Vision analysis failed for '{image.filename}'.")
            state.bump_evidence()
            raise ToolError(f"vision model failed on {image.filename}: {exc}") from exc
        finding.damage_types = list(dict.fromkeys(finding.damage_types)) if finding.damage_visible else []
        state.image_findings[image.filename] = finding
        state.bump_evidence()
        return (
            f"{image.filename}: damage_visible={finding.damage_visible}, types={[d.value for d in finding.damage_types]}, "
            f"severity={finding.severity}, hazards={finding.hazards_visible}, kind={finding.image_kind}"
        )

    # ------------------------------------------------------------------ documents
    def extract_document(self, state: ClaimState, target: str | None) -> str:
        doc = next((d for d in state.submission.documents if d.filename == target), None)
        if doc is None:
            raise ToolError(f"unknown document {target!r}")

        def fail(msg: str) -> ToolError:
            state.document_findings[doc.filename] = DocumentFinding(filename=doc.filename, processed=False, error=msg)
            state.add_limitation(f"Document '{doc.filename}': {msg}.")
            state.bump_evidence()
            return ToolError(f"{doc.filename}: {msg}")

        try:
            text = read_document_text(Path(doc.path))
        except ToolError as exc:
            raise fail(str(exc)) from exc
        text = re.sub(r"[ \t]+", " ", text).strip()
        if len(text) < 30:
            raise fail("no extractable text (possibly a scanned PDF; OCR is not supported)")
        text = text[: settings.max_document_chars]

        finding = DocumentFinding(
            filename=doc.filename,
            processed=True,
            char_count=len(text),
            amounts_eur=find_amounts(text),
            dates=list(dict.fromkeys(_DATE_RE.findall(text))),
            excerpt=text[:300],
        )
        try:
            raw = self.llm.chat_json(
                DOCUMENT_SYSTEM, f'Document \'{doc.filename}\':\n"""{text[:3000]}"""', DOCUMENT_SCHEMA, max_tokens=220
            )
            finding.document_type = raw.get("document_type", "other")
            finding.key_facts = [str(f) for f in raw.get("key_facts", [])][:4]
        except LLMError as exc:
            # Deterministic extraction (amounts/dates/excerpt) is still useful without the LLM.
            state.add_limitation(f"Document '{doc.filename}' was parsed but not classified by the LLM ({exc}).")
        state.document_findings[doc.filename] = finding
        state.bump_evidence()
        return (
            f"{doc.filename}: type={finding.document_type}, chars={finding.char_count}, "
            f"amounts_eur={finding.amounts_eur[:5]}, dates={finding.dates[:5]}"
        )
