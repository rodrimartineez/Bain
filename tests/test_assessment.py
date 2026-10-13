"""Priority rubric, grounding checks, narrative validation and report schema."""

import pytest
from pydantic import ValidationError

from claim_triage.llm import LLMError
from claim_triage.schemas import Priority, TriageReport
from tests.fakes import DEFAULTS, FakeLLM


def test_priority_factors_are_traceable(make_service, make_submission):
    report = make_service().triage(make_submission())
    p = report.priority
    assert p.score == sum(f.points for f in p.factors)
    assert any("Safety hazard" in f.factor and f.source == "declared" for f in p.factors)
    assert p.level in {"P1", "P2", "P3", "P4"}


def test_high_risk_claim_is_p1(make_service, make_submission):
    severe = lambda *_: (
        DEFAULTS["description"]()
        | {  # noqa: E731
            "habitability": "uninhabitable",
            "ongoing_damage": True,
            "safety_hazards": ["exposed wiring"],
        }
    )
    report = make_service(FakeLLM({"description": severe})).triage(make_submission())
    assert report.priority.level == "P1"
    assert report.needs_human_review


def test_minor_text_only_claim_is_low_priority(make_service, make_submission):
    minor = lambda *_: (
        DEFAULTS["description"]()
        | {  # noqa: E731
            "safety_hazards": [],
            "habitability": "habitable",
            "ongoing_damage": False,
        }
    )
    sub = make_submission(
        description="Dishwasher leaked a little water yesterday and lifted two kitchen tiles.",
        images=[],
        documents=[],
        claimed_amount_eur=None,
    )
    report = make_service(FakeLLM({"description": minor})).triage(sub)
    assert report.priority.level == "P4"
    assert any("No photos submitted" in u for u in report.uncertainties)


def test_hallucinated_amount_is_dropped(make_service, make_submission):
    invented = lambda *_: DEFAULTS["description"]() | {"estimated_loss_eur": 99999}  # noqa: E731
    report = make_service(FakeLLM({"description": invented})).triage(make_submission())
    assert any("99999" in u for u in report.uncertainties)
    assert not any("99,999" in f.statement for f in report.facts)


def test_invalid_narrative_json_retries_then_uses_template(make_service, make_submission):
    llm = FakeLLM({"narrative": lambda *_: {"executive_summary": "short"}})  # fails schema validation
    report = make_service(llm).triage(make_submission())
    assert llm.calls.count("narrative") == 2
    assert report.narrative_source == "template_fallback"
    assert report.recommended_actions


def test_narrative_with_coverage_decision_is_rejected(make_service, make_submission):
    decides = lambda *_: (
        DEFAULTS["narrative"]()
        | {  # noqa: E731
            "executive_summary": "The claim is approved and we will pay the full amount to the claimant."
        }
    )
    report = make_service(FakeLLM({"narrative": decides})).triage(make_submission())
    assert report.narrative_source == "template_fallback"
    assert "claim is approved" not in report.executive_summary.lower()
    assert "we will pay" not in report.executive_summary.lower()


def test_narrative_citing_unknown_claim_is_rejected(make_service, make_submission):
    cites = lambda *_: (
        DEFAULTS["narrative"]()
        | {  # noqa: E731
            "preliminary_assessment": "Very similar to precedent HC-1999-0001 which was paid in full."
        }
    )
    report = make_service(FakeLLM({"narrative": cites})).triage(make_submission())
    assert report.narrative_source == "template_fallback"


def test_narrative_llm_outage_uses_template(make_service, make_submission):
    report = make_service(FakeLLM({"narrative": LLMError("down")})).triage(make_submission())
    assert report.narrative_source == "template_fallback"
    assert report.termination == "completed"


def test_inconsistent_evidence_raises_review_flags(make_service, make_submission):
    no_damage = lambda *_: {
        "damage_visible": False,
        "damage_types": [],
        "observations": ["Clean intact tiles"],  # noqa: E731
        "severity": "unknown",
        "hazards_visible": [],
        "image_kind": "photograph",
    }
    from claim_triage.ingestion import RawFile
    from tests.conftest import pdf_bytes

    sub = make_submission(
        documents=[RawFile("invoice.pdf", pdf_bytes("Invoice 2026-03-14 tiling. TOTAL EUR 2,100.00"))], claimed_amount_eur=6000
    )
    report = make_service(FakeLLM({"image": no_damage})).triage(sub)
    status = {c.check: c.status for c in report.consistency_checks}
    assert status["photos_show_damage"] == "fail"
    assert status["claimed_amount_vs_documents"] == "warn"
    assert status["document_dates"] == "warn"
    assert report.needs_human_review
    assert report.confidence.level != "high"


def test_report_schema_rejects_bad_priority():
    with pytest.raises(ValidationError):
        Priority(level="P0", label="x", score=1, factors=[])


def test_report_has_disclaimer_and_is_json_serialisable(make_service, make_submission):
    report = make_service().triage(make_submission())
    data = report.model_dump(mode="json")
    assert "does not decide coverage" in data["disclaimer"]
    assert data["confidence"]["note"].startswith("Heuristic")
    assert TriageReport.model_validate_json(report.model_dump_json()).report_id == report.report_id


def test_placeholder_hazard_does_not_score_points(make_service, make_submission):
    placeholder = lambda *_: DEFAULTS["description"]() | {"safety_hazards": ["unknown", "None"]}
    report = make_service(FakeLLM({"description": placeholder})).triage(make_submission())
    assert not any("Safety hazard" in f.factor for f in report.priority.factors)


def test_repeated_sentences_in_narrative_are_removed():
    from claim_triage.schemas import AssessmentNarrative

    n = AssessmentNarrative(
        executive_summary="Photos show no damage. Photos show no damage. Photos show no damage. Amount differs.",
        preliminary_assessment="The claimant states a leak happened in the bathroom.",
        recommended_actions=["Inspect", "Inspect"],
    )
    assert n.executive_summary == "Photos show no damage. Amount differs."
    assert n.recommended_actions == ["Inspect"]
