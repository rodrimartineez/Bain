"""ReAct coordinator behaviour: real tool selection, guardrails, fallbacks and termination."""

from claim_triage.ingestion import RawFile
from claim_triage.llm import LLMError
from claim_triage.schemas import TriageReport
from tests.conftest import pdf_bytes
from tests.fakes import DEFAULTS, FakeLLM


def actions(report: TriageReport) -> list[str]:
    return [s.action for s in report.trace]


def test_full_run_executes_every_tool_and_finishes(make_service, make_submission):
    report = make_service().triage(make_submission())
    acts = actions(report)
    assert report.termination == "completed"
    assert acts[-1] == "finish"
    for tool in [
        "analyze_description",
        "analyze_image",
        "extract_document",
        "search_similar_claims",
        "cross_check_evidence",
        "assess_claim",
    ]:
        assert tool in acts
    assert report.similar_claims, "vector search results must reach the report"
    assert all(s.status == "ok" for s in report.trace)
    TriageReport.model_validate(report.model_dump())  # round-trips through the schema


def test_llm_choice_is_respected(make_service, make_submission):
    """The planner (not a fixed sequence) decides: here it prefers documents first."""

    def prefer_docs(_s, user, _schema):
        menu = user.split("ALLOWED ACTIONS:\n", 1)[1].splitlines()
        line = next((m for m in menu if "extract_document" in m), menu[0])
        action, target = line[2:].split(" target=")
        return {"reason": "documents first", "action": action, "target": target.strip('"')}

    report = make_service(FakeLLM({"planner": prefer_docs})).triage(make_submission())
    assert report.trace[0].action == "extract_document"
    assert report.trace[0].decided_by == "llm"


def test_invalid_llm_action_falls_back_to_rules(make_service, make_submission):
    bad = lambda *_: {"reason": "x", "action": "delete_database", "target": ""}  # noqa: E731
    report = make_service(FakeLLM({"planner": bad})).triage(make_submission())
    assert report.termination == "completed"
    assert any(s.decided_by == "fallback_rules" and "invalid action" in s.reason for s in report.trace)
    assert "delete_database" not in actions(report)


def test_planner_outage_falls_back_and_completes(make_service, make_submission):
    report = make_service(FakeLLM({"planner": LLMError("timeout")})).triage(make_submission())
    assert report.termination == "completed"
    assert any(s.decided_by == "fallback_rules" for s in report.trace)


def test_no_action_is_repeated_unnecessarily(make_service, make_submission):
    report = make_service().triage(make_submission())
    acts = actions(report)
    assert acts.count("analyze_description") == 1
    assert acts.count("analyze_image") == 1
    assert acts.count("search_similar_claims") <= 2
    assert acts.count("assess_claim") <= 2


def test_research_reruns_when_new_evidence_arrives_after_search(make_service, make_submission):
    """If the planner searches before analysing the photo, the photo is new evidence -> re-search is offered."""

    def search_early(_s, user, _schema):
        menu = user.split("ALLOWED ACTIONS:\n", 1)[1].splitlines()
        order = ["analyze_description", "search_similar_claims", "analyze_image"]
        for wanted in order:
            line = next((m for m in menu if m.startswith(f"- {wanted}")), None)
            if line:
                break
        line = line or menu[0]
        action, target = line[2:].split(" target=")
        return {"reason": "search early", "action": action, "target": target.strip('"')}

    report = make_service(FakeLLM({"planner": search_early})).triage(make_submission(documents=[]))
    assert actions(report).count("search_similar_claims") == 2


def test_step_limit_terminates_in_a_controlled_way(make_service, make_submission):
    report = make_service(max_steps=3).triage(make_submission())
    assert report.termination == "max_steps_reached"
    assert len(report.trace) == 3
    assert report.needs_human_review
    assert any("step limit" in u for u in report.uncertainties)
    assert report.narrative_source == "template_fallback"


def test_failing_vision_tool_is_recorded_and_run_continues(make_service, make_submission):
    report = make_service(FakeLLM({"image": LLMError("vision model crashed")})).triage(make_submission())
    img_step = next(s for s in report.trace if s.action == "analyze_image")
    assert img_step.status == "error"
    assert report.termination == "completed"
    assert any("Vision analysis failed" in u for u in report.uncertainties)
    assert any(c.check == "inputs_processed" and c.status == "warn" for c in report.consistency_checks)
    assert report.needs_human_review and any("inputs_processed" in r for r in report.human_review_reasons)


def test_corrupt_image_and_textless_pdf_become_limitations(make_service, make_submission):
    sub = make_submission(
        images=[RawFile("broken.jpg", b"\xff\xd8\xff garbage")], documents=[RawFile("scan.pdf", pdf_bytes(None))]
    )
    report = make_service().triage(sub)
    assert report.termination == "completed"
    assert not report.image_findings[0].processed
    assert "no extractable text" in report.document_findings[0].error
    assert report.confidence.components["input_coverage"] < 1


def test_vector_search_outage_is_bounded(index, make_service, make_submission):
    index.embedder.fail = True
    report = make_service().triage(make_submission())
    assert actions(report).count("search_similar_claims") == 2  # retried once, then given up
    assert report.similar_claims == []
    assert report.termination == "completed"
    assert any("Vector search failed" in u for u in report.uncertainties)


def test_rules_mode_makes_no_planner_calls(make_service, make_submission):
    llm = FakeLLM()
    report = make_service(llm, mode="rules").triage(make_submission())
    assert "planner" not in llm.calls
    assert all(s.decided_by == "rules" for s in report.trace)
    assert report.termination == "completed"


def test_wrong_target_for_valid_action_is_repaired_not_discarded(make_service, make_submission):
    """qwen2.5 was observed returning analyze_description with an image filename as target."""
    calls = {"n": 0}

    def wrong_target(_s, user, _schema):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"reason": "start with the text", "action": "analyze_description", "target": "ceiling.png"}
        return DEFAULTS["planner"](_s, user, _schema)

    report = make_service(FakeLLM({"planner": wrong_target})).triage(make_submission())
    first = report.trace[0]
    assert first.action == "analyze_description" and first.target is None
    assert first.decided_by == "llm" and "guardrail" in first.reason
