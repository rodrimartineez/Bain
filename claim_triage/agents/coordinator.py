"""Coordinator agent: the ReAct loop.

Each iteration:  observe state -> compute the valid actions (tool preconditions) -> REASON: the policy
(LLM by default) picks one action and states why -> ACT: run the tool -> OBSERVE: record the result in
the state and the trace -> repeat until `finish` or the step limit.

Guardrails: the LLM can only choose among actions whose preconditions hold (enforced twice: by the
JSON-schema enum at decoding time and by validation afterwards); completed actions disappear from the
menu, so the loop cannot repeat work; invalid or failed LLM decisions fall back to a deterministic
policy; a hard step limit ends the run in a controlled way with an explicit 'max_steps_reached' report.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass

from ..config import settings
from ..llm import LLMClient, LLMError
from ..report import build_report
from ..schemas import TraceStep, TriageReport
from ..state import ClaimState
from .base import Agent, Tool, ToolError

Option = tuple[str, str | None]  # (action, target)

RULE_ORDER = [
    "analyze_description",
    "analyze_image",
    "extract_document",
    "search_similar_claims",
    "cross_check_evidence",
    "assess_claim",
    "finish",
]

PLANNER_SYSTEM = """You coordinate a team of agents that triage property-damage insurance claims.
You run a ReAct loop: each turn, pick exactly ONE next action from ALLOWED ACTIONS and give a short reason.
Goal: analyse every submitted input, retrieve similar historical claims using the most complete evidence,
cross-check the evidence, assess the claim, then finish.
Guidance:
- Similar-claim search works best after the description AND photos are analysed.
- Repeat the search only if new evidence (e.g. photos showing another damage type) arrived after it.
- Unanalysed inputs make the assessment weaker; analyse them before assessing.
- Choose finish only when the assessment is up to date.
Tools:
{tools}
Reason in at most 20 words. Answer in JSON."""


@dataclass
class Decision:
    action: str
    target: str | None
    reason: str
    decided_by: str


class RulePolicy:
    """Deterministic planner: first valid action in a fixed priority order. Used in 'rules' mode,
    as the fallback when the LLM fails, and as an oracle in tests."""

    def decide(self, state: ClaimState, options: list[Option], decided_by: str = "rules", note: str = "") -> Decision:
        action, target = min(options, key=lambda o: RULE_ORDER.index(o[0]) if o[0] in RULE_ORDER else 99)
        reason = f"{note}Next pending step in the standard order: {action}" + (f" on {target}" if target else "")
        return Decision(action, target, reason, decided_by)


class LLMPolicy:
    def __init__(self, llm: LLMClient, tools: list[Tool]) -> None:
        self.llm = llm
        self.fallback = RulePolicy()
        tool_lines = "\n".join(f"- {t.name} ({t.agent}): {t.description}" for t in tools)
        tool_lines += "\n- finish (coordinator): compile and validate the final triage report."
        self.system = PLANNER_SYSTEM.format(tools=tool_lines)

    def decide(self, state: ClaimState, options: list[Option]) -> Decision:
        if len(options) == 1:  # nothing to reason about; skip an LLM call
            action, target = options[0]
            return Decision(action, target, "Only one valid action in the current state.", "rules")
        actions = sorted({a for a, _ in options})
        targets = sorted({t for _, t in options if t}) + [""]
        schema = {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "action": {"type": "string", "enum": actions},
                "target": {"type": "string", "enum": targets},
            },
            "required": ["reason", "action", "target"],
        }
        menu = "\n".join(f"- {a}" + (f" target={t}" if t else ' target=""') for a, t in options)
        last = state.trace[-1].observation if state.trace else "none (first step)"
        user = f"STATE: {json.dumps(state.summary_for_planner())}\nLAST OBSERVATION: {last[:300]}\nALLOWED ACTIONS:\n{menu}"
        try:
            raw = self.llm.chat_json(self.system, user, schema, max_tokens=90)
        except LLMError as exc:
            return self.fallback.decide(state, options, "fallback_rules", f"LLM planner unavailable ({str(exc)[:80]}). ")
        action, target = raw.get("action"), raw.get("target") or None
        reason = str(raw.get("reason", "")).strip()[:200] or "(no reason given)"
        if (action, target) not in options:
            # The action is valid but the target is not (e.g. analyze_description with an image name, or
            # analyze_image with no name): keep the model's choice of action if its target is unambiguous.
            same = [o for o in options if o[0] == action]
            if len(same) == 1 or (same and target is None):
                fixed = same[0][1]
                return Decision(action, fixed, f"{reason} [guardrail: target {target!r} -> {fixed!r}]", "llm")
            return self.fallback.decide(state, options, "fallback_rules", f"LLM proposed invalid action {action}/{target}. ")
        return Decision(action, target, reason, "llm")


class CoordinatorAgent:
    name = "coordinator"

    def __init__(self, agents: list[Agent], llm: LLMClient | None, mode: str | None = None, max_steps: int | None = None) -> None:
        self.tools = {t.name: t for a in agents for t in a.tools()}
        mode = mode or settings.coordinator_mode
        self.policy = LLMPolicy(llm, list(self.tools.values())) if mode == "llm" and llm else RulePolicy()
        self.max_steps = max_steps or settings.max_react_steps

    def options(self, state: ClaimState) -> list[Option]:
        opts: list[Option] = [(t.name, target) for t in self.tools.values() for target in t.targets(state)]
        a = state.assessment
        if a and (a.evidence_version == state.evidence_version or state.assessments_run >= 2 or not opts):
            opts.append(("finish", None))
        return opts

    def run(self, state: ClaimState, on_step: Callable[[TraceStep], None] | None = None) -> TriageReport:
        termination = "max_steps_reached"
        for step in range(1, self.max_steps + 1):
            options = self.options(state)
            if not options:
                termination = "aborted"
                state.add_limitation("No valid action was available (required tools failed); report built from partial evidence.")
                break
            started = time.perf_counter()
            decision = self.policy.decide(state, options)
            if decision.action == "finish":
                state.finished = True
                self._record(state, step, decision, "coordinator", "ok", "Final report compiled and validated.", started, on_step)
                termination = "completed"
                break
            tool = self.tools[decision.action]
            try:
                observation, status = tool.run(state, decision.target), "ok"
            except ToolError as exc:
                observation, status = f"ERROR: {exc}", "error"
            except Exception as exc:  # noqa: BLE001 - a bug in one tool must not kill the whole run
                observation, status = f"ERROR (unexpected {exc.__class__.__name__}): {exc}", "error"
                state.add_limitation(f"Tool {tool.name} crashed unexpectedly; its output is missing.")
            self._record(state, step, decision, tool.agent, status, observation, started, on_step)
        if termination == "max_steps_reached":
            state.add_limitation(f"ReAct step limit ({self.max_steps}) reached before the coordinator finished.")
        return build_report(state, termination)

    @staticmethod
    def _record(
        state: ClaimState,
        step: int,
        d: Decision,
        agent: str,
        status: str,
        observation: str,
        started: float,
        on_step: Callable[[TraceStep], None] | None,
    ) -> None:
        trace = TraceStep(
            step=step,
            agent=agent,
            action=d.action,
            target=d.target,
            reason=d.reason,
            decided_by=d.decided_by,
            status=status,
            observation=observation[:600],
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        state.trace.append(trace)
        if on_step:
            on_step(trace)
