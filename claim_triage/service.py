"""Wires agents, knowledge base and coordinator together (composition root)."""

from __future__ import annotations

import uuid
from collections.abc import Callable

from .agents.assessor import TriageAssessorAgent
from .agents.auditor import EvidenceAuditorAgent
from .agents.coordinator import CoordinatorAgent
from .agents.intake import IntakeAnalystAgent
from .agents.research import KnowledgeResearcherAgent
from .knowledge_base import VectorIndex
from .llm import LLMClient, OllamaClient
from .schemas import ClaimSubmission, TraceStep, TriageReport
from .state import ClaimState


class TriageService:
    def __init__(self, llm: LLMClient, index: VectorIndex, mode: str | None = None, max_steps: int | None = None) -> None:
        self.llm = llm
        self.index = index
        self.mode = mode
        self.max_steps = max_steps

    @classmethod
    def from_settings(cls) -> TriageService:
        client = OllamaClient()
        return cls(client, VectorIndex.load_or_build(client))

    def new_coordinator(self) -> CoordinatorAgent:
        agents = [
            IntakeAnalystAgent(self.llm),
            KnowledgeResearcherAgent(self.index),
            EvidenceAuditorAgent(),
            TriageAssessorAgent(self.llm),
        ]
        return CoordinatorAgent(agents, self.llm, mode=self.mode, max_steps=self.max_steps)

    def triage(self, submission: ClaimSubmission, on_step: Callable[[TraceStep], None] | None = None) -> TriageReport:
        claim_id = submission.metadata.claim_id or f"CLM-{uuid.uuid4().hex[:6].upper()}"
        state = ClaimState(claim_id=claim_id, submission=submission)
        return self.new_coordinator().run(state, on_step)
