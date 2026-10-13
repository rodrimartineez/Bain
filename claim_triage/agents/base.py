"""Common building blocks: a Tool is an action the coordinator can choose; an Agent owns tools."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..state import ClaimState


class ToolError(RuntimeError):
    """A tool failed in a way that should be recorded in the trace (the run continues)."""


@dataclass(frozen=True)
class Tool:
    name: str
    agent: str
    description: str
    # Returns the valid targets for this tool in the current state. [] -> not available,
    # [None] -> available without a target, ["a.jpg", ...] -> one action per target.
    targets: Callable[[ClaimState], list[str | None]]
    # Executes the tool, mutates the state and returns a one-line observation for the trace.
    run: Callable[[ClaimState, str | None], str]


class Agent:
    """A specialist with a single responsibility. Subclasses expose their tools."""

    name: str = "agent"

    def tools(self) -> list[Tool]:
        raise NotImplementedError
