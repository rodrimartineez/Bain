"""Run one demo case end-to-end from the command line (no web UI), printing the live ReAct trace.

python scripts/run_case.py data/demo/01_ceiling_leak [--mode llm|rules]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claim_triage.config import settings  # noqa: E402
from claim_triage.ingestion import RawFile, build_submission  # noqa: E402
from claim_triage.knowledge_base import VectorIndex  # noqa: E402
from claim_triage.llm import OllamaClient  # noqa: E402
from claim_triage.schemas import ClaimMetadata, TraceStep, TriageReport  # noqa: E402
from claim_triage.service import TriageService  # noqa: E402


def load_case(case_dir: Path) -> tuple[dict, object]:
    case = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    submission = build_submission(
        case["description"],
        ClaimMetadata(**case.get("metadata", {})),
        [RawFile(n, (case_dir / n).read_bytes()) for n in case.get("images", [])],
        [RawFile(n, (case_dir / n).read_bytes()) for n in case.get("documents", [])],
    )
    return case, submission


def run(case_dir: Path, service: TriageService, verbose: bool = True) -> TriageReport:
    _case, submission = load_case(case_dir)

    def show(step: TraceStep) -> None:
        if verbose:
            print(
                f"[{step.step:>2}] {step.agent:<20} {step.action:<22} {step.target or '':<22} "
                f"({step.decided_by}, {step.duration_ms / 1000:.1f}s) {step.status}\n      reason: {step.reason}\n"
                f"      obs:    {step.observation[:220]}",
                flush=True,
            )

    return service.triage(submission, show)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--mode", choices=["llm", "rules"], default=settings.coordinator_mode)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    client = OllamaClient()
    service = TriageService(client, VectorIndex.load_or_build(client), mode=args.mode)
    report = run(args.case_dir, service)
    out = args.out or settings.runs_dir.parent / f"report_{args.case_dir.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(
        f"\nPRIORITY {report.priority.level} ({report.priority.label}, score {report.priority.score}) | "
        f"damage {report.primary_damage_type} | confidence {report.confidence.level} {report.confidence.score} | "
        f"human review {report.needs_human_review} | termination {report.termination}\nSaved {out}"
    )


if __name__ == "__main__":
    main()
