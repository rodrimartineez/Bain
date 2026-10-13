"""Small evaluation harness (own data, not Bain's ground truth, which was unavailable).

1. Retrieval: hit@3 / MRR over paraphrased queries, damage-type precision@3, and whether off-topic
   queries are correctly rejected by the similarity threshold. Needs only the embedding model.
2. End-to-end (--e2e): runs every demo case through the full multi-agent pipeline and compares with the
   `expected` block in its case.json. Slow on CPU-bound hardware (minutes per case).

    python scripts/run_eval.py            # retrieval only
    python scripts/run_eval.py --e2e      # retrieval + end-to-end cases
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from claim_triage.config import settings  # noqa: E402
from claim_triage.knowledge_base import VectorIndex  # noqa: E402
from claim_triage.llm import OllamaClient  # noqa: E402
from claim_triage.service import TriageService  # noqa: E402
from scripts.run_case import run as run_case  # noqa: E402


def eval_retrieval(index: VectorIndex) -> dict:
    data = json.loads((ROOT / "data" / "eval" / "retrieval_queries.json").read_text(encoding="utf-8"))
    rows, hits3, rr, type_prec = [], 0, 0.0, 0.0
    for q in data["queries"]:
        results = index.search(q["query"], top_k=5)
        ids = [h.claim["claim_id"] for h in results]
        rank = next((i + 1 for i, cid in enumerate(ids) if cid in q["relevant"]), None)
        hits3 += 1 if rank and rank <= 3 else 0
        rr += 1 / rank if rank else 0
        top3_types = [h.claim["damage_type"] for h in results[:3]]
        type_prec += sum(t == q["damage_type"] for t in top3_types) / 3
        rows.append({"query": q["query"], "first_relevant_rank": rank, "top3": ids[:3], "top1_similarity": results[0].similarity})
    rejected = [q for q in data["irrelevant_queries"] if not index.search(q, top_k=1, min_similarity=settings.min_similarity)]
    n = len(data["queries"])
    return {
        "queries": n,
        "hit_at_3": round(hits3 / n, 3),
        "mrr_at_5": round(rr / n, 3),
        "damage_type_precision_at_3": round(type_prec / n, 3),
        "irrelevant_rejected": f"{len(rejected)}/{len(data['irrelevant_queries'])} (threshold {settings.min_similarity})",
        "rows": rows,
    }


def eval_e2e(service: TriageService) -> list[dict]:
    results = []
    for case_file in sorted(settings.demo_dir.glob("*/case.json")):
        case = json.loads(case_file.read_text(encoding="utf-8"))
        exp = case.get("expected", {})
        print(f"\n=== {case_file.parent.name} ===", flush=True)
        started = time.time()
        report = run_case(case_file.parent, service)
        retrieved = [h.claim_id for h in report.similar_claims]
        checks = {
            "completed": report.termination == "completed",
            "damage_type": str(report.primary_damage_type.value if report.primary_damage_type else None)
            == exp.get("primary_damage_type"),
        }
        if "priority" in exp:
            checks["priority_in_expected"] = report.priority.level in exp["priority"]
        if "needs_human_review" in exp:
            checks["human_review"] = report.needs_human_review == exp["needs_human_review"]
        if exp.get("similar_claims_should_include"):
            checks["expected_precedent_retrieved"] = any(c in retrieved for c in exp["similar_claims_should_include"])
        results.append(
            {
                "case": case_file.parent.name,
                "seconds": round(time.time() - started),
                "priority": report.priority.level,
                "primary_damage_type": report.primary_damage_type,
                "retrieved": retrieved,
                "narrative_source": report.narrative_source,
                "llm_decisions": sum(s.decided_by == "llm" for s in report.trace),
                "fallback_decisions": sum(s.decided_by == "fallback_rules" for s in report.trace),
                "steps": len(report.trace),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
        print(json.dumps(results[-1], default=str), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e", action="store_true")
    parser.add_argument("--mode", choices=["llm", "rules"], default=settings.coordinator_mode)
    args = parser.parse_args()
    client = OllamaClient()
    index = VectorIndex.load_or_build(client)
    out = {
        "embedding_model": settings.embedding_model,
        "text_model": settings.text_model,
        "vision_model": settings.vision_model,
        "retrieval": eval_retrieval(index),
    }
    r = out["retrieval"]
    print(
        f"Retrieval: hit@3={r['hit_at_3']} MRR@5={r['mrr_at_5']} type-precision@3={r['damage_type_precision_at_3']} "
        f"irrelevant rejected {r['irrelevant_rejected']}"
    )
    if args.e2e:
        out["coordinator_mode"] = args.mode
        out["end_to_end"] = eval_e2e(TriageService(client, index, mode=args.mode))
        passed = sum(c["passed"] for c in out["end_to_end"])
        print(f"\nEnd-to-end: {passed}/{len(out['end_to_end'])} cases passed all checks")
    path = ROOT / "output" / ("eval_results_e2e.json" if args.e2e else "eval_results_retrieval.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
