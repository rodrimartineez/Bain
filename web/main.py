"""FastAPI backend: intake endpoint, background triage jobs with live trace, knowledge-base and demo endpoints.

Run:  uvicorn web.main:app --port 8000   (or: python run.py)
"""

from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from claim_triage.config import settings
from claim_triage.ingestion import IntakeError, RawFile, build_submission
from claim_triage.llm import LLMError, OllamaClient
from claim_triage.schemas import ClaimMetadata, TraceStep
from claim_triage.service import TriageService

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Claim Triage Assistant", version="1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
# Local models run on one GPU: process one claim at a time, queue the rest.
_executor = ThreadPoolExecutor(max_workers=1)
_service: TriageService | None = None
_service_lock = threading.Lock()


def get_service() -> TriageService:
    global _service
    with _service_lock:
        if _service is None:
            _service = TriageService.from_settings()
        return _service


def set_service(service: TriageService | None) -> None:
    """Used by tests to inject a service with fake models."""
    global _service
    _service = service


# ----------------------------------------------------------------------------- pages


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    info: dict[str, Any] = {
        "text_model": settings.text_model,
        "vision_model": settings.vision_model,
        "embedding_model": settings.embedding_model,
        "coordinator_mode": settings.coordinator_mode,
    }
    try:
        models = OllamaClient().available_models()
        required = {settings.text_model, settings.vision_model, settings.embedding_model}
        missing = [m for m in required if m not in models and f"{m}:latest" not in models]
        info["ollama"] = "ok" if not missing else f"missing models: {', '.join(missing)}"
    except LLMError as exc:
        info["ollama"] = f"unreachable: {exc}"
    info["knowledge_base_records"] = len(_service.index.claims) if _service else None
    info["ok"] = info["ollama"] == "ok"
    return info


# ----------------------------------------------------------------------------- claims


@app.post("/api/claims", status_code=202)
async def submit_claim(
    description: str = Form(""),
    claim_id: str | None = Form(None),
    policy_number: str | None = Form(None),
    incident_date: str | None = Form(None),
    property_type: str | None = Form(None),
    location: str | None = Form(None),
    claimed_amount_eur: str | None = Form(None),
    images: list[UploadFile] = File(default=[]),
    documents: list[UploadFile] = File(default=[]),
) -> dict[str, str]:
    amount: float | None = None
    if claimed_amount_eur and claimed_amount_eur.strip():
        try:
            amount = float(claimed_amount_eur.replace(",", "."))
        except ValueError:
            raise HTTPException(422, detail=["Claimed amount must be a number."]) from None
        if amount < 0:
            raise HTTPException(422, detail=["Claimed amount cannot be negative."])
    metadata = ClaimMetadata(
        claim_id=(claim_id or "").strip() or None,
        policy_number=(policy_number or "").strip() or None,
        incident_date=(incident_date or "").strip() or None,
        property_type=(property_type or "").strip() or None,
        location=(location or "").strip() or None,
        claimed_amount_eur=amount,
    )
    raw_images = [RawFile(f.filename or "image", await f.read()) for f in images if f.filename]
    raw_docs = [RawFile(f.filename or "document", await f.read()) for f in documents if f.filename]
    try:
        submission = build_submission(description, metadata, raw_images, raw_docs)
    except IntakeError as exc:
        raise HTTPException(422, detail=exc.errors) from exc
    try:
        service = get_service()
    except LLMError as exc:
        raise HTTPException(503, detail=[f"AI backend unavailable: {exc}. Is Ollama running?"]) from exc

    job_id = uuid.uuid4().hex[:10]
    job = {
        "job_id": job_id,
        "status": "queued",
        "steps": [],
        "report": None,
        "error": None,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "inputs": {
            "description": submission.description,
            "metadata": metadata.model_dump(exclude_none=True),
            "images": [i.filename for i in submission.images],
            "documents": [d.filename for d in submission.documents],
        },
    }
    with _jobs_lock:
        _jobs[job_id] = job

    def on_step(step: TraceStep) -> None:
        with _jobs_lock:
            job["steps"].append(step.model_dump())

    def work() -> None:
        job["status"] = "running"
        try:
            report = service.triage(submission, on_step)
            job["report"] = report.model_dump(mode="json")
            job["status"] = "completed"
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI instead of hanging
            job["status"] = "failed"
            job["error"] = f"{exc.__class__.__name__}: {exc}"
        _save_run(job)

    _executor.submit(work)
    return {"job_id": job_id}


@app.get("/api/claims/{job_id}")
def get_claim(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            saved = settings.runs_dir / f"{job_id}.json"
            if saved.exists():
                return json.loads(saved.read_text(encoding="utf-8"))
            raise HTTPException(404, detail=["Unknown job id."])
        return dict(job, steps=list(job["steps"]))


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    runs = []
    for path in sorted(settings.runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]:
        data = json.loads(path.read_text(encoding="utf-8"))
        report = data.get("report") or {}
        runs.append(
            {
                "job_id": data["job_id"],
                "status": data["status"],
                "created_at": data.get("created_at"),
                "claim_id": report.get("claim_id"),
                "priority": (report.get("priority") or {}).get("level"),
            }
        )
    return runs


def _save_run(job: dict[str, Any]) -> None:
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    (settings.runs_dir / f"{job['job_id']}.json").write_text(json.dumps(job, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------------- knowledge base


@app.get("/api/knowledge-base")
def knowledge_base() -> dict[str, Any]:
    data = json.loads(settings.kb_path.read_text(encoding="utf-8"))
    claims = data["claims"]
    by_type: dict[str, int] = {}
    for c in claims:
        by_type[c["damage_type"]] = by_type.get(c["damage_type"], 0) + 1
    return {
        "notice": data["notice"],
        "count": len(claims),
        "by_damage_type": by_type,
        "embedding_model": settings.embedding_model,
        "min_similarity": settings.min_similarity,
    }


@app.get("/api/knowledge-base/search")
def kb_search(q: str, k: int = 5) -> list[dict[str, Any]]:
    if len(q.strip()) < 3:
        raise HTTPException(422, detail=["Query too short."])
    try:
        hits = get_service().index.search(q, top_k=min(k, 10))
    except LLMError as exc:
        raise HTTPException(503, detail=[str(exc)]) from exc
    return [
        {
            "claim_id": h.claim["claim_id"],
            "similarity": h.similarity,
            "damage_type": h.claim["damage_type"],
            "outcome": h.claim["outcome"],
            "description": h.claim["description"],
        }
        for h in hits
    ]


# ----------------------------------------------------------------------------- demo cases


@app.get("/api/demo-cases")
def demo_cases() -> list[dict[str, Any]]:
    cases = []
    for case_file in sorted(settings.demo_dir.glob("*/case.json")):
        case = json.loads(case_file.read_text(encoding="utf-8"))
        case["id"] = case_file.parent.name
        cases.append(case)
    return cases


@app.get("/api/demo-cases/{case_id}/files/{name}", include_in_schema=False)
def demo_file(case_id: str, name: str) -> FileResponse:
    base = settings.demo_dir.resolve()
    path = (base / case_id / name).resolve()
    if base not in path.parents or not path.is_file():
        raise HTTPException(404, detail=["File not found."])
    return FileResponse(path)


@app.exception_handler(HTTPException)
async def http_error(_request: Any, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, list) else [str(exc.detail)]
    return JSONResponse(status_code=exc.status_code, content={"errors": detail})
