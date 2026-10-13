"""HTTP API: intake validation errors, async job lifecycle and report delivery (fake models injected)."""

import time

import pytest
from fastapi.testclient import TestClient

from claim_triage.config import settings
from tests.conftest import DESCRIPTION, pdf_bytes, png_bytes
from web import main


@pytest.fixture
def client(make_service, tmp_path):
    # Settings is frozen; redirect output folders to tmp for the duration of the test.
    original = {k: getattr(settings, k) for k in ("runs_dir", "upload_dir")}
    object.__setattr__(settings, "runs_dir", tmp_path / "runs")
    object.__setattr__(settings, "upload_dir", tmp_path / "uploads")
    main.set_service(make_service())
    yield TestClient(main.app)
    main.set_service(None)
    for k, v in original.items():
        object.__setattr__(settings, k, v)


def wait(client, job_id, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/claims/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_submit_claim_end_to_end(client):
    resp = client.post(
        "/api/claims",
        data={"description": DESCRIPTION, "claim_id": "CLM-T1", "claimed_amount_eur": "4350", "incident_date": "2026-10-08"},
        files=[
            ("images", ("ceiling.png", png_bytes(), "image/png")),
            ("documents", ("estimate.pdf", pdf_bytes("Estimate TOTAL EUR 4,350.00"), "application/pdf")),
        ],
    )
    assert resp.status_code == 202
    job = wait(client, resp.json()["job_id"])
    assert job["status"] == "completed", job["error"]
    report = job["report"]
    assert report["claim_id"] == "CLM-T1"
    assert report["similar_claims"]
    assert [s["action"] for s in job["steps"]][-1] == "finish"


def test_empty_description_returns_422(client):
    resp = client.post("/api/claims", data={"description": ""})
    assert resp.status_code == 422
    assert "Description is required" in resp.json()["errors"][0]


def test_unsupported_file_returns_422(client):
    resp = client.post(
        "/api/claims",
        data={"description": DESCRIPTION},
        files=[("documents", ("notes.docx", b"PK..", "application/octet-stream"))],
    )
    assert resp.status_code == 422
    assert "unsupported file type" in resp.json()["errors"][0]


def test_invalid_amount_returns_422(client):
    resp = client.post("/api/claims", data={"description": DESCRIPTION, "claimed_amount_eur": "lots"})
    assert resp.status_code == 422


def test_unknown_job_404(client):
    assert client.get("/api/claims/doesnotexist").status_code == 404


def test_demo_cases_listed_and_path_traversal_blocked(client):
    cases = client.get("/api/demo-cases").json()
    assert {c["id"] for c in cases} >= {"01_ceiling_leak", "04_minor_text_only"}
    assert all(c["synthetic"] for c in cases)
    assert client.get("/api/demo-cases/01_ceiling_leak/files/..%2F..%2F..%2Fpyproject.toml").status_code == 404


def test_index_page_served(client):
    assert "Claim Triage Assistant" in client.get("/").text
