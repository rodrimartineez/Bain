from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from fpdf import FPDF
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claim_triage.ingestion import RawFile, build_submission  # noqa: E402
from claim_triage.knowledge_base import VectorIndex, load_claims  # noqa: E402
from claim_triage.schemas import ClaimMetadata  # noqa: E402
from claim_triage.service import TriageService  # noqa: E402
from tests.fakes import FakeEmbedder, FakeLLM  # noqa: E402

DESCRIPTION = (
    "On 2026-10-08 a pipe burst in the bathroom of the flat above. Water came through our bedroom ceiling and "
    "dripped from the ceiling light. Plaster is bubbling. Estimate attached."
)


def png_bytes(size: tuple[int, int] = (200, 150), color=(150, 100, 60)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def pdf_bytes(text: str | None) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    if text:
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 6, text)
    else:
        pdf.rect(20, 20, 100, 10, style="F")
    return bytes(pdf.output())


@pytest.fixture
def index() -> VectorIndex:
    return VectorIndex.build(load_claims(), FakeEmbedder())


@pytest.fixture
def make_submission(tmp_path):
    def _make(description: str = DESCRIPTION, images=None, documents=None, **meta):
        images = images if images is not None else [RawFile("ceiling.png", png_bytes())]
        documents = (
            documents
            if documents is not None
            else [RawFile("estimate.pdf", pdf_bytes("Repair estimate 2026-10-09. Plaster and paint. TOTAL EUR 4,350.00"))]
        )
        metadata = ClaimMetadata(**({"incident_date": "2026-10-08", "claimed_amount_eur": 4350} | meta))
        return build_submission(description, metadata, images, documents, upload_root=tmp_path)

    return _make


@pytest.fixture
def make_service(index):
    def _make(llm: FakeLLM | None = None, mode: str = "llm", max_steps: int = 12) -> TriageService:
        return TriageService(llm or FakeLLM(), index, mode=mode, max_steps=max_steps)

    return _make
