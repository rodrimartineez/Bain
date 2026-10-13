"""Claim intake: validates the submission before any agent runs.

Hard errors (raise ``IntakeError`` -> HTTP 422) are problems the user must fix: empty description,
unsupported file type, empty file, too many/too large files. Soft problems that only become visible
when content is parsed (corrupt image, PDF without text) are accepted with a warning and surface
as limitations in the report, so one bad attachment does not block triage of the rest.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .config import settings
from .schemas import ClaimMetadata, ClaimSubmission, UploadedFile

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".md"}


class IntakeError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class RawFile:
    filename: str
    content: bytes


def _safe_name(name: str) -> str:
    name = Path(name or "file").name
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)[:100] or "file"


def _check_image(path: Path) -> str | None:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:  # verify() invalidates the object; reopen to decode pixels
            img.load()
            if min(img.size) < 32:
                return f"image is too small to analyse ({img.size[0]}x{img.size[1]} px)"
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        return f"image could not be decoded ({exc.__class__.__name__})"
    return None


def build_submission(
    description: str,
    metadata: ClaimMetadata,
    images: list[RawFile],
    documents: list[RawFile],
    upload_root: Path | None = None,
) -> ClaimSubmission:
    errors: list[str] = []
    description = (description or "").strip()
    if len(description) < settings.min_description_chars:
        errors.append(
            f"Description is required and must be at least {settings.min_description_chars} characters "
            "(what happened, where, and when)."
        )
    if len(images) > settings.max_images:
        errors.append(f"At most {settings.max_images} images are allowed.")
    if len(documents) > settings.max_documents:
        errors.append(f"At most {settings.max_documents} documents are allowed.")

    max_bytes = int(settings.max_file_mb * 1024 * 1024)
    for group, allowed in ((images, IMAGE_EXTENSIONS), (documents, DOCUMENT_EXTENSIONS)):
        for f in group:
            ext = Path(f.filename).suffix.lower()
            if ext not in allowed:
                errors.append(f"'{f.filename}': unsupported file type '{ext or 'none'}'. Allowed: {', '.join(sorted(allowed))}.")
            elif len(f.content) == 0:
                errors.append(f"'{f.filename}': file is empty.")
            elif len(f.content) > max_bytes:
                errors.append(f"'{f.filename}': file exceeds {settings.max_file_mb:.0f} MB.")
    if errors:
        raise IntakeError(errors)

    submission_dir = (upload_root or settings.upload_dir) / uuid.uuid4().hex[:12]
    submission_dir.mkdir(parents=True, exist_ok=True)

    def store(raw: RawFile, kind: str, index: int) -> UploadedFile:
        path = submission_dir / f"{kind[0]}{index}_{_safe_name(raw.filename)}"
        path.write_bytes(raw.content)
        warning = _check_image(path) if kind == "image" else None
        return UploadedFile(
            filename=_safe_name(raw.filename), path=str(path), kind=kind, size_bytes=len(raw.content), warning=warning
        )

    return ClaimSubmission(
        description=description,
        metadata=metadata,
        images=[store(f, "image", i) for i, f in enumerate(images)],
        documents=[store(f, "document", i) for i, f in enumerate(documents)],
    )
