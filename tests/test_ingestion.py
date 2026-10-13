import pytest

from claim_triage.agents.intake import find_amounts, parse_amount
from claim_triage.ingestion import IntakeError, RawFile
from tests.conftest import png_bytes


def test_valid_submission_is_stored(make_submission):
    sub = make_submission()
    assert sub.images[0].warning is None
    assert sub.documents[0].kind == "document"


@pytest.mark.parametrize("description", ["", "   ", "too short"])
def test_empty_or_short_description_rejected(make_submission, description):
    with pytest.raises(IntakeError) as exc:
        make_submission(description=description)
    assert "Description is required" in exc.value.errors[0]


def test_unsupported_and_empty_files_rejected_with_all_errors(make_submission):
    with pytest.raises(IntakeError) as exc:
        make_submission(images=[RawFile("photo.gif", b"GIF89a")], documents=[RawFile("notes.docx", b"PK"), RawFile("a.txt", b"")])
    errors = " | ".join(exc.value.errors)
    assert "unsupported file type '.gif'" in errors
    assert "unsupported file type '.docx'" in errors
    assert "file is empty" in errors


def test_too_many_images_rejected(make_submission):
    with pytest.raises(IntakeError):
        make_submission(images=[RawFile(f"{i}.png", png_bytes()) for i in range(6)])


def test_corrupt_image_accepted_with_warning(make_submission):
    sub = make_submission(images=[RawFile("broken.jpg", b"\xff\xd8\xff not an image")])
    assert sub.images[0].warning and "could not be decoded" in sub.images[0].warning


def test_tiny_image_flagged(make_submission):
    sub = make_submission(images=[RawFile("tiny.png", png_bytes((10, 10)))])
    assert "too small" in sub.images[0].warning


@pytest.mark.parametrize("raw,value", [("4,350.00", 4350.0), ("4.350,00", 4350.0), ("4350", 4350.0), ("1 150", 1150.0)])
def test_parse_amount_formats(raw, value):
    assert parse_amount(raw) == value


def test_find_amounts_only_currency_values():
    assert find_amounts("Ref 2026-1147, 9 m2. Item EUR 1,150.00 and 780 € ; TOTAL: 4.350,00 EUR") == [1150.0, 780.0, 4350.0]
