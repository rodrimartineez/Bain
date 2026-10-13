"""Generates the SYNTHETIC demo inputs in data/demo/ (images drawn with Pillow, PDFs with fpdf2).

The original Bain test assets were not available, so these are procedurally generated illustrations
and fictional documents. They are NOT photographs or documents from real claims.

    python scripts/generate_demo_assets.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "data" / "demo"
W, H = 800, 600


# ----------------------------------------------------------------------------- images


def _noise(img: Image.Image, amount: int, seed: int) -> Image.Image:
    rnd = random.Random(seed)
    px = img.load()
    for _ in range(W * H // 6):
        x, y = rnd.randrange(W), rnd.randrange(H)
        r, g, b = px[x, y]
        d = rnd.randint(-amount, amount)
        px[x, y] = (max(0, min(255, r + d)), max(0, min(255, g + d)), max(0, min(255, b + d)))
    return img


def ceiling_water_stain(path: Path) -> None:
    rnd = random.Random(7)
    img = Image.new("RGB", (W, H), (232, 229, 221))
    d = ImageDraw.Draw(img)
    for x in range(0, W, 100):
        d.line([(x, 0), (x, H)], fill=(206, 201, 190), width=3)
    for y in range(0, H, 100):
        d.line([(0, y), (W, y)], fill=(206, 201, 190), width=3)
    stain = Image.new("L", (W, H), 0)
    sd = ImageDraw.Draw(stain)
    for _ in range(80):
        cx, cy, r = 400 + rnd.gauss(0, 85), 260 + rnd.gauss(0, 55), rnd.uniform(25, 85)
        sd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rnd.randint(110, 255))
    stain = stain.filter(ImageFilter.GaussianBlur(22))
    img = Image.composite(Image.new("RGB", (W, H), (146, 100, 52)), img, stain.point(lambda v: int(v * 0.85)))
    ring = Image.new("L", (W, H), 0)
    ImageDraw.Draw(ring).ellipse([215, 95, 585, 425], outline=200, width=10)
    img = Image.composite(Image.new("RGB", (W, H), (110, 72, 35)), img, ring.filter(ImageFilter.GaussianBlur(6)))
    d = ImageDraw.Draw(img)
    # ceiling light fitting inside the stain
    d.ellipse([370, 230, 430, 290], fill=(245, 245, 240), outline=(120, 120, 120), width=3)
    d.ellipse([388, 248, 412, 272], fill=(255, 250, 215))
    # bubbling / peeling plaster and drips
    for _ in range(14):
        x, y = 400 + rnd.gauss(0, 90), 260 + rnd.gauss(0, 60)
        d.ellipse([x, y, x + 14, y + 10], fill=(200, 180, 150), outline=(120, 85, 50))
    for i in range(6):
        x = 330 + i * 28
        d.ellipse([x, 430 + (i % 3) * 18, x + 7, 442 + (i % 3) * 18], fill=(125, 90, 55))
    _noise(img, 10, 1).save(path, quality=90)


def roof_missing_tiles(path: Path) -> None:
    rnd = random.Random(11)
    img = Image.new("RGB", (W, H), (150, 170, 190))
    d = ImageDraw.Draw(img)
    d.polygon([(0, 140), (W, 60), (W, H), (0, H)], fill=(60, 45, 40))  # exposed underlayment / roof deck
    tile_w, tile_h = 70, 40
    missing = {(r, c) for r in range(3, 8) for c in range(4, 8) if rnd.random() < 0.75}
    for row in range(14):
        for col in range(13):
            x = col * tile_w - (tile_w // 2 if row % 2 else 0)
            y = 150 + row * tile_h - col * 7
            if (row, col) in missing:
                if rnd.random() < 0.3:  # broken tile fragment
                    d.polygon([(x, y), (x + 30, y + 5), (x + 10, y + 30)], fill=(150, 70, 50))
                continue
            shade = rnd.randint(-15, 15)
            d.rectangle([x, y, x + tile_w - 4, y + tile_h - 2], fill=(165 + shade, 75 + shade, 55 + shade), outline=(110, 50, 35))
    # wooden battens visible where tiles are missing
    for row in range(3, 8):
        y = 150 + row * tile_h - 40
        d.line([(250, y + 10), (560, y - 20)], fill=(150, 115, 70), width=6)
    # fallen tiles in the gutter
    for _ in range(6):
        x = rnd.randint(100, 700)
        d.polygon([(x, 560), (x + 40, 548), (x + 55, 575), (x + 10, 590)], fill=(150, 72, 52), outline=(90, 40, 30))
    _noise(img, 12, 2).save(path, quality=90)


def intact_floor(path: Path) -> None:
    img = Image.new("RGB", (W, H), (214, 208, 196))
    d = ImageDraw.Draw(img)
    for x in range(0, W, 120):
        d.line([(x, 0), (x, H)], fill=(170, 165, 155), width=4)
    for y in range(0, H, 120):
        d.line([(0, y), (W, y)], fill=(170, 165, 155), width=4)
    d.rectangle([560, 0, 800, 90], fill=(240, 240, 240), outline=(180, 180, 180), width=3)  # cabinet base
    _noise(img, 8, 3).save(path, quality=90)


# ----------------------------------------------------------------------------- documents


def _pdf(path: Path, title: str, lines: list[str]) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(
        0,
        6,
        "SYNTHETIC DEMO DOCUMENT - fictional company and figures, generated for a take-home assignment.",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 11)
    for line in lines:
        pdf.multi_cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))


def scanned_pdf_without_text(path: Path) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(200, 200, 200)
    for i in range(12):
        pdf.rect(20, 30 + i * 15, 170 - (i % 4) * 20, 6, style="F")  # looks like text lines, but it is drawing only
    pdf.output(str(path))


# ----------------------------------------------------------------------------- cases


def write_case(case_dir: Path, case: dict) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case.json").write_text(json.dumps(case, indent=2), encoding="utf-8")
    (case_dir / "description.txt").write_text(case["description"], encoding="utf-8")


def main() -> None:
    # 1. Main demo: multimodal water leak with electrical hazard.
    c1 = DEMO / "01_ceiling_leak"
    write_case(
        c1,
        {
            "title": "Ceiling leak around a light fitting (main demo, multimodal)",
            "synthetic": True,
            "description": (
                "On 2026-10-08 a pipe in the bathroom of the flat above burst while the neighbours were away. Water came "
                "through our bedroom ceiling for several hours before the building manager shut off the supply. There is a "
                "large brown stain, the plaster is bubbling and water dripped out of the ceiling light, so we switched off "
                "that circuit. The room is unusable but the rest of the flat is fine. A plumber's estimate for the repairs "
                "is attached."
            ),
            "metadata": {
                "claim_id": "CLM-DEMO-001",
                "policy_number": "HOME-55821",
                "incident_date": "2026-10-08",
                "property_type": "apartment",
                "location": "Madrid",
                "claimed_amount_eur": 4350,
            },
            "images": ["ceiling_stain.jpg"],
            "documents": ["repair_estimate.pdf"],
            "expected": {
                "primary_damage_type": "water_leak",
                "priority": ["P1", "P2"],
                "similar_claims_should_include": ["HC-2024-0040", "HC-2023-0003"],
            },
        },
    )
    ceiling_water_stain(c1 / "ceiling_stain.jpg")
    _pdf(
        c1 / "repair_estimate.pdf",
        "Fontaneria Rapida S.L. - Repair estimate",
        [
            "Estimate no. E-2026-1147        Date: 2026-10-09",
            "Client: policy HOME-55821, bedroom ceiling, Madrid",
            "Findings: water ingress from the upper floor bathroom supply pipe. Plaster saturated over approx. 9 m2. "
            "Ceiling light fitting wet; electrical check recommended before re-energising the circuit.",
            "1. Removal of damaged plaster and drying (dehumidifier, 5 days) ........ EUR 1,150.00",
            "2. New plasterboard ceiling, 9 m2 ........................................ EUR 1,620.00",
            "3. Painting of ceiling and walls ......................................... EUR 780.00",
            "4. Electrician inspection and new ceiling light fitting ................ EUR 800.00",
            "TOTAL (VAT included) .................................................... EUR 4,350.00",
        ],
    )

    # 2. Storm damage to a roof with ongoing rain ingress.
    c2 = DEMO / "02_storm_roof"
    write_case(
        c2,
        {
            "title": "Storm tore tiles off a roof (ongoing ingress)",
            "synthetic": True,
            "description": (
                "Last night's storm with very strong winds ripped a large section of tiles off our roof. Rain is still "
                "getting into the attic and dripping onto the landing ceiling. We have put buckets down but need an urgent "
                "temporary cover. Two elderly parents live with us. Weather report and roofer quote attached."
            ),
            "metadata": {
                "claim_id": "CLM-DEMO-002",
                "policy_number": "HOME-30917",
                "incident_date": "2026-10-12",
                "property_type": "house",
                "location": "A Coruna",
                "claimed_amount_eur": 9800,
            },
            "images": ["roof_damage.jpg"],
            "documents": ["roofer_quote.txt"],
            "expected": {
                "primary_damage_type": "storm_wind",
                "priority": ["P1"],
                "similar_claims_should_include": ["HC-2023-0010", "HC-2024-0034"],
            },
        },
    )
    roof_missing_tiles(c2 / "roof_damage.jpg")
    (c2 / "roofer_quote.txt").write_text(
        "SYNTHETIC DEMO DOCUMENT\nTejados Atlantico - Quote Q-8812 - Date: 2026-10-12\n"
        "Emergency tarpaulin and securing of loose tiles: EUR 650.00\n"
        "Replacement of approx. 25 m2 of roof tiles and battens: EUR 7,900.00\n"
        "Attic insulation drying and replacement: EUR 1,250.00\nTotal: EUR 9,800.00\n"
        "Note: regional weather service reported gusts of 118 km/h on 2026-10-12.\n",
        encoding="utf-8",
    )

    # 3. Inconsistent claim: declared damage not supported by the photo and an invoice predating the incident.
    c3 = DEMO / "03_inconsistent_claim"
    write_case(
        c3,
        {
            "title": "Inconsistent evidence (photo shows no damage, invoice predates incident)",
            "synthetic": True,
            "description": (
                "On 2026-09-20 a pipe under the bathroom floor leaked and destroyed all the floor tiles and the wooden "
                "skirting in the bathroom and hallway. Everything has to be replaced. Invoice attached."
            ),
            "metadata": {
                "claim_id": "CLM-DEMO-003",
                "policy_number": "HOME-77410",
                "incident_date": "2026-09-20",
                "property_type": "house",
                "location": "Tarragona",
                "claimed_amount_eur": 6000,
            },
            "images": ["bathroom_floor.jpg"],
            "documents": ["invoice.pdf"],
            "expected": {
                "primary_damage_type": "water_leak",
                "needs_human_review": True,
                "similar_claims_should_include": ["HC-2024-0032"],
            },
        },
    )
    intact_floor(c3 / "bathroom_floor.jpg")
    _pdf(
        c3 / "invoice.pdf",
        "Reformas Costa S.L. - Invoice",
        [
            "Invoice no. F-2026-0311        Date: 2026-03-14",
            "Bathroom floor tiling and skirting replacement.",
            "TOTAL: EUR 2,100.00",
        ],
    )

    # 4. Minor text-only claim.
    c4 = DEMO / "04_minor_text_only"
    write_case(
        c4,
        {
            "title": "Minor dishwasher leak, text only",
            "synthetic": True,
            "description": (
                "Yesterday morning, 2026-10-12, the dishwasher leaked a small amount of water and two kitchen floor tiles "
                "near it have lifted. Nothing else was damaged and we are living normally in the flat. Estimated repair "
                "around 600 euros."
            ),
            "metadata": {"claim_id": "CLM-DEMO-004", "property_type": "apartment", "location": "Zaragoza"},
            "images": [],
            "documents": [],
            "expected": {
                "primary_damage_type": "water_leak",
                "priority": ["P3", "P4"],
                "similar_claims_should_include": ["HC-2023-0006"],
            },
        },
    )

    # Negative inputs for demonstrating validation (not a case: no case.json).
    neg = DEMO / "negative_inputs"
    neg.mkdir(parents=True, exist_ok=True)
    (neg / "corrupt_photo.jpg").write_bytes(b"\xff\xd8\xff\xe0 this is not really a jpeg" + bytes(range(256)) * 4)
    scanned_pdf_without_text(neg / "scanned_no_text.pdf")
    (neg / "unsupported.docx").write_bytes(b"PK\x03\x04 fake docx")
    (neg / "empty.txt").write_bytes(b"")
    (neg / "README.txt").write_text(
        "Files for demonstrating input validation:\n"
        "- unsupported.docx / empty.txt -> rejected at intake (HTTP 422, shown in the UI)\n"
        "- corrupt_photo.jpg -> accepted with a warning; image tool fails; limitation shown in the report\n"
        "- scanned_no_text.pdf -> accepted; document tool reports no extractable text (no OCR)\n",
        encoding="utf-8",
    )
    print(f"Demo assets written to {DEMO}", file=sys.stderr)


if __name__ == "__main__":
    main()
