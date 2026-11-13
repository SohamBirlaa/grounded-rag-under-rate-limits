"""Measure PDF text extraction quality for different pdfplumber x_tolerance values (free, local).

  uv run python scripts/check_extraction.py

glued  = words of >=18 letters with no space (e.g. "newstate-of-the-artBLEUscoreof28.4" fragments)
split  = share of 1-letter alphabetic tokens other than a/A/I (over-splitting, e.g. "T r a n s")
Pick the smallest tolerance with few glued words and no rise in split.
"""
import re
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
PDFS = sorted((ROOT / "data/corpus/pdfs").glob("*.pdf"))
PROBES = ["BLEU", "28.4", "dmodel", "d model", "12.5 Hz", "state-of-the-art"]

for pdf_path in PDFS:
    print(f"\n{pdf_path.name}")
    with pdfplumber.open(str(pdf_path)) as pdf:
        pages = pdf.pages[:12]
        for tol in (3.0, 2.0, 1.5, 1.0):
            text = "\n".join((p.extract_text(x_tolerance=tol) or "") for p in pages)
            words = text.split()
            glued = sum(1 for w in words if len(w) >= 18 and re.fullmatch(r"[A-Za-z\-,\.]+", w))
            singles = [w for w in words if re.fullmatch(r"[A-Za-z]", w) and w not in ("a", "A", "I")]
            hits = {p: text.count(p) for p in PROBES}
            print(f"  x_tolerance={tol:<4} words={len(words):6d} glued={glued:4d} "
                  f"split={100 * len(singles) / max(1, len(words)):4.1f}%  probes={hits}")

print("\nShare of characters inside detected tables (what RELAY_PDF_DROP_TABLE_TEXT=1 removes from prose):")
for pdf_path in PDFS:
    total = inside = 0
    heavy = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            boxes = [t.bbox for t in page.find_tables()]
            chars = page.chars
            n_in = sum(1 for c in chars if any(c["x0"] >= b[0] - 1 and c["x1"] <= b[2] + 1 and
                                               c["top"] >= b[1] - 1 and c["bottom"] <= b[3] + 1 for b in boxes))
            total += len(chars)
            inside += n_in
            if chars and n_in / len(chars) > 0.3:
                heavy.append(f"p{i}:{100 * n_in / len(chars):.0f}%")
    print(f"  {pdf_path.name}: {100 * inside / max(1, total):.1f}% of chars; pages >30%: {', '.join(heavy) or 'none'}")
