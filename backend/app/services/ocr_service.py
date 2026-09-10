"""
OCR / text-extraction service.

Strategy (hybrid, per spec section 9's suggested free/open-source options):
  1. Native PDFs: extract embedded text directly with PyMuPDF (fast, exact).
  2. Scanned PDFs (little/no embedded text): rasterize each page and run
     Tesseract OCR on the resulting image.
  3. JPG/PNG images: run Tesseract OCR directly.

The output is one `PageText` per page, each carrying the page number so
downstream extraction can preserve page-level evidence/grounding (spec 4.3).
"""
from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from app.core.config import get_settings
from app.core.exceptions import OCRProcessingError

logger = logging.getLogger("document_intelligence")
settings = get_settings()

if settings.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD

# Render scanned PDF pages at a higher DPI than the PDF default (72) for
# materially better OCR accuracy.
_RENDER_ZOOM = 3.0


@dataclass
class PageText:
    page_number: int  # 1-indexed
    text: str
    source: str  # "native_pdf_text" | "ocr"


@dataclass
class OCRResult:
    pages: list[PageText]
    engine_used: str  # "pymupdf_text" | "tesseract_ocr" | "mixed"

    @property
    def full_text(self) -> str:
        return "\n\n".join(f"[PAGE {p.page_number}]\n{p.text}" for p in self.pages)


def _ocr_image(image: Image.Image) -> str:
    """OCR an image with a fast primary pass and quality-based fallbacks."""
    try:
        prepared = image.convert("L")

        def quality(text: str) -> tuple[int, int, int]:
            upper = text.upper()
            financial_terms = sum(
                upper.count(term)
                for term in (
                    "TOTAL", "ASSETS", "LIABILITIES", "CAPITAL", "REVENUE",
                    "INCOME", "EXPENDITURE", "PROFIT", "CASH FLOW",
                    "OPERATING", "INVESTING", "FINANCING", "GST", "TAX",
                )
            )
            paired_rows = sum(
                1 for line in text.splitlines()
                if re.search(r"[A-Za-z]{3}", line) and re.search(r"\d[\d,.]*", line)
            )
            numeric_tokens = len(re.findall(r"\d[\d,.]*", text))
            return paired_rows, financial_terms, numeric_tokens

        primary = pytesseract.image_to_string(prepared, config="--psm 6")
        primary_q = quality(primary)

        # Annual financial statements benefit materially from PSM 4: it often
        # preserves the numeric columns that PSM 6 drops on scanned reports.
        # Receipts/images keep the faster PSM 6 path unless the result is weak.
        statement_like = bool(re.search(
            r"balance sheet|profit and loss|cash flow|cash flows from|capital and liabilities|consolidated",
            primary,
            re.I,
        ))
        if statement_like:
            psm4 = pytesseract.image_to_string(prepared, config="--psm 4")
            return max((primary, psm4), key=quality)

        if primary_q[0] >= 6 or len(primary.strip()) >= 100:
            return primary

        candidates = [primary]
        for psm in (4, 11):
            candidates.append(pytesseract.image_to_string(prepared, config=f"--psm {psm}"))
        return max(candidates, key=quality)
    except Exception as exc:
        raise OCRProcessingError(
            "OCR engine failed to process the document.", details={"reason": str(exc)}
        ) from exc


def extract_text_from_pdf(content: bytes) -> OCRResult:
    pages: list[PageText] = []
    engines_used: set[str] = set()

    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise OCRProcessingError(
            "Unable to open PDF for text extraction.", details={"reason": str(exc)}
        ) from exc

    try:
        for i, page in enumerate(doc, start=1):
            native_text = page.get_text("text") or ""
            if len(native_text.strip()) >= settings.OCR_TEXT_DENSITY_THRESHOLD:
                pages.append(PageText(page_number=i, text=native_text, source="native_pdf_text"))
                engines_used.add("pymupdf_text")
            else:
                # Likely a scanned page -> rasterize and OCR it.
                pix = page.get_pixmap(matrix=fitz.Matrix(_RENDER_ZOOM, _RENDER_ZOOM))
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = _ocr_image(img)
                pages.append(PageText(page_number=i, text=ocr_text, source="ocr"))
                engines_used.add("tesseract_ocr")
    finally:
        doc.close()

    engine_used = "mixed" if len(engines_used) > 1 else next(iter(engines_used), "pymupdf_text")
    return OCRResult(pages=pages, engine_used=engine_used)


def extract_text_from_image(content: bytes) -> OCRResult:
    try:
        img = Image.open(io.BytesIO(content))
        img.load()
    except Exception as exc:
        raise OCRProcessingError(
            "Unable to open image for OCR.", details={"reason": str(exc)}
        ) from exc

    text = _ocr_image(img)
    return OCRResult(pages=[PageText(page_number=1, text=text, source="ocr")], engine_used="tesseract_ocr")


def extract_text(filename_ext: str, content: bytes) -> OCRResult:
    """filename_ext: one of 'pdf', 'jpg', 'png' (as reported by file_validation)."""
    if filename_ext == "pdf":
        return extract_text_from_pdf(content)
    return extract_text_from_image(content)
