import io

import fitz
from PIL import Image

from app.services.file_validation import validate_file


def _make_pdf_bytes(num_pages: int = 1) -> bytes:
    doc = fitz.open()
    for _ in range(num_pages):
        page = doc.new_page()
        page.insert_text((72, 72), "Sample text " * 20)
    data = doc.tobytes()
    doc.close()
    return data


def _make_png_bytes() -> bytes:
    img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_valid_pdf_passes():
    result = validate_file("statement.pdf", _make_pdf_bytes(2))
    assert result.is_valid is True
    assert result.page_count == 2
    assert result.file_type == "pdf"
    assert result.errors == []


def test_pdf_exceeding_page_limit_fails():
    result = validate_file("statement.pdf", _make_pdf_bytes(5))
    assert result.is_valid is False
    assert any("exceeds" in e for e in result.errors)


def test_valid_png_passes():
    result = validate_file("invoice.png", _make_png_bytes())
    assert result.is_valid is True
    assert result.file_type == "png"


def test_unsupported_extension_rejected():
    result = validate_file("document.docx", b"not really a docx but has bytes")
    assert result.is_valid is False
    assert any("Unsupported file type" in e for e in result.errors)


def test_empty_file_rejected():
    result = validate_file("empty.pdf", b"")
    assert result.is_valid is False
    assert "Uploaded file is empty." in result.errors


def test_corrupted_pdf_rejected():
    result = validate_file("broken.pdf", b"%PDF-1.4 this is not a valid pdf stream")
    assert result.is_valid is False
    assert any("Corrupted or unreadable" in e for e in result.errors)
