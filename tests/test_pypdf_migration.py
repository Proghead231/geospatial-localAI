"""Tests for pypdf migration."""
import os
import tempfile
from pypdf import PdfReader
import pytest


def test_pypdf_import():
    """Verify pypdf can be imported."""
    from pypdf import PdfReader
    assert PdfReader is not None


def test_pypdf_version():
    """Verify pypdf version is >= 3.0.0."""
    import pypdf
    version = tuple(map(int, pypdf.__version__.split('.')[:2]))
    assert version >= (3, 0), f"pypdf version {pypdf.__version__} < 3.0.0"


def test_pdf_reader_basic():
    """Test that PdfReader can be instantiated with a minimal PDF."""
    # Create a minimal valid PDF in memory
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Hello World) Tj
ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000360 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
434
%%EOF"""
    
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp.write(pdf_content)
        tmp_path = tmp.name
    
    try:
        reader = PdfReader(tmp_path)
        assert len(reader.pages) == 1
        text = reader.pages[0].extract_text()
        assert "Hello World" in text
    finally:
        os.unlink(tmp_path)


def test_requirements_updated():
    """Verify requirements.txt uses pypdf instead of PyPDF2."""
    req_path = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')
    with open(req_path, 'r') as f:
        content = f.read()
    
    assert 'pypdf' in content.lower(), "pypdf not found in requirements.txt"
    assert 'pypdf2' not in content.lower(), "PyPDF2 still present in requirements.txt"


def test_main_imports():
    """Verify main.py imports from pypdf, not PyPDF2."""
    main_path = os.path.join(os.path.dirname(__file__), '..', 'main.py')
    with open(main_path, 'r') as f:
        content = f.read()
    
    assert 'from pypdf import' in content, "main.py should import from pypdf"
    assert 'from PyPDF2 import' not in content, "main.py still imports from PyPDF2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
