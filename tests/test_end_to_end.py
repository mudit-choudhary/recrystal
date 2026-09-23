"""PDF -> layout -> blocks through the real model, on a PDF built here.

Skipped unless the weights are already on disk (never downloads).
"""

import fitz
import pytest
import textreflow

from recrystal import layout_pdf, parse_pdf
from recrystal.detector import DEFAULT_WEIGHTS, LayoutDetector, weights_dir

WEIGHTS = weights_dir() / DEFAULT_WEIGHTS
pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason="weights not downloaded")

BODY = ("Graph neural networks learn representations by passing messages between "
        "neighbouring nodes. We study how the depth of such networks affects accuracy. ") * 6


@pytest.fixture(scope="module")
def pdf(tmp_path_factory):
    path = tmp_path_factory.mktemp("pdf") / "paper.pdf"
    doc = fitz.open()
    for _ in range(2):
        page = doc.new_page(width=612, height=792)
        page.insert_textbox(fitz.Rect(72, 72, 540, 720), BODY * 2, fontsize=10)
    doc.save(path)
    return path


@pytest.fixture(scope="module")
def detector():
    return LayoutDetector(WEIGHTS, providers=["CPUExecutionProvider"])


def test_layout_meets_the_textreflow_contract(pdf, detector):
    layout = layout_pdf(pdf, detector=detector, max_pages=1)
    assert textreflow.validate_layout(layout) == []
    assert (layout["num_pages"], layout["pdf_pages"]) == (1, 2)


def test_parse_pdf_recovers_the_text(pdf, detector):
    blocks, _ = parse_pdf(pdf, detector=detector)
    text = " ".join(b["text"] for b in blocks)
    assert "passing messages between neighbouring nodes" in text
    assert {b["page"] for b in blocks} == {0, 1}
