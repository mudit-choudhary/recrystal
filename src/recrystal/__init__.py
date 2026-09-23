# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Mudit Choudhary
"""recrystal — a layout parser for research papers: PDF in, typed blocks out.

A fine-tuned YOLO11 layout detector (run through onnxruntime) finds the
regions, PyMuPDF supplies the words, and `textreflow` assembles them into
ordered, typed blocks — the input of `grain_growth` (grain-growth-chunking).

    from recrystal import parse_pdf

    blocks, dropped = parse_pdf("paper.pdf")
    # [{"type": "title", "page": 0, "text": "..."}, {"type": "paragraph", ...}, ...]

The weights are downloaded from Hugging Face on first use (see
`recrystal.detector.weights_path`). `layout_pdf` returns the intermediate
layout instead, for inspection or a different assembler.
"""

from .detector import LayoutDetector, OnnxYolo, letterbox, nms, weights_path
from .parser import (
    SPLITTABLE_LABELS,
    SWALLOW_LABELS,
    assign_words_to_regions,
    extract_pages,
    find_gutters,
    get_detector,
    layout_pdf,
    parse_pdf,
    split_region_columns,
    words_to_lines,
)

__version__ = "1.0.0"

__all__ = [
    "LayoutDetector", "OnnxYolo", "SPLITTABLE_LABELS", "SWALLOW_LABELS",
    "assign_words_to_regions", "extract_pages", "find_gutters", "get_detector",
    "layout_pdf", "letterbox", "nms", "parse_pdf", "split_region_columns",
    "weights_path", "words_to_lines", "__version__",
]
