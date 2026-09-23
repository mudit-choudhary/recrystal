# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Mudit Choudhary
"""PDF -> layout (the `textreflow` input contract) -> typed blocks.

Combines the layout detector's regions with PyMuPDF's words. Every word is
assigned to the detected region containing its centre (smallest region wins,
so a Caption sitting on top of a Picture claims its own words). Words covered
by no detection are grouped into fallback Text regions so no content is lost —
unless they sit inside a Picture/Page-header/Page-footer box, in which case
they are recorded in `swallowed_text` but excluded from the text flow.
"""

from pathlib import Path

import fitz
import textreflow

from .detector import LayoutDetector

# Words whose centre falls in no detected region are grouped into fallback
# Text regions unless they sit inside one of these region types.
SWALLOW_LABELS = {"Picture", "Page-header", "Page-footer"}

_detector = None


def get_detector():
    """A process-wide LayoutDetector on the default weights, built on first use."""
    global _detector
    if _detector is None:
        _detector = LayoutDetector()
    return _detector


def _center_in(bbox, x, y):
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _area(bbox):
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def assign_words_to_regions(words, regions):
    """Attach each PyMuPDF word to the smallest region containing its center.

    `words` are PyMuPDF tuples (x0, y0, x1, y1, text, block_no, line_no, word_no).
    Returns (regions, swallowed) where each region gains a "words" list and
    fallback Text regions are appended for uncovered words; `swallowed` holds
    words dropped because they sit inside Picture/header/footer boxes.
    """
    for region in regions:
        region["words"] = []

    # Sort candidate regions by area so the first containing hit is the smallest.
    by_area = sorted(regions, key=lambda r: _area(r["bbox"]))
    leftovers = {}  # PyMuPDF block_no -> words
    swallowed = []

    for w in words:
        x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
        if not text.strip():
            continue
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

        hit = next((r for r in by_area if _center_in(r["bbox"], cx, cy)), None)
        if hit is not None:
            if hit["label"] in SWALLOW_LABELS:
                swallowed.append({"text": text, "label": hit["label"],
                                  "bbox": [x0, y0, x1, y1]})
            else:
                hit["words"].append([x0, y0, x1, y1, text])
        else:
            leftovers.setdefault(w[5], []).append([x0, y0, x1, y1, text])

    for block_words in leftovers.values():
        xs0 = min(w[0] for w in block_words)
        ys0 = min(w[1] for w in block_words)
        xs1 = max(w[2] for w in block_words)
        ys1 = max(w[3] for w in block_words)
        regions.append({
            "label": "Text",
            "conf": 0.0,
            "fallback": True,
            "bbox": [xs0, ys0, xs1, ys1],
            "words": block_words,
        })

    return regions, swallowed


def find_gutters(words, min_gap=None, min_rows=3):
    """Vertical whitespace gutters inside one region's words.

    A gutter is an x-interval *no* word in the region crosses, at least
    `min_gap` wide, with words on both sides on at least `min_rows` different
    text rows. Both conditions matter: the first rules out ordinary word
    spacing (some line always covers that x in running prose), the second
    rules out the one-off wide space of a centred heading or a tab stop.

    `min_rows` is 3 rather than 2 because two rows are cheap to align by
    coincidence — a two-line region with a chance gap at the same x would
    otherwise be torn in half. Real column layouts run to many rows (the
    author blocks this was built for have four or five).

    `min_gap` defaults to roughly one line-height, measured from the region's
    own words, because column gutters scale with the font: in a 10pt paper,
    word spaces run ~2.5pt and real gutters ~12pt.

    Returns the gutter mid-points, left to right.
    """
    if len(words) < 2 * min_rows:
        return []
    if min_gap is None:
        heights = sorted(w[3] - w[1] for w in words)
        min_gap = max(6.0, 0.9 * heights[len(heights) // 2])

    # candidate gaps: sweep left to right, tracking the rightmost edge so far
    gaps, reach = [], None
    for w in sorted(words, key=lambda w: w[0]):
        if reach is not None and w[0] - reach > min_gap:
            gaps.append((reach, w[0]))
        reach = max(reach or w[2], w[2])
    if not gaps:
        return []

    # rows: cluster words by vertical centre so "both sides on the same line"
    # can be counted
    heights = sorted(w[3] - w[1] for w in words)
    tol = max(2.0, heights[len(heights) // 2] * 0.6)
    rows, last = [], None
    for w in sorted(words, key=lambda w: (w[1] + w[3]) / 2):
        cy = (w[1] + w[3]) / 2
        if last is not None and abs(cy - last) <= tol:
            rows[-1].append(w)
        else:
            rows.append([w])
        last = cy

    accepted = []
    for left_edge, right_edge in gaps:
        spanning = sum(1 for row in rows
                       if any(w[2] <= left_edge for w in row) and any(w[0] >= right_edge for w in row))
        if spanning >= min_rows:
            accepted.append((left_edge + right_edge) / 2)
    return accepted


# Labels a wide region may be split on, as an *allowlist* — a new or
# unrecognised label is never split, which is the safe default. Splitting only
# helps where the layout model merged content from two page columns into one
# region, and only where the parts stay meaningful on their own:
#
#   Text       the mid-page column merge this was written for
#   Authors    the 3-across author block
#   List-item  two side-by-side list items merged into one region
#   Caption    captions of side-by-side subfigures
#   Footnote   the full-width footnote band under a two-column page
#
# Everything else is excluded deliberately:
#   Table, Formula   their columns belong to their *rows*; splitting on a
#                    gutter separates row labels from their values
#   Picture          read as one unit; its text is swallowed anyway
#   Title,           a heading is one logical unit by definition, so splitting
#   Section-header   can only ever turn one heading into two
#   Page-header,     dropped before assembly, so splitting them
#   Page-footer      is work with no effect on the output
SPLITTABLE_LABELS = {"Text", "Authors", "List-item", "Caption", "Footnote"}


# Each column produced by a split must be at least this fraction of the
# region's width. A genuine column is ~45% of a two-column region; a
# pseudocode line-number margin ("1:", "2:", …) is ~3%, and splitting there
# would strip the numbers off their statements.
MIN_COLUMN_FRACTION = 0.15


def split_region_columns(region, page_width, min_width_fraction=0.5, min_gap=None):
    """Split a region whose words form separate columns into one region per
    column; otherwise return it unchanged (as a single-item list).

    Fixes two real cases: a 3-across `Authors` block that would otherwise be
    read across the page ("Alice   Bob | Univ A   Univ B"), and a two-column
    stretch the layout model merged into one wide `Text` box, whose lines
    would otherwise be stitched together across the gutter.

    Only labels in `SPLITTABLE_LABELS` are eligible; see the note there for
    why each of the others is excluded.
    """
    words = region.get("words") or []
    width = region["bbox"][2] - region["bbox"][0]
    if not words or region["label"] not in SPLITTABLE_LABELS or width < min_width_fraction * page_width:
        return [region]

    cuts = find_gutters(words, min_gap)
    if not cuts:
        return [region]

    groups = [[] for _ in range(len(cuts) + 1)]
    for w in words:
        centre = (w[0] + w[2]) / 2
        idx = sum(1 for c in cuts if centre > c)
        groups[idx].append(w)

    out = []
    for group in groups:
        if not group:
            continue
        out.append({**region, "words": group, "column_split": True,
                    "bbox": [round(min(w[0] for w in group), 2), round(min(w[1] for w in group), 2),
                             round(max(w[2] for w in group), 2), round(max(w[3] for w in group), 2)]})
    if len(out) < 2:
        return [region]
    # Reject a split that carves off a sliver — that is a margin (line
    # numbers, bullets), not a column.
    if any((p["bbox"][2] - p["bbox"][0]) < MIN_COLUMN_FRACTION * width for p in out):
        return [region]
    return out


def words_to_lines(region_words):
    """Group a region's words into reading-order lines of text."""
    if not region_words:
        return []

    heights = sorted(w[3] - w[1] for w in region_words)
    line_tol = max(2.0, heights[len(heights) // 2] * 0.6)

    lines = []  # each: {"y": center, "words": [...]}
    for w in sorted(region_words, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        cy = (w[1] + w[3]) / 2
        if lines and abs(cy - lines[-1]["y"]) <= line_tol:
            lines[-1]["words"].append(w)
            n = len(lines[-1]["words"])
            lines[-1]["y"] += (cy - lines[-1]["y"]) / n
        else:
            lines.append({"y": cy, "words": [w]})

    return [" ".join(w[4] for w in sorted(l["words"], key=lambda w: w[0]))
            for l in lines]


def extract_pages(pdf_path, detector=None, max_pages=None):
    """Detect, attach words, split merged columns and build lines.

    Returns the `pages` list of a layout: per page, `page`, `width`, `height`,
    `regions` (each with `label`, `conf`, `bbox`, `lines`) and `swallowed_text`.
    """
    detector = detector or get_detector()
    layout_pages = detector.detect_pdf(pdf_path, max_pages=max_pages)

    doc = fitz.open(pdf_path)
    for page_entry in layout_pages:
        page = doc[page_entry["page"]]
        words = page.get_text("words")
        regions, swallowed = assign_words_to_regions(words, page_entry["regions"])

        # A region the model merged across a column gutter is split here,
        # before lines are built — otherwise its lines would be stitched
        # together across the columns.
        regions = [part for r in regions
                   for part in split_region_columns(r, page_entry["width"])]

        for region in regions:
            region["lines"] = words_to_lines(region.pop("words"))
        # Drop empty non-visual detections (visual ones are kept as anchors).
        page_entry["regions"] = [
            r for r in regions if r["lines"] or r["label"] in ("Picture", "Table", "Formula")
        ]
        page_entry["swallowed_text"] = swallowed
    doc.close()
    return layout_pages


def layout_pdf(pdf_path, detector=None, max_pages=None):
    """One PDF -> a layout dict, ready for `textreflow.assemble`.

    `num_pages` is what was parsed, `pdf_pages` what the document has, so a
    truncated parse (`max_pages`) stays detectable.
    """
    pages = extract_pages(pdf_path, detector=detector, max_pages=max_pages)
    with fitz.open(pdf_path) as doc:
        pdf_pages = len(doc)
    return {"source_pdf": str(Path(pdf_path)), "num_pages": len(pages),
            "pdf_pages": pdf_pages, "pages": pages}


def parse_pdf(pdf_path, detector=None, max_pages=None):
    """One PDF -> (blocks, dropped): typed blocks in reading order, and the
    page furniture removed from the flow. See `textreflow.assemble`."""
    return textreflow.assemble(layout_pdf(pdf_path, detector=detector, max_pages=max_pages))
