"""Word-to-region assignment, column splitting and line building (no model required)."""

import pytest

from recrystal.parser import (
    SPLITTABLE_LABELS,
    assign_words_to_regions,
    find_gutters,
    split_region_columns,
    words_to_lines,
)


def region(label, x0, y0, x1, y1, conf=0.9):
    return {"label": label, "conf": conf, "bbox": [x0, y0, x1, y1]}


def word(x0, y0, x1, y1, text, block_no=0):
    return (x0, y0, x1, y1, text, block_no, 0, 0)


class TestAssignment:
    def test_word_goes_to_containing_region(self):
        regions = [region("Text", 0, 0, 100, 100)]
        out, swallowed = assign_words_to_regions([word(10, 10, 30, 20, "hello")], regions)
        assert out[0]["words"][0][4] == "hello"
        assert swallowed == []

    def test_smallest_region_wins(self):
        # A Caption box sitting inside a Picture box claims its own words.
        pic = region("Picture", 0, 0, 300, 300)
        cap = region("Caption", 20, 250, 280, 290)
        out, swallowed = assign_words_to_regions(
            [word(30, 260, 60, 280, "Figure"), word(150, 100, 170, 110, "axis")],
            [pic, cap],
        )
        cap_out = next(r for r in out if r["label"] == "Caption")
        assert [w[4] for w in cap_out["words"]] == ["Figure"]
        # The word inside only the Picture is swallowed, not kept in the flow.
        assert [s["text"] for s in swallowed] == ["axis"]

    def test_uncovered_words_become_fallback_text(self):
        regions = [region("Text", 0, 0, 100, 100)]
        out, _ = assign_words_to_regions(
            [word(500, 500, 520, 510, "stray", block_no=3)], regions)
        fallback = [r for r in out if r.get("fallback")]
        assert len(fallback) == 1
        assert fallback[0]["label"] == "Text"
        assert fallback[0]["words"][0][4] == "stray"

    def test_header_text_swallowed(self):
        regions = [region("Page-header", 0, 0, 600, 30)]
        out, swallowed = assign_words_to_regions([word(10, 5, 80, 25, "Running")], regions)
        assert swallowed[0]["label"] == "Page-header"
        assert all(not r.get("words") for r in out)


def row(y, entries):
    """entries: [(x0, x1, text), ...] on one text row."""
    return [[x0, y, x1, y + 10, text] for x0, x1, text in entries]


class TestColumnSplitting:
    def test_two_column_merge_is_split(self):
        # One wide "Text" region the model merged across the gutter: each row
        # has words at 40-280 and 330-570, with ordinary 3pt word spacing
        # inside each column and a 50pt gutter between them.
        words = []
        for y in (100, 115, 130, 145):
            words += row(y, [(40, 150, "left"), (153, 280, "side"), (330, 440, "right"), (443, 570, "side")])
        region = {"label": "Text", "conf": 0.9, "bbox": [40, 100, 570, 155], "words": words}
        parts = split_region_columns(region, page_width=612)
        assert len(parts) == 2
        assert all(p["column_split"] for p in parts)
        assert [words_to_lines(p["words"])[0] for p in parts] == ["left side", "right side"]
        assert parts[0]["bbox"][2] < parts[1]["bbox"][0]      # left part ends before the right begins

    def test_three_across_authors_split(self):
        # A real author block is name / university / city — three rows or
        # more, which is why min_rows can be 3 without losing this case.
        words = row(80, [(40, 150, "Alice"), (240, 350, "Bob"), (440, 550, "Carol")])
        words += row(96, [(40, 150, "UniA"), (240, 350, "UniB"), (440, 550, "UniC")])
        words += row(112, [(40, 150, "CityA"), (240, 350, "CityB"), (440, 550, "CityC")])
        region = {"label": "Authors", "conf": 0.9, "bbox": [40, 80, 550, 122], "words": words}
        parts = split_region_columns(region, page_width=612)
        assert [words_to_lines(p["words"]) for p in parts] == [
            ["Alice", "UniA", "CityA"], ["Bob", "UniB", "CityB"], ["Carol", "UniC", "CityC"]]

    def test_two_aligned_rows_are_not_enough(self):
        """Two rows can align by coincidence; a real column layout has more."""
        words = row(100, [(40, 150, "Short"), (400, 560, "tail")])
        words += row(115, [(40, 150, "Also"), (400, 560, "here")])
        region = {"label": "Text", "conf": 0.9, "bbox": [40, 100, 560, 125], "words": words}
        assert len(split_region_columns(region, page_width=612)) == 1

    def test_narrow_margin_is_not_a_column(self):
        """A pseudocode line-number margin ("1:", "2:") is a sliver, not a
        column — splitting there would strip the numbers off their lines."""
        words = []
        for i, y in enumerate((100, 115, 130, 145, 160)):
            words += row(y, [(40, 58, f"{i+1}:")] + [(120 + 55 * j, 170 + 55 * j, f"tok{j}")
                                                     for j in range(8)])
        region = {"label": "Text", "conf": 0.9, "bbox": [40, 100, 555, 170], "words": words}
        assert len(split_region_columns(region, page_width=612)) == 1

    def test_table_is_never_split(self):
        # A table's gutters separate columns of the same rows; splitting it
        # would leave the row labels in one region and the numbers in others.
        words = row(100, [(40, 150, "Method"), (240, 350, "Cora"), (440, 550, "Pubmed")])
        words += row(115, [(40, 150, "Degree"), (240, 350, "91.67"), (440, 550, "82.70")])
        words += row(130, [(40, 150, "PageRank"), (240, 350, "92.41"), (440, 550, "83.41")])
        region = {"label": "Table", "conf": 0.9, "bbox": [40, 100, 550, 140], "words": words}
        parts = split_region_columns(region, page_width=612)
        assert len(parts) == 1
        assert words_to_lines(parts[0]["words"]) == [
            "Method Cora Pubmed", "Degree 91.67 82.70", "PageRank 92.41 83.41"]

    def _two_column_words(self):
        w = []
        for y in (100, 115, 130):
            w += row(y, [(40, 150, "left"), (153, 280, "side"), (330, 440, "right"), (443, 570, "side")])
        return w

    @pytest.mark.parametrize("label", ["Table", "Formula", "Picture",
                                       "Title", "Section-header",
                                       "Page-header", "Page-footer"])
    def test_non_splittable_labels_are_left_whole(self, label):
        """Only SPLITTABLE_LABELS may be divided. Tables and formulas would
        lose their rows; a heading is one unit; page headers/footers are
        dropped later anyway. Anything unrecognised must default to safe."""
        region = {"label": label, "conf": 0.9, "bbox": [40, 100, 570, 145],
                  "words": self._two_column_words()}
        assert len(split_region_columns(region, page_width=612)) == 1

    @pytest.mark.parametrize("label", sorted(SPLITTABLE_LABELS))
    def test_splittable_labels_do_split(self, label):
        region = {"label": label, "conf": 0.9, "bbox": [40, 100, 570, 145],
                  "words": self._two_column_words()}
        assert len(split_region_columns(region, page_width=612)) == 2

    def test_unknown_label_defaults_to_not_splitting(self):
        region = {"label": "SomeFutureClass", "conf": 0.9, "bbox": [40, 100, 570, 145],
                  "words": self._two_column_words()}
        assert len(split_region_columns(region, page_width=612)) == 1

    def test_normal_paragraph_not_split(self):
        words = []
        for y in (100, 112, 124):
            words += row(y, [(40 + 60 * i, 95 + 60 * i, f"w{i}") for i in range(9)])
        region = {"label": "Text", "conf": 0.9, "bbox": [40, 100, 575, 134], "words": words}
        assert len(split_region_columns(region, page_width=612)) == 1

    def test_narrow_region_never_split(self):
        words = row(100, [(40, 90, "a"), (200, 250, "b")]) + row(115, [(40, 90, "c"), (200, 250, "d")])
        region = {"label": "Text", "conf": 0.9, "bbox": [40, 100, 250, 125], "words": words}
        assert len(split_region_columns(region, page_width=612)) == 1

    def test_one_off_wide_space_is_not_a_gutter(self):
        # a single centred heading line with a big gap: only one row spans it
        words = row(100, [(40, 150, "Title"), (400, 560, "Continued")])
        words += row(115, [(40 + 40 * i, 75 + 40 * i, f"w{i}") for i in range(13)])
        assert find_gutters(words, min_gap=14.0) == []


class TestWordsToLines:
    def test_lines_in_reading_order(self):
        words = [
            [10, 20, 40, 30, "world"],
            [0, 20, 9, 30, "hello"],
            [0, 40, 30, 50, "second"],
        ]
        assert words_to_lines(words) == ["hello world", "second"]

    def test_slight_baseline_jitter_same_line(self):
        words = [
            [0, 20.0, 30, 30.0, "left"],
            [40, 21.5, 70, 31.5, "right"],
        ]
        assert words_to_lines(words) == ["left right"]

    def test_empty(self):
        assert words_to_lines([]) == []
