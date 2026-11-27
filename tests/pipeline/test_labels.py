"""Compound labels: the label pattern, ranges, text clean-up and drawing-to-label linking."""

from __future__ import annotations

import pytest

from molscout.pipeline.labels import (
    LABEL_RE,
    mentioned_labels,
    clean_text,
    expand_range,
    label_of_text,
    link_codes,
    link_regions,
)
from molscout.pipeline.schema import OcrLine

from .fixtures.a2.fakes import Block, Region, text


@pytest.mark.parametrize("token", ["12", "12a", "12b'", "S1", "S12b", "3", "105c", "7′"])
def test_label_re_accepts_journal_labels(token):
    assert LABEL_RE.fullmatch(token)


@pytest.mark.parametrize("token", ["2019", "1234", "IC50", "nM", "a12", "12-", "12ab", "", "12.5"])
def test_label_re_rejects_non_labels(token):
    assert not LABEL_RE.fullmatch(token)


@pytest.mark.parametrize(
    ("text_in", "expected"),
    [
        ("12a–f", ["12a", "12b", "12c", "12d", "12e", "12f"]),
        ("12a-12f", ["12a", "12b", "12c", "12d", "12e", "12f"]),
        ("12a − 12c", ["12a", "12b", "12c"]),
        ("5–8", ["5", "6", "7", "8"]),
        ("S1–S3", ["S1", "S2", "S3"]),
        ("12a'–c'", ["12a'", "12b'", "12c'"]),
        ("12a", ["12a"]),
        ("12a,b", ["12a", "12b"]),
        ("12a–c and 14", ["12a", "12b", "12c", "14"]),
        ("3, 5, 7", ["3", "5", "7"]),
        ("12a to 12c", ["12a", "12b", "12c"]),
    ],
)
def test_expand_range(text_in, expected):
    assert expand_range(text_in) == expected


@pytest.mark.parametrize("text_in", ["12a–13c", "8–5", "1–500", "12f–a", "hello", "", "12a–"])
def test_expand_range_refuses_unclear_ranges(text_in):
    assert expand_range(text_in) == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("$\\mathrm{IC}_{50}$ = 14 nM", "IC50 = 14 nM"),
        ("R$^{1}$", "R1"),
        ("IC $_{50}$ (μM)", "IC50 (μM)"),
        ("14$^{a}$", "14"),
        ("3.2$^{*,\\dagger}$", "3.2"),
        ("R¹ and R²", "R1 and R2"),
        ("K$_i$ = 3 nM", "Ki = 3 nM"),
        ("$\\mathbf{12a}$", "12a"),
        ("10$\\%$ inhibition", "10% inhibition"),
        ("12 \\mu M", "12 µM"),
        ("  many   spaces ", "many spaces"),
        ("IC 50", "IC50"),
    ],
)
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("12a", "12a"),
        ("(12a)", "12a"),
        ("**12a**", "12a"),
        ("Compound 12a", "12a"),
        ("12a, R = Me", "12a"),
        ("12a: IC50 = 14 nM", "12a"),
        ("12a (IC50 = 14 nM)", "12a"),
        ("CCT400028", "CCT400028"),
        ("**dAurA383**", "dAurA383"),
        ("A549", None),
        ("HCT116", None),
        ("IC50 = 14 nM", None),
        ("The compounds were tested against 12 kinases in this study.", None),
        ("2019", None),
        ("Figure 2", None),
        ("", None),
    ],
)
def test_label_of_text(body, expected):
    assert label_of_text(body) == expected


@pytest.mark.parametrize(
    ("caption", "expected"),
    [
        ("Figure 2. Structure of the lead compound 7.", ["7"]),
        ("Figure 3. Binding mode of **12a** in EGFR.", ["12a"]),
        ("Figure 1. Compounds 1 and 2 bound to the kinase.", ["1", "2"]),
        ("Figure 4. Docking poses in the ATP pocket (PDB 5HT2).", []),
        ("Figure 5. Dose response in A549 cells.", []),
        ("Figure 6. Design of compounds 12a–c.", ["12a", "12b", "12c"]),
        ("Thus, (R)- 16 was more potent than 10, whereas (S)-17 was weaker.", ["16", "17"]),
        ("(S)-3-(4-Chlorophenyl)-2-methylpropanoic acid", []),
    ],
)
def test_mentioned_labels(caption, expected):
    assert mentioned_labels(caption) == expected


def _region(rid: str, bbox: list[float], page: int = 1, **kw) -> Region:
    return Region(region_id=rid, page=page, bbox=bbox, **kw)


def test_links_label_below_drawing():
    regions = [_region("p1-r1", [100, 100, 200, 200])]
    blocks = [text(1, 0, [135, 206, 165, 216], "12a")]
    assert link_regions(regions, blocks) == {"p1-r1": "12a"}


def test_links_label_beside_drawing():
    regions = [_region("p1-r1", [100, 100, 200, 200])]
    blocks = [text(1, 0, [208, 140, 230, 152], "(7)")]
    assert link_regions(regions, blocks) == {"p1-r1": "7"}


def test_links_label_held_in_a_caption_block():
    regions = [_region("p1-r1", [100, 100, 200, 200])]
    blocks = [text(1, 3, [140, 204, 160, 214], "3b", kind="caption")]
    assert link_regions(regions, blocks) == {"p1-r1": "3b"}


def test_far_label_is_not_linked():
    regions = [_region("p1-r1", [100, 100, 200, 200])]
    blocks = [text(1, 0, [135, 300, 165, 310], "12a")]
    assert link_regions(regions, blocks) == {}


def test_threshold_scales_with_page_size():
    regions = [_region("p1-r1", [100, 100, 200, 200])]
    blocks = [text(1, 0, [135, 235, 165, 245], "12a")]
    assert link_regions(regions, blocks) == {"p1-r1": "12a"}
    assert link_regions(regions, blocks, page_sizes={1: (300.0, 400.0)}) == {}


def test_label_on_another_page_is_not_linked():
    regions = [_region("p1-r1", [100, 100, 200, 200])]
    blocks = [text(2, 0, [135, 206, 165, 216], "12a")]
    assert link_regions(regions, blocks) == {}


def test_each_drawing_gets_its_own_label():
    regions = [
        _region("p1-r1", [100, 100, 200, 200]),
        _region("p1-r2", [260, 100, 360, 200]),
    ]
    blocks = [
        text(1, 0, [140, 205, 160, 215], "1"),
        text(1, 1, [300, 205, 320, 215], "2"),
    ]
    assert link_regions(regions, blocks) == {"p1-r1": "1", "p1-r2": "2"}


def test_label_halfway_between_two_drawings_is_ambiguous():
    regions = [
        _region("p1-r1", [100, 100, 200, 200]),
        _region("p1-r2", [240, 100, 340, 200]),
    ]
    blocks = [text(1, 0, [210, 140, 230, 152], "5")]
    assert link_regions(regions, blocks) == {}


def test_long_paragraph_below_drawing_is_not_a_label():
    regions = [_region("p1-r1", [100, 100, 200, 200])]
    body = (
        "Compound 12a was prepared in three steps from commercially available starting materials."
    )
    blocks = [text(1, 0, [100, 205, 500, 240], body)]
    assert link_regions(regions, blocks) == {}


def test_non_structure_region_is_skipped():
    regions = [_region("p1-r1", [100, 100, 200, 200], is_structure=False)]
    blocks = [text(1, 0, [135, 206, 165, 216], "12a")]
    assert link_regions(regions, blocks) == {}


def _figure(caption: str) -> list[Block]:
    return [
        Block(page=1, index=4, type="image", bbox=[50, 50, 400, 300]),
        Block(page=1, index=5, type="caption", bbox=[50, 305, 400, 330], text=caption, parent=4),
    ]


def test_caption_label_used_when_figure_holds_one_drawing():
    regions = [_region("p1-r1", [100, 100, 250, 250])]
    blocks = _figure("Figure 2. Structure of the lead compound 7.")
    assert link_regions(regions, blocks) == {"p1-r1": "7"}


def test_caption_label_not_used_for_figure_with_two_drawings():
    regions = [
        _region("p1-r1", [60, 100, 180, 250]),
        _region("p1-r2", [220, 100, 380, 250]),
    ]
    blocks = _figure("Figure 2. Structure of the lead compound 7.")
    assert link_regions(regions, blocks) == {}


def test_caption_with_two_labels_is_not_used():
    regions = [_region("p1-r1", [100, 100, 250, 250])]
    blocks = _figure("Figure 1. Compounds 1 and 2 bound to the kinase.")
    assert link_regions(regions, blocks) == {}


def test_direct_label_wins_over_caption():
    regions = [_region("p1-r1", [100, 100, 250, 250])]
    blocks = [
        *_figure("Figure 2. Structure of the lead compound 7."),
        text(1, 6, [160, 254, 180, 264], "9"),
    ]
    assert link_regions(regions, blocks) == {"p1-r1": "9"}


def test_page_number_in_the_margin_is_not_a_label():
    regions = [_region("p1-r1", [100, 700, 200, 750])]
    blocks = [text(1, 0, [140, 762, 160, 772], "270")]
    assert link_regions(regions, blocks) == {}


def test_mentioned_labels_ignore_hydrogen_locants_and_capitals():
    assert mentioned_labels("the quinolin-2(1H)-one core of compound 4A") == []


# --- legend codes (link_codes) ----------------------------------------------------------------


def test_a_code_printed_under_a_drawing_links_it_though_it_is_no_compound_label():
    regions = [_region("p1-r1", [100, 100, 200, 200])]
    blocks = [text(1, 0, [135, 206, 165, 216], "(M0)")]
    assert link_regions(regions, blocks) == {}
    assert link_codes(regions, blocks, lambda words: {"M0": "M0"}.get(words.strip("()"))) == {"p1-r1": "M0"}


def test_a_compound_label_nearer_the_drawing_wins_over_a_code():
    regions = [_region("p1-r1", [100, 100, 200, 200])]
    blocks = [text(1, 0, [135, 204, 165, 214], "12a"), text(1, 1, [135, 225, 165, 235], "M0")]
    assert link_codes(regions, blocks, {"M0": "M0"}.get) == {}


def test_a_code_read_by_ocr_links_its_drawing():
    regions = [_region("p1-r1", [100, 100, 200, 200])]
    lines = [OcrLine(page=1, bbox=[140, 206, 160, 216], text="M3", score=0.99, source_block="image:p1-1")]
    assert link_codes(regions, [], {"M3": "M3"}.get, ocr_lines=lines) == {"p1-r1": "M3"}
    assert link_codes(regions, [], {"M4": "M4"}.get, ocr_lines=lines) == {}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<eq>\\mathsf { I C } _ { 5 0 } \\mu \\mathsf { M }</eq>", "IC50µM"),
        ("$0 . 3 9 6   \\pm   0 . 1 8 0$", "0.396±0.180"),
        ("KDM4A <eq>\\mathsf { I C } _ { 5 0 }   \\mu \\mathsf { M }</eq>", "KDM4A IC50µM"),
        ("$5 \\times 10 ^ { - 3 }$", "5×10-3"),
    ],
)
def test_clean_text_reads_spaced_mineru_math(raw, expected):
    assert clean_text(raw) == expected
