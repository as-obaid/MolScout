"""R-group tables: HTML parsing, header reading, label columns, ranges and page splits."""

from __future__ import annotations

import pytest

from molscout.pipeline.tables import (
    IMAGE_CELL,
    find_rgroup_tables,
    parse_html_table,
    parse_table_block,
    variable_name,
)

from .fixtures.a2.fakes import Block, html_table, table, text


def test_parse_simple_table():
    html = "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>"
    assert parse_html_table(html) == [["a", "b"], ["c", "d"]]


def test_parse_expands_rowspan_and_colspan():
    html = (
        "<table><thead><tr><th rowspan='2'>Compd</th>"
        '<th colspan="2">IC<sub>50</sub> (nM)</th></tr>'
        "<tr><th>EGFR</th><th>HER2</th></tr></thead>"
        "<tbody><tr><td>1</td><td>5</td><td>7</td></tr></tbody></table>"
    )
    assert parse_html_table(html) == [
        ["Compd", "IC50 (nM)", "IC50 (nM)"],
        ["Compd", "EGFR", "HER2"],
        ["1", "5", "7"],
    ]


def test_parse_rowspan_in_middle_column():
    html = (
        "<table><tr><td>1a</td><td rowspan=3>H</td><td>Me</td></tr>"
        "<tr><td>1b</td><td>Et</td></tr><tr><td>1c</td><td>Pr</td></tr></table>"
    )
    assert parse_html_table(html) == [["1a", "H", "Me"], ["1b", "H", "Et"], ["1c", "H", "Pr"]]


def test_parse_keeps_sub_and_superscripts_and_drops_footnotes():
    html = (
        "<table><tr><td>R<sub>1</sub></td><td>IC<sub>50</sub> (μM)<sup>a</sup></td>"
        "<td>R<sup>2</sup></td><td>14<sup>*</sup></td><td>3.2 ± 0.1<sup>b,c</sup></td>"
        "<td>12a†</td><td>NH<sub>3</sub><sup>+</sup></td></tr></table>"
    )
    assert parse_html_table(html) == [["R1", "IC50 (μM)", "R2", "14", "3.2 ± 0.1", "12a", "NH3+"]]


def test_parse_handles_latex_cells_line_breaks_and_images():
    html = (
        "<table><tr><td>$\\mathrm{R}^{1}$</td><td>4-F<br/>phenyl</td>"
        "<td><img src='images/c1.jpg'/></td><td><b>12a</b></td></tr></table>"
    )
    assert parse_html_table(html) == [
        ["R1", "4-F phenyl", f"{IMAGE_CELL[:-1]}:images/c1.jpg]", "12a"]
    ]


def test_parse_pads_ragged_rows_and_ignores_bad_spans():
    html = "<table><tr><td colspan='x'>a</td><td>b</td><td>c</td></tr><tr><td>d</td></tr></table>"
    assert parse_html_table(html) == [["a", "b", "c"], ["d", "", ""]]


def test_parse_empty_or_broken_html():
    assert parse_html_table("") == []
    assert parse_html_table("<table><tr><td>a") == [["a"]]


def _basic() -> Block:
    return table(
        3,
        7,
        html_table(
            [
                ["Compd", "R<sup>1</sup>", "R<sup>2</sup>", "IC<sub>50</sub> (nM)<sup>a</sup>"],
                ["12a", "H", "Me", "14"],
                ["12b", "F", "Me", "&gt;1000"],
                ["12c", "Cl", "Et", "NT"],
            ]
        ),
    )


def test_find_basic_rgroup_table():
    (found,) = find_rgroup_tables([_basic()])
    assert found.table_id == "p3-t7"
    assert found.page == 3
    assert found.bbox == [50, 300, 550, 600]
    assert found.label_col == 0
    assert found.variables == {"R1": 1, "R2": 2}
    assert found.activity_cols == ["IC50 (nM)"]
    assert found.headers == ["Compd", "R1", "R2", "IC50 (nM)"]
    assert [found.label_of(r) for r in found.rows] == ["12a", "12b", "12c"]
    assert found.rows[1] == {"Compd": "12b", "R1": "F", "R2": "Me", "IC50 (nM)": ">1000"}


def test_two_row_header_is_combined():
    html = (
        "<table><tr><th rowspan='2'>Cmpd</th><th rowspan='2'>R</th>"
        "<th colspan='2'>IC50 (nM)</th></tr>"
        "<tr><th>EGFR</th><th>HER2</th></tr>"
        "<tr><td>5a</td><td>OMe</td><td>3</td><td>40</td></tr>"
        "<tr><td>5b</td><td>CF3</td><td>8</td><td>90</td></tr></table>"
    )
    (found,) = find_rgroup_tables([table(2, 1, html)])
    assert found.variables == {"R": 1}
    assert found.activity_cols == ["IC50 (nM) EGFR", "IC50 (nM) HER2"]
    assert [found.label_of(r) for r in found.rows] == ["5a", "5b"]
    assert found.rows[0]["IC50 (nM) HER2"] == "40"


def test_label_column_found_from_values_when_header_is_blank():
    html = html_table(
        [["", "X", "Ki (nM)"], ["7a", "N", "12"], ["7b", "CH", "30"], ["7c", "O", "5"]]
    )
    (found,) = find_rgroup_tables([table(1, 0, html)])
    assert found.label_col == 0
    assert found.variables == {"X": 1}
    assert found.activity_cols == ["Ki (nM)"]


def test_label_column_not_first():
    html = html_table(
        [
            ["Entry", "Ar", "Product", "Yield (%)"],
            ["1", "Ph", "3a", "80"],
            ["2", "4-MeOC6H4", "3b", "75"],
        ]
    )
    (found,) = find_rgroup_tables([table(1, 0, html)])
    assert found.label_col == 2
    assert [found.label_of(r) for r in found.rows] == ["3a", "3b"]
    assert found.variables == {"Ar": 1}


def test_variable_header_spellings():
    html = html_table(
        [
            ["No.", "R 1", "R′", "Ar<sub>1</sub>", "Het", "Linker", "Y", "A", "B"],
            ["9", "H", "Me", "Ph", "pyridin-3-yl", "CH2", "O", "N", "CH"],
        ]
    )
    (found,) = find_rgroup_tables([table(1, 0, html)])
    assert list(found.variables) == ["R1", "R'", "Ar1", "Het", "linker", "Y", "A", "B"]


def test_label_range_row_is_split_across_values():
    html = html_table(
        [
            ["Compd", "R1", "R2", "IC50 (µM)"],
            ["12a–c", "H, Me, Et", "Cl", "1.2, 3.4, 5.6"],
            ["13", "F", "Br", "7"],
        ]
    )
    (found,) = find_rgroup_tables([table(1, 0, html)])
    assert [found.label_of(r) for r in found.rows] == ["12a", "12b", "12c", "13"]
    assert [r["R1"] for r in found.rows] == ["H", "Me", "Et", "F"]
    assert [r["R2"] for r in found.rows] == ["Cl", "Cl", "Cl", "Br"]
    assert found.rows[2]["IC50 (µM)"] == "5.6"


def test_label_range_row_kept_whole_when_values_do_not_split():
    html = html_table([["Compd", "R1", "R2"], ["13a–c", "H, Me", "Cl"]])
    (found,) = find_rgroup_tables([table(1, 0, html)])
    assert [found.label_of(r) for r in found.rows] == ["13a–c"]


def test_substituent_with_locant_commas_is_not_split():
    html = html_table([["Compd", "Ar"], ["4a,b", "2,4-diCl-Ph, 3,5-diF-Ph"]])
    (found,) = find_rgroup_tables([table(1, 0, html)])
    assert [r["Ar"] for r in found.rows] == ["2,4-diCl-Ph", "3,5-diF-Ph"]


def test_section_rows_and_empty_rows_are_skipped():
    html = (
        "<table><tr><th>Compd</th><th>R</th><th>MIC (µg/mL)</th></tr>"
        "<tr><td colspan='3'>Series B</td></tr><tr><td>8</td><td>Cl</td><td>2</td></tr>"
        "<tr><td></td><td></td><td></td></tr></table>"
    )
    (found,) = find_rgroup_tables([table(1, 0, html)])
    assert [found.label_of(r) for r in found.rows] == ["8"]
    assert found.activity_cols == ["MIC (µg/mL)"]


def test_tables_without_variables_or_labels_are_not_rgroup_tables():
    activity_only = table(1, 0, html_table([["Compd", "IC50 (nM)", "SI"], ["1", "5", "10"]]))
    conditions = table(1, 1, html_table([["Entry", "Solvent", "Yield (%)"], ["1", "DMF", "60"]]))
    no_labels = table(1, 2, html_table([["R", "Count"], ["H", "many"], ["Me", "few"]]))
    assert find_rgroup_tables([activity_only, conditions, no_labels]) == []
    parsed = parse_table_block(activity_only)
    assert parsed is not None
    assert parsed.variables == {}
    assert parsed.activity_cols == ["IC50 (nM)"]


def test_non_table_blocks_and_missing_html_are_ignored():
    blocks = [
        text(1, 0, [0, 0, 10, 10], "Compd R1"),
        Block(page=1, index=1, type="table", bbox=[0, 0, 1, 1]),
    ]
    assert find_rgroup_tables(blocks) == []


def test_activity_header_variants():
    html = html_table(
        [
            [
                "Compd",
                "R",
                "pIC50",
                "% inhibition at 10 µM",
                "EC50 (µM)",
                "GI50",
                "Kd (nM)",
                "Selectivity",
            ],
            ["1", "H", "7.2", "95", "0.3", "1", "4", "10"],
        ]
    )
    (found,) = find_rgroup_tables([table(1, 0, html)])
    assert found.activity_cols == [
        "pIC50",
        "% inhibition at 10 µM",
        "EC50 (µM)",
        "GI50",
        "Kd (nM)",
    ]


@pytest.mark.parametrize(("header", "name"), [("[R]", "R"), ("(R1)", "R1"), ("[R]=", "R"), ("[x]", None)])
def test_variable_name_reads_a_bracketed_header(header, name):
    assert variable_name(header) == name


@pytest.mark.parametrize(("header", "name"), [("Ra", "Ra"), ("R d", "Rd"), ("Rc'", "Rc'"), ("RA", None), ("Rab", None)])
def test_variable_name_reads_an_r_with_a_letter(header, name):
    assert variable_name(header) == name


def test_lettered_r_headers_are_variables():
    html = html_table([["Compd", "A", "Ra", "Rb", "IC50 (nM)"], ["4", "N", "H", "Me", "12"]])
    (found,) = find_rgroup_tables([table(1, 0, html)])
    assert list(found.variables) == ["A", "Ra", "Rb"]


def test_a_variable_header_spanning_the_label_column_names_the_column_beside_it():
    # MinerU's standard tier prints "[R]" once over the label column and the R column (26_6st3).
    html = ("<table><tr><td rowspan='2' colspan='2'>[R]</td><td colspan='2'><img src='core.jpg'/></td></tr>"
            "<tr><td>FIH IC50 μM</td><td>KDM4A IC50 μM</td></tr>"
            "<tr><td>6</td><td>precursor</td><td>&gt;25</td><td>&gt;25</td></tr>"
            "<tr><td>8</td><td><img src='r8.jpg'/></td><td>0.2</td><td>&gt;25</td></tr>"
            "<tr><td>9</td><td>Me</td><td>0.3</td><td>20</td></tr></table>")
    (found,) = find_rgroup_tables([table(3, 5, html)])
    assert (found.label_col, found.variables) == (0, {"R": 1})
    assert found.activity_cols == ["FIH IC50 μM", "KDM4A IC50 μM"]
    assert [row["R"] for row in found.rows] == ["precursor", "[image:r8.jpg]", "Me"]


def test_activity_header_read_with_ocr_errors_in_the_subscript():
    # MinerU reads the subscript 50 as "s0" or "5o" (417_6ilz prints IC50 and gets "ICs0").
    html = html_table([["Compd", "R", "ICs0 (nM)", "EC5o (µM)", "ICs", "ICU"], ["1", "H", "7", "3", "a", "b"]])
    (found,) = find_rgroup_tables([table(1, 0, html)])
    assert found.activity_cols == ["ICs0 (nM)", "EC5o (µM)"]


def _split_parts(second_html: str, caption: str | None = None) -> list[Block]:
    first = table(
        4,
        9,
        html_table([["Compd", "R1", "IC50 (nM)"], ["1a", "H", "5"], ["1b", "Me", "6"]]),
        bbox=[50, 500, 550, 760],
    )
    second = table(5, 0, second_html, bbox=[50, 40, 550, 200])
    blocks = [first, second]
    if caption:
        blocks.append(text(5, 1, [50, 20, 550, 38], caption, kind="caption"))
        blocks[-1] = Block(**{**blocks[-1].__dict__, "parent": 0})
    return blocks


def test_table_split_across_pages_with_repeated_header_is_merged():
    second = html_table([["Compd", "R1", "IC50 (nM)"], ["1c", "Et", "7"]])
    (found,) = find_rgroup_tables(_split_parts(second))
    assert found.table_id == "p4-t9"
    assert [found.label_of(r) for r in found.rows] == ["1a", "1b", "1c"]


def test_table_split_across_pages_without_header_is_merged():
    second = (
        "<table><tr><td>1c</td><td>Et</td><td>7</td></tr>"
        "<tr><td>1d</td><td>Pr</td><td>9</td></tr></table>"
    )
    (found,) = find_rgroup_tables(_split_parts(second))
    assert [found.label_of(r) for r in found.rows] == ["1a", "1b", "1c", "1d"]


def test_new_table_with_its_own_caption_is_not_merged():
    second = html_table([["Compd", "R1", "IC50 (nM)"], ["2a", "Et", "7"]])
    found = find_rgroup_tables(_split_parts(second, caption="Table 3. Second series."))
    assert [t.table_id for t in found] == ["p4-t9", "p5-t0"]


def test_continued_caption_still_merges():
    second = html_table([["Compd", "R1", "IC50 (nM)"], ["1c", "Et", "7"]])
    (found,) = find_rgroup_tables(_split_parts(second, caption="Table 2. Continued"))
    assert [found.label_of(r) for r in found.rows] == ["1a", "1b", "1c"]


def test_parse_reads_mineru_eq_tags():
    html = (
        "<table><tr><td>FIH <eq>\\mathsf { I C } _ { 5 0 } \\mu \\mathsf { M }</eq></td>"
        "<td><eq>0 . 2 1 0   \\pm   0 . 0 5 1</eq></td></tr></table>"
    )
    assert parse_html_table(html) == [["FIH IC50µM", "0.210±0.051"]]


def test_activity_header_glued_to_unit():
    html = html_table([["Compd", "R", "Target A IC50μM"], ["4", "H", "0.3"]])
    (found,) = find_rgroup_tables([table(1, 0, html)])
    assert found.activity_cols == ["Target A IC50μM"]


def test_label_sharing_a_cell_with_a_drawn_r_group():
    html = (
        "<table><tr><td rowspan='2'>R</td>"
        "<td colspan='2'><img src='images/h.jpg'/> Enzyme</td></tr>"
        "<tr><td>Site A IC50μM</td><td>Site B IC50μM</td></tr>"
        "<tr><td>6 precursor</td><td>&gt;25</td><td>&gt;25</td></tr>"
        "<tr><td><img src='images/r8.jpg'/> 8</td><td>0.25±0.08</td><td>0.09±0.04</td></tr>"
        "<tr><td>9 <img src='images/r9.jpg'/></td><td>0.21</td><td>0.09</td></tr>"
        "<tr><td>10</td><td>0.39</td><td>0.12</td></tr></table>"
    )
    (found,) = find_rgroup_tables([table(2, 7, html)])
    assert found.label_col == 0
    assert found.variables == {"R": 1}
    assert [found.label_of(r) for r in found.rows] == ["6", "8", "9", "10"]
    assert [r["R"] for r in found.rows] == [
        "precursor",
        "[image:images/r8.jpg]",
        "[image:images/r9.jpg]",
        "",
    ]
    assert found.activity_cols == ["Enzyme Site A IC50μM", "Enzyme Site B IC50μM"]


def test_a_table_printed_as_side_by_side_copies_is_read_as_one():
    html = (
        "<table><tr><td>Cmpd</td><td>R</td><td>X</td><td>IC50 [μM]</td><td>Cmpd</td><td></td><td>X</td>"
        "<td>IC50 [μM]</td></tr>"
        "<tr><td>1</td><td><img src='a.jpg'/></td><td>CH</td><td>424</td><td>9</td><td><img src='b.jpg'/></td>"
        "<td>N</td><td>19.6</td></tr>"
        "<tr><td>2</td><td>Me</td><td>N</td><td>22</td><td></td><td></td><td></td><td></td></tr></table>"
    )
    (found,) = find_rgroup_tables([table(2, 2, html)])
    assert found.variables == {"R": 1, "X": 2}
    assert [(found.label_of(r), r["R"], r["X"]) for r in found.rows] == [
        ("1", "[image:a.jpg]", "CH"), ("2", "Me", "N"), ("9", "[image:b.jpg]", "N"),
    ]
    assert found.activity_cols == ["IC50 [μM]"]


def test_repeated_activity_headers_without_a_second_label_column_are_not_copies():
    html = html_table([["Cmpd", "R", "IC50", "IC50"], ["1", "Me", "3", "4"], ["2", "Et", "5", "6"]])
    (found,) = find_rgroup_tables([table(1, 0, html)])
    assert len(found.rows) == 2 and len(found.headers) == 4
