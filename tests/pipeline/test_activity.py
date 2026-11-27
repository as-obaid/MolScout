"""Activity: does the paper report activity, and which labels were tested."""

from __future__ import annotations

import pytest

from molscout.pipeline import activity
from molscout.pipeline.activity import activity_sentences, reports_activity
from molscout.pipeline.tables import find_rgroup_tables

from .fixtures.a2.fakes import html_table, table, text


def _para(body: str, index: int = 0):
    return text(2, index, [50, 100, 550, 160], body)


@pytest.mark.parametrize(
    "body",
    [
        "Compound 12a showed an IC50 of 14 nM against EGFR.",
        "The most potent analogue had a Ki = 0.3 µM.",
        "Inhibition reached 85% at 10 µM.",
        "The MIC was 2 µg/mL for S. aureus.",
        "$\\mathrm{IC}_{50}$ = 14 nM was measured for the lead.",
        "pIC50 values ranged from 6.1 to 8.3.",
    ],
)
def test_reports_activity_from_text_with_values(body):
    assert reports_activity([_para(body)])


@pytest.mark.parametrize(
    "body",
    [
        "IC50 values were determined as described previously.",
        "The compounds were synthesised in 4 steps with 60% overall yield.",
        "",
    ],
)
def test_no_activity_without_values(body):
    assert not reports_activity([_para(body)])


def test_reports_activity_from_table_column():
    html = html_table([["Compd", "IC50 (nM)"], ["1", "5"]])
    assert reports_activity([table(1, 0, html)])


def test_table_without_activity_column_does_not_count():
    html = html_table([["Compd", "R", "Yield (%)"], ["1", "H", "50"]])
    assert not reports_activity([table(1, 0, html)])


def test_tested_labels_from_rgroup_table_skip_untested_rows():
    html = html_table(
        [
            ["Compd", "R1", "IC50 (nM)", "Ki (nM)"],
            ["12a", "H", "14", ""],
            ["12b", "F", "&gt;10000", "NT"],
            ["12c", "Cl", "NT", "n.t."],
            ["12d", "Br", "", ""],
            ["12e", "I", "ND", ""],
        ]
    )
    blocks = [table(1, 0, html)]
    assert activity.tested_labels(blocks, find_rgroup_tables(blocks)) == {"12a", "12b", "12e"}


def test_tested_labels_from_activity_only_table():
    html = html_table([["Cmpd", "EC50 (µM)"], ["3–5", "0.2"], ["7", "1.1"], ["8", "–"]])
    assert activity.tested_labels([table(1, 0, html)], []) == {"3", "4", "5", "7"}


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("12a (IC50 = 14 nM) was the most potent analogue.", {"12a"}),
        ("Compound 7 inhibited EGFR with a Ki of 3 nM.", {"7"}),
        ("Compounds 12a–c displayed IC50 values below 1 µM.", {"12a", "12b", "12c"}),
        ("The bromo analogue **9b** gave 92% inhibition at 1 µM.", {"9b"}),
        ("We prepared 20 analogues and measured IC50 values of 5 to 900 nM.", set()),
        ("The reference inhibitor showed IC50 = 5 nM (ref 23).", set()),
        ("Compound 3 was prepared from 2 in two steps.", set()),
        ("Replacing the methyl group (14, IC50 = 40 nM) lowered potency.", {"14"}),
        ("Compound 7 inhibited EGFR with an IC50 of 9 nM. Compound 8 was not soluble.", {"7"}),
        ("8– 10 were tested for inhibition of FIH (IC50 > 25 µM).", {"8", "9", "10"}),
        ("2 equivalents were used and inhibition was 50%.", set()),
    ],
)
def test_tested_labels_from_sentences(body, expected):
    assert activity.tested_labels([_para(body)], []) == expected


def test_tested_labels_join_tables_and_text():
    html = html_table([["Compd", "R", "IC50 (nM)"], ["1", "H", "5"]])
    blocks = [table(1, 0, html), _para("Compound 4 had a Kd of 12 nM.", index=1)]
    assert activity.tested_labels(blocks, find_rgroup_tables(blocks)) == {"1", "4"}


def test_tables_argument_counts_even_without_table_blocks():
    html = html_table([["Compd", "R", "IC50 (nM)"], ["1", "H", "5"]])
    tables = find_rgroup_tables([table(1, 0, html)])
    assert activity.tested_labels([], tables) == {"1"}


def test_activity_sentences_split_text():
    body = "Compound 7 had an IC50 of 9 nM. It was selective. Analogue 8 showed Ki = 2 nM."
    assert activity_sentences(body) == [
        "Compound 7 had an IC50 of 9 nM.",
        "Analogue 8 showed Ki = 2 nM.",
    ]


def test_tested_labels_from_label_and_drawing_cells():
    html = (
        "<table><tr><td>R</td><td>Site A IC50μM</td></tr>"
        "<tr><td><img src='images/r8.jpg'/> 8</td><td>0.25</td></tr>"
        "<tr><td>9 <img src='images/r9.jpg'/></td><td>NT</td></tr></table>"
    )
    blocks = [table(1, 0, html)]
    assert activity.tested_labels(blocks, find_rgroup_tables(blocks)) == {"8"}
