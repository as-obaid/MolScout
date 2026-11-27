"""Named compounds from experimental sections: (label, systematic name) pairs."""

from __future__ import annotations

import pytest

from molscout.pipeline.names import looks_systematic, named_compounds

from .fixtures.a2.fakes import text


def _named(*bodies: str) -> list[tuple[str, str]]:
    return named_compounds(
        [text(5, i, [50, 100 + 40 * i, 550, 130 + 40 * i], b) for i, b in enumerate(bodies)]
    )


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            "N-(4-Fluorophenyl)-1H-indole-2-carboxamide (12a). To a solution of the acid in DMF",
            ("12a", "N-(4-Fluorophenyl)-1H-indole-2-carboxamide"),
        ),
        (
            "12b: 5-Chloro-N-methylpyridin-2-amine. Yield 60%, white solid.",
            ("12b", "5-Chloro-N-methylpyridin-2-amine"),
        ),
        (
            "Compound 12c. 4-(Piperidin-1-yl)benzoic acid. White solid (210 mg, 70%).",
            ("12c", "4-(Piperidin-1-yl)benzoic acid"),
        ),
        (
            "tert-Butyl 4-(4-aminophenyl)piperazine-1-carboxylate (5). The amine was dissolved.",
            ("5", "tert-Butyl 4-(4-aminophenyl)piperazine-1-carboxylate"),
        ),
        (
            "**(S)-3-(4-Chlorophenyl)-2-methylpropanoic acid (9)**. Prepared by procedure A.",
            ("9", "(S)-3-(4-Chlorophenyl)-2-methylpropanoic acid"),
        ),
        (
            "Ethyl 2-(1H-pyrazol-4-yl)acetate hydrochloride (S3). Colourless oil.",
            ("S3", "Ethyl 2-(1H-pyrazol-4-yl)acetate hydrochloride"),
        ),
        (
            "6-Bromo-$N^{2}$-cyclopropylquinazoline-2,4-diamine (14). Yellow solid.",
            ("14", "6-Bromo-N2-cyclopropylquinazoline-2,4-diamine"),
        ),
    ],
)
def test_named_compound_patterns(body, expected):
    assert _named(body) == [expected]


@pytest.mark.parametrize(
    "body",
    [
        "The title compound (12a) was obtained as a white solid.",
        "the crude product (3) was purified by column chromatography.",
        "Compound 12a: yield 70%, mp 120 °C.",
        "General procedure for N-aryl amides (12a–l). A mixture of the acid and amine",
        "The solution was stirred for 2 h (4). Water was added.",
        "Compound 12a. The title compound was prepared by general procedure B.",
        "",
    ],
)
def test_non_names_are_rejected(body):
    assert _named(body) == []


AFFORD = "The ester was hydrolysed to afford 2-methyl-N-phenylbenzamide (7) as a white solid."


def test_mid_sentence_name_is_read_in_the_experimental_section():
    assert _named("Experimental Section", AFFORD) == [("7", "2-methyl-N-phenylbenzamide")]
    assert _named("4.2. Chemistry", AFFORD) == [("7", "2-methyl-N-phenylbenzamide")]


def test_heading_that_opens_the_section_keeps_its_own_name():
    heading = "Synthesis of 4-methoxybenzoic acid (3)"
    assert _named(heading, AFFORD) == [
        ("3", "4-methoxybenzoic acid"),
        ("7", "2-methyl-N-phenylbenzamide"),
    ]


def test_mid_sentence_name_outside_the_experimental_section_is_ignored():
    caption_like = "Representative 4-hydroxypyrimidine (8) and spiro[4.5]decanone (18) are shown."
    assert _named(AFFORD) == []
    assert _named("Experimental Section", "References", AFFORD) == []
    assert _named(caption_like) == []


def test_several_names_in_one_block_and_first_name_wins():
    body = (
        "2-Methyl-N-phenylbenzamide (7). White solid. "
        "N-Benzyl-2-methylbenzamide (8). Colourless oil. "
        "N-Methylbenzamide (7). Duplicate label."
    )
    assert _named(body) == [
        ("7", "2-Methyl-N-phenylbenzamide"),
        ("8", "N-Benzyl-2-methylbenzamide"),
    ]


def test_names_across_blocks_keep_reading_order():
    assert _named("4-Methoxybenzoic acid (3).", "1-(4-Chlorophenyl)piperazine (4).") == [
        ("3", "4-Methoxybenzoic acid"),
        ("4", "1-(4-Chlorophenyl)piperazine"),
    ]


def test_tables_and_images_are_ignored():
    blocks = [text(5, 0, [0, 0, 1, 1], "4-Methoxybenzoic acid (3).", kind="table")]
    assert named_compounds(blocks) == []


@pytest.mark.parametrize(
    ("name", "ok"),
    [
        ("N-(4-fluorophenyl)-1H-indole-2-carboxamide", True),
        ("4-(Piperidin-1-yl)benzoic acid", True),
        ("2,4-dichloropyrimidine", True),
        ("benzamide", True),
        ("The title compound", False),
        ("the acid", False),
        ("compound", False),
        ("stirred for 2 h", False),
        ("CCT400028", False),
        ("N-aryl", False),
        ("x" * 400, False),
    ],
)
def test_looks_systematic(name, ok):
    assert looks_systematic(name) is ok
