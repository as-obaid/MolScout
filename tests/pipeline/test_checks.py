import pytest

from molscout.pipeline.checks import formula_status, junk_reason, printed_formulas, strip_salts
from molscout.pipeline.schema import LayoutBlock
from molscout.scoring.smiles import canonical_smiles

# --- junk_reason --------------------------------------------------------------------------


@pytest.mark.parametrize("smiles", ["CCOc1ccccc1", "CCCCO", "C" * 39, "C1" + "C" * 38 + "C1", "C" * 20 + "O" + "C" * 20])
def test_ordinary_molecules_are_not_junk(smiles):
    assert junk_reason(smiles) is None


@pytest.mark.parametrize(
    ("smiles", "reason"),
    [
        ("", "invalid"),
        ("C1CC", "invalid"),
        ("not smiles", "invalid"),
        ("[*:1]c1ccccc1", "leftover dummy"),
        ("*c1ccccc1", "leftover dummy"),
        ("[R1]c1ccccc1", "leftover dummy"),
        ("c1ccccc1[R]", "leftover dummy"),
        ("CCCC", "too few heavy atoms"),
        ("CC(=O)O", "too few heavy atoms"),
        ("C" * 40, "long carbon chain"),
        ("c1ccccc1" + "C" * 40, "long carbon chain"),
        ("c1ccccc1[CH2]", "radical"),
        ("[O]c1ccccc1", "radical"),
        ("c1ccccc1O.c1ccncc1", "multiple fragments"),
    ],
)
def test_junk_classes(smiles, reason):
    assert junk_reason(smiles) == reason


@pytest.mark.parametrize(
    ("smiles", "parent"),
    [
        ("CN1CCC(CC1)c1ccccc1.Cl", "CN1CCC(CC1)c1ccccc1"),
        ("CN1CCC(CC1)c1ccccc1.Cl.Cl", "CN1CCC(CC1)c1ccccc1"),
        ("CC[NH+](C)Cc1ccccc1.[Cl-]", "CC[NH+](C)Cc1ccccc1"),
        ("CCc1ccccc1C(=O)[O-].[Na+]", "CCc1ccccc1C(=O)[O-]"),
        ("OC(=O)C(F)(F)F.NCCc1ccccc1", "NCCc1ccccc1"),
        ("NCCc1ccccc1.O.O", "NCCc1ccccc1"),
        ("NCCc1ccccc1.OC(=O)/C=C\\C(=O)O", "NCCc1ccccc1"),
        ("NCCc1ccccc1.CS(=O)(=O)O", "NCCc1ccccc1"),
        ("NCCc1ccccc1.CS(C)=O", "NCCc1ccccc1"),
    ],
)
def test_salts_and_solvents_are_stripped(smiles, parent):
    assert junk_reason(smiles) is None
    assert strip_salts(smiles) == canonical_smiles(parent)


def test_strip_salts_keeps_a_single_fragment():
    assert strip_salts("OCC") == "CCO"


def test_strip_salts_refuses_two_real_fragments_and_invalid_input():
    assert strip_salts("c1ccccc1O.c1ccncc1") is None
    assert strip_salts("C1CC") is None


def test_salt_alone_keeps_the_largest_and_is_too_small():
    assert strip_salts("Cl.OC(=O)C(F)(F)F") == canonical_smiles("OC(=O)C(F)(F)F")
    assert junk_reason("Cl.Cl") == "too few heavy atoms"


# --- printed_formulas -------------------------------------------------------------------------


def text(*paragraphs, kinds=None):
    kinds = kinds or ["text"] * len(paragraphs)
    return [LayoutBlock(page=5, index=i, type=kind, bbox=[0, 0, 1, 1], text=p) for i, (p, kind) in enumerate(zip(paragraphs, kinds))]


@pytest.mark.parametrize(
    ("paragraph", "expected"),
    [
        # Ion formula after "calcd for", adduct after it.
        ("N-(4-Fluorophenyl)benzamide (12a). White solid. HRMS (ESI) calcd for C13H11FNO [M + H]+ 216.0819, found 216.0821.", {"12a": "C13H10FNO"}),
        # ACS order: adduct first.
        ("Compound 7: HRMS (ESI) m/z: [M + H]+ Calcd for C9H11N2O 163.0866; Found 163.0868.", {"7": "C9H10N2O"}),
        # Neutral formula printed next to the adduct: the m/z shows it is already neutral.
        ("Amide (3b). HRMS calcd for C13H10FNO [M+H]+: 216.0819; found: 216.0822.", {"3b": "C13H10FNO"}),
        ("Ester (5). HRMS (ESI) calcd for C10H12NaO2 [M+Na]+ 187.0730, found 187.0733.", {"5": "C10H12O2"}),
        ("Acid (6c). HRMS (ESI) calcd for C7H5O2 [M−H]− 121.0295, found 121.0290.", {"6c": "C7H6O2"}),
        ("Acid (6d). HRMS (ESI) calcd for C7H5O2 (M-H)- 121.0295.", {"6d": "C7H6O2"}),
        ("Base (9). HRMS calcd for C$_{19}$H$_{21}$N$_{6}$ [M+H]$^{+}$ 333.1822, found 333.1820.", {"9": "C19H20N6"}),
        ("Base (9b). HRMS calcd for C₁₉H₂₁N₆ [M+H]⁺ 333.1822.", {"9b": "C19H20N6"}),
        ("Base (9c). HRMS calcd for C19H21N6 [M+H]+.", {"9c": "C19H20N6"}),
        ("Ester (10). HRMS (EI) calcd for C8H8O2 136.0524, found 136.0520.", {"10": "C8H8O2"}),
        ("Salt (11). Anal. Calcd for C19H20N6·0.5H2O: C, 66.80; H, 6.20; N, 24.60.", {"11": "C19H20N6"}),
        ("Amide (13). HRMS calcd. for C13H11FNO+ [M+H]+ 216.0819.", {"13": "C13H10FNO"}),
    ],
)
def test_printed_formulas(paragraph, expected):
    assert printed_formulas(text(paragraph)) == expected


def test_blocks_without_text_are_skipped():
    # MinerU writes image and table blocks with text null; they must not stop the formula check.
    empty = [
        LayoutBlock(page=4, index=0, type="title", bbox=[0, 0, 1, 1], text=None),
        LayoutBlock(page=4, index=1, type="image", bbox=[0, 0, 1, 1], text=None),
    ]
    blocks = empty + text("Amide (13). HRMS calcd. for C13H11FNO+ [M+H]+ 216.0819.")
    assert printed_formulas(blocks) == {"13": "C13H10FNO"}


def test_formula_without_a_label_is_skipped():
    assert printed_formulas(text("HRMS (ESI) calcd for C13H11FNO [M+H]+ 216.0819.")) == {}


def test_label_from_the_heading_block_before():
    blocks = text("4.1.2. 2-Phenylbenzamide (14)", "White solid. HRMS calcd for C13H12NO [M+H]+ 198.0913.", kinds=["title", "text"])
    assert printed_formulas(blocks) == {"14": "C13H11NO"}


def test_label_is_not_taken_from_an_earlier_text_block():
    blocks = text("Benzamide (15).", "White solid. HRMS calcd for C13H12NO [M+H]+ 198.0913.")
    assert printed_formulas(blocks) == {}


def test_two_compounds_in_one_block_each_take_the_nearest_label():
    paragraph = (
        "Ester (16a): HRMS calcd for C8H9O2 [M+H]+ 137.0597. "
        "Ester (16b): HRMS calcd for C9H11O2 [M+H]+ 151.0754."
    )
    assert printed_formulas(text(paragraph)) == {"16a": "C8H8O2", "16b": "C9H10O2"}


def test_conflicting_formulas_drop_the_label():
    blocks = text("Amide (17). HRMS calcd for C13H11FNO [M+H]+ 216.0819.", "Amide (17). Anal. Calcd for C13H12FNO: C, 70.0.")
    assert printed_formulas(blocks) == {}


def test_same_formula_twice_is_kept():
    blocks = text("Amide (18). HRMS calcd for C13H11FNO [M+H]+ 216.0819.", "Amide (18). Anal. Calcd for C13H10FNO: C, 72.5.")
    assert printed_formulas(blocks) == {"18": "C13H10FNO"}


def test_mass_that_fits_neither_reading_is_skipped():
    assert printed_formulas(text("Amide (19). HRMS calcd for C13H11FNO [M+H]+ 250.1234.")) == {}


def test_charged_formula_without_an_adduct_is_skipped():
    assert printed_formulas(text("Amide (20). HRMS calcd for C13H11FNO+ 216.0819.")) == {}


# --- formula_status -------------------------------------------------------------------------


def test_formula_match():
    assert formula_status("O=C(Nc1ccc(F)cc1)c1ccccc1", "C13H10FNO") == "match"


def test_formula_mismatch():
    assert formula_status("O=C(Nc1ccc(Cl)cc1)c1ccccc1", "C13H10FNO") == "mismatch"


def test_formula_ignores_charge_and_order():
    assert formula_status("C[N+](C)(C)Cc1ccccc1", "C10H16N") == "match"
    assert formula_status("CCO", "OH6C2") == "match"


def test_formula_matches_salt_or_parent():
    assert formula_status("NCCc1ccccc1.Cl", "C8H11N") == "match"
    assert formula_status("NCCc1ccccc1.Cl", "C8H12ClN") == "match"


@pytest.mark.parametrize(("smiles", "formula"), [("CCO", None), ("CCO", ""), ("C1CC", "C2H6O"), ("CCO", "C2H6Xx"), ("CCO", "c2h6o")])
def test_formula_status_none_when_it_cannot_compare(smiles, formula):
    assert formula_status(smiles, formula) == "none"
