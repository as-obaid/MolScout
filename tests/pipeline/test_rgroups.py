import shutil

import pytest

from molscout.pipeline.opsin import Opsin, find_jar
from molscout.pipeline.rgroups import ABBREVIATIONS, Fragment, to_fragment, to_fragments
from molscout.scoring.smiles import canonical_smiles


class FakeOpsin:
    """Answers like Opsin.to_smiles(radicals=True) for a few names and records each batch."""

    RADICALS = {
        "2-pyridyl": "*c1ccccn1",
        "morpholin-4-yl": "*N1CCOCC1",
        "4-fluorophenyl": "*c1ccc(F)cc1",
        "1,4-phenylene": "*c1ccc(*)cc1",
        "benzene": "c1ccccc1",
    }

    def __init__(self):
        self.batches = []

    def to_smiles(self, names, radicals=False):
        self.batches.append((list(names), radicals))
        table = self.RADICALS if radicals else {}
        return [table.get(name) for name in names]


def substituent(smiles, position=None):
    return Fragment(canonical_smiles(smiles), "substituent", position)


@pytest.mark.parametrize(
    ("text", "smiles"),
    [
        ("H", "[H]*"),
        ("F", "*F"),
        ("Cl", "*Cl"),
        ("Br", "*Br"),
        ("I", "*I"),
        ("Me", "*C"),
        ("Et", "*CC"),
        ("nPr", "*CCC"),
        ("n-Pr", "*CCC"),
        ("iPr", "*C(C)C"),
        ("i-Pr", "*C(C)C"),
        ("tBu", "*C(C)(C)C"),
        ("t-Bu", "*C(C)(C)C"),
        ("cPr", "*C1CC1"),
        ("c-Pr", "*C1CC1"),
        ("OMe", "*OC"),
        ("OEt", "*OCC"),
        ("OH", "*O"),
        ("NH2", "*N"),
        ("NHMe", "*NC"),
        ("NMe2", "*N(C)C"),
        ("CN", "*C#N"),
        ("NO2", "*[N+](=O)[O-]"),
        ("CF3", "*C(F)(F)F"),
        ("OCF3", "*OC(F)(F)F"),
        ("SO2Me", "*S(C)(=O)=O"),
        ("COOH", "*C(=O)O"),
        ("CO2H", "*C(=O)O"),
        ("COOMe", "*C(=O)OC"),
        ("CONH2", "*C(N)=O"),
        ("Ac", "*C(C)=O"),
        ("Ph", "*c1ccccc1"),
        ("Bn", "*Cc1ccccc1"),
        ("Boc", "*C(=O)OC(C)(C)C"),
    ],
)
def test_abbreviations(text, smiles):
    assert to_fragment(text, None) == substituent(smiles)


@pytest.mark.parametrize(
    ("text", "smiles"),
    [
        ("C2H5", "*CC"),
        ("i-C3H7", "*C(C)C"),
        ("n-C4H9", "*CCCC"),
        ("C6H5", "*c1ccccc1"),
        ("C6H11", "*C1CCCCC1"),
        ("N3", "*N=[N+]=[N-]"),
        ("n-Hex", "*CCCCCC"),
        ("Bu-t", "*C(C)(C)C"),
        ("Ad", "*C12CC3CC(CC(C3)C1)C2"),
        ("Mes", "*c1c(C)cc(C)cc1C"),
        ("TMS", "*[Si](C)(C)C"),
        ("Tf", "*S(=O)(=O)C(F)(F)F"),
        ("allyl", "*CC=C"),
        ("propargyl", "*CC#C"),
        ("3-Py", "*c1cccnc1"),
        ("2-Th", "*c1cccs1"),
        ("2-Naph", "*c1ccc2ccccc2c1"),
        ("Mor", "*N1CCOCC1"),
        ("CH2C6H5", "*Cc1ccccc1"),
        ("OC2H5", "*OCC"),
        ("c-hex", "*C1CCCCC1"),
        ("c-pr", "*C1CC1"),
    ],
)
def test_formula_groups_and_ring_abbreviations(text, smiles):
    assert to_fragment(text, None) == substituent(smiles)


@pytest.mark.parametrize("text", ["Py", "Th", "Tol", "An", "Pip"])
def test_abbreviations_that_do_not_say_where_they_attach_give_none(text):
    assert to_fragment(text, None) is None


def test_abbreviation_values_start_at_the_attachment_atom():
    for key, smiles in ABBREVIATIONS.items():
        assert smiles.startswith("*"), key
        assert Fragment(smiles, "substituent").kind == "substituent"


@pytest.mark.parametrize(
    ("text", "smiles"),
    [
        (" OMe ", "*OC"),
        ("-OMe", "*OC"),
        ("–OMe", "*OC"),
        ("Ph-", "*c1ccccc1"),
        ("CH₃", "*C"),
        ("CH$_3$", "*C"),
        ("(CF3)", "*C(F)(F)F"),
    ],
)
def test_cell_text_is_normalized(text, smiles):
    assert to_fragment(text, None) == substituent(smiles)


@pytest.mark.parametrize(
    ("text", "smiles"),
    [
        ("CH3", "*C"),
        ("CH2CH3", "*CC"),
        ("CH2OH", "*CO"),
        ("CH2Ph", "*Cc1ccccc1"),
        ("CH2CF3", "*CC(F)(F)F"),
        ("OCH2Ph", "*OCc1ccccc1"),
        ("(CH2)2OH", "*CCO"),
        ("CH2CH2NMe2", "*CCN(C)C"),
        ("COCH3", "*C(C)=O"),
        ("CH2OMe", "*COC"),
        ("NHCOMe", "*NC(C)=O"),
    ],
)
def test_condensed_formulas(text, smiles):
    assert to_fragment(text, None) == substituent(smiles)


@pytest.mark.parametrize(
    ("text", "smiles"),
    [
        ("4-F-Ph", "*c1ccc(F)cc1"),
        ("4-FC6H4", "*c1ccc(F)cc1"),
        ("4-F-C6H4", "*c1ccc(F)cc1"),
        ("p-F-Ph", "*c1ccc(F)cc1"),
        ("2-Me-Ph", "*c1ccccc1C"),
        ("3,4-diCl-Ph", "*c1ccc(Cl)c(Cl)c1"),
        ("3,5-Cl2-Ph", "*c1cc(Cl)cc(Cl)c1"),
        ("2,4-(OMe)2-Ph", "*c1ccc(OC)cc1OC"),
        ("3-Cl-4-F-Ph", "*c1ccc(F)c(Cl)c1"),
        ("4-MeO-Ph", "*c1ccc(OC)cc1"),
        ("4-CF3-C6H4", "*c1ccc(C(F)(F)F)cc1"),
        ("4-F-Bn", "*Cc1ccc(F)cc1"),
        ("4-Cl-phenyl", "*c1ccc(Cl)cc1"),
    ],
)
def test_substituted_phenyl_shorthand(text, smiles):
    assert to_fragment(text, None) == substituent(smiles)


@pytest.mark.parametrize("text", ["F-Ph", "7-F-Ph", "3,4-Cl-Ph", "2,2-diF-Ph", "4'-F-Ph", "2,6-diF"])
def test_phenyl_shorthand_that_cannot_be_placed_gives_none(text):
    assert to_fragment(text, None) is None


@pytest.mark.parametrize(
    ("text", "smiles", "position"),
    [
        ("(p)-OH", "*O", "para"),
        ("p-OH", "*O", "para"),
        ("o-Me", "*C", "ortho"),
        ("(m)-Cl", "*Cl", "meta"),
        ("4-OMe", "*OC", "4"),
        ("(3)-F", "*F", "3"),
        ("3'-CF3", "*C(F)(F)F", "3'"),
        ("4-CH2OH", "*CO", "4"),
    ],
)
def test_positional_groups_carry_a_position_hint(text, smiles, position):
    assert to_fragment(text, None) == substituent(smiles, position)


@pytest.mark.parametrize(
    ("text", "smiles"),
    [
        ("-CH2-", "*C*"),
        ("-O-", "*O*"),
        ("-S-", "*S*"),
        ("-NH-", "*N*"),
        ("-NHCO-", "*NC(=O)*"),
        ("-CONH-", "*C(=O)N*"),
        ("-SO2-", "*S(=O)(=O)*"),
        ("–CH2CH2–", "*CC*"),
        ("-OCH2-", "*OC*"),
        ("-CH2O-", "*CO*"),
        ("-N(Me)-", "*N(C)*"),
        ("-C(O)-", "*C(=O)*"),
        ("-(CH2)3-", "*CCC*"),
        ("CH2CH2", "*CC*"),
        ("NHCO", "*NC(=O)*"),
    ],
)
def test_bivalent_linkers_keep_written_direction(text, smiles):
    assert to_fragment(text, None) == Fragment(smiles, "linker")


@pytest.mark.parametrize("text", ["-OH-", "-Ph-", "-CH3-", "-xyz-"])
def test_linkers_with_a_terminal_group_inside_give_none(text):
    assert to_fragment(text, None) is None


@pytest.mark.parametrize(
    ("text", "smiles"),
    [
        ("N", "N"),
        ("CH", "[CH]"),
        ("C", "C"),
        ("NH", "[NH]"),
        ("O", "O"),
        ("S", "S"),
        ("CH2", "[CH2]"),
        ("CF", "CF"),
        ("CCl", "CCl"),
        ("C-OMe", "COC"),
        ("C(Me)", "CC"),
        ("CMe", "CC"),
        ("NMe", "NC"),
        ("SO2", "S(=O)(=O)"),
        ("C=O", "C(=O)"),
    ],
)
def test_ring_atom_and_single_atom_variables(text, smiles):
    assert to_fragment(text, None) == Fragment(smiles, "atom")


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("[*]c1ccncc1", Fragment(canonical_smiles("*c1ccncc1"), "substituent")),
        ("*C(=O)N", Fragment(canonical_smiles("*C(N)=O"), "substituent")),
        ("[*:1]CC[*:2]", Fragment("*CC*", "linker")),
    ],
)
def test_smiles_cell_text(text, fragment):
    assert to_fragment(text, None) == fragment


@pytest.mark.parametrize(
    "text",
    ["", "   ", "-", "—", "n.d.", "see text", "R1", "X", "Het", "bond", "c1ccncc1", "2-pyridyl", "12a", "=O"],
)
def test_unparseable_text_gives_none_without_opsin(text):
    assert to_fragment(text, None) is None


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("2-pyridyl", Fragment("*c1ccccn1", "substituent")),
        ("morpholin-4-yl", Fragment("*N1CCOCC1", "substituent")),
        ("4-fluorophenyl", Fragment("*c1ccc(F)cc1", "substituent")),
        ("1,4-phenylene", Fragment("*c1ccc(*)cc1", "linker")),
        ("benzene", None),
        ("unknownyl", None),
    ],
)
def test_names_go_through_opsin_as_radicals(text, fragment):
    fake = FakeOpsin()
    assert to_fragment(text, fake) == fragment
    assert fake.batches == [([text], True)]


def test_abbreviations_do_not_call_opsin():
    fake = FakeOpsin()
    assert to_fragment("OMe", fake) == substituent("*OC")
    assert fake.batches == []


def test_to_fragments_sends_all_names_in_one_batch():
    fake = FakeOpsin()
    texts = ["2-pyridyl", "Me", "morpholin-4-yl", "4-F-Ph", "benzene"]
    got = to_fragments(texts, fake)
    assert got == [to_fragment(text, FakeOpsin()) for text in texts]
    assert fake.batches[0] == (["2-pyridyl", "morpholin-4-yl", "benzene"], True)


@pytest.mark.parametrize(
    ("smiles", "kind"),
    [("*C", "linker"), ("*C*", "substituent"), ("C", "substituent"), ("*C", "atom"), ("*C", "group"), ("not smiles", "atom")],
)
def test_fragment_rejects_wrong_kind_or_dummy_count(smiles, kind):
    with pytest.raises(ValueError):
        Fragment(smiles, kind)


REAL_JAR = find_jar()
needs_opsin = pytest.mark.skipif(
    shutil.which("java") is None or REAL_JAR is None or not REAL_JAR.is_file(),
    reason="needs Java and the OPSIN jar (scripts/fetch_opsin.py)",
)


@needs_opsin
def test_heteroaryl_names_with_real_opsin():
    texts = ["2-pyridyl", "morpholin-4-yl", "4-fluorophenyl", "thiophen-2-yl", "4-methylpiperazin-1-yl", "cyclohexyl"]
    got = to_fragments(texts, Opsin(REAL_JAR))
    expected = ["*c1ccccn1", "*N1CCOCC1", "*c1ccc(F)cc1", "*c1cccs1", "*N1CCN(C)CC1", "*C1CCCCC1"]
    assert got == [Fragment(canonical_smiles(smiles), "substituent") for smiles in expected]
