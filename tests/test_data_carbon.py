"""MolRecBench-Wild CARBON graphs become reference SMILES the way the official evaluator builds them."""

import pytest

from molscout.data.carbon import (
    carbon_reference,
    exclusion_reasons,
    expand_abbreviations,
    label_atoms,
    load_templates,
)
from molscout.scoring import canonical_smiles, stereo_stripped_smiles


def record(symbols, bonds, coords=None, **fields):
    n = len(symbols)
    base = {
        "symbols": symbols,
        "bonds": bonds,
        "coords": coords or [[float(i), 0.0] for i in range(n)],
        "charges": [None] * n,
        "radicals": [None] * n,
        "valences": [None] * n,
        "isotopes": [None] * n,
        "attach_points": [None] * n,
        "brackets": [],
    }
    return {**base, **fields}


def smiles_of(rec):
    reference = carbon_reference("x", rec)
    assert reference.error is None, reference.error
    return reference.smiles


def test_plain_graph():
    assert smiles_of(record(["C", "C", "O"], [[0, 1, 1], [1, 2, 1]])) == "CCO"


def test_charges_and_double_bonds():
    rec = record(["C", "N", "O", "O"], [[0, 1, 1], [1, 2, 2], [1, 3, 1]], charges=[None, 1, None, -1])
    assert smiles_of(rec) == canonical_smiles("C[N+](=O)[O-]")


def test_aromatic_bonds():
    ring = [[i, (i + 1) % 6, 4] for i in range(6)]
    assert smiles_of(record(["C"] * 6, ring)) == "c1ccccc1"


def test_wedges_set_tetrahedral_stereo():
    coords = [[0.0, 0.0], [0.0, 1.0], [-0.87, -0.5], [0.87, -0.5], [0.3, -0.2]]
    symbols = ["C", "F", "Cl", "Br", "O"]
    up = smiles_of(record(symbols, [[0, 1, 1], [0, 2, 1], [0, 3, 1], [0, 4, 5]], coords))
    down = smiles_of(record(symbols, [[0, 1, 1], [0, 2, 1], [0, 3, 1], [0, 4, 6]], coords))
    assert "@" in up and "@" in down and up != down
    assert stereo_stripped_smiles(up) == stereo_stripped_smiles(down)


def test_expands_methyl_template():
    assert "Me" in load_templates()
    assert smiles_of(record(["O", "[Me]"], [[0, 1, 1]])) == "CO"


def test_expands_a_label_between_two_atoms():
    # Hex has a template with two attachment points; the official evaluator gives CCCCCCCO.
    assert smiles_of(record(["C", "[Hex]", "O"], [[0, 1, 1], [1, 2, 1]])) == "CCCCCCCO"


def test_label_template_is_chosen_by_distinct_bond_types():
    # Two single bonds give key "1", as one does; CO2's charged template then cannot sanitize,
    # and the official evaluator drops the graph too.
    reference = carbon_reference("x", record(["C", "[CO2]", "C"], [[0, 1, 1], [1, 2, 1]]))
    assert reference.smiles is None and "valence" in reference.error


def test_numbered_r_group_keeps_its_number():
    assert smiles_of(record(["C", "C", "[R1]"], [[0, 1, 1], [1, 2, 1]])) == canonical_smiles("CC[1*]")


@pytest.mark.parametrize("label", ["[X]", "[Z]", "[L]", "[M]", "[E]", "[R']", "[Ar]", "[Alkyl]"])
def test_variable_labels_become_bare_star(label):
    assert smiles_of(record(["C", "C", label], [[0, 1, 1], [1, 2, 1]])) == canonical_smiles("CC*")


@pytest.mark.parametrize("label", ["Qwerty", "R", "R3", "X1"])
def test_label_with_no_template_excludes_with_reason(label):
    # The official track leaves these out: no template, and not on its list of labels to keep.
    reference = carbon_reference("x", record(["C", f"[{label}]"], [[0, 1, 1]]))
    assert reference.smiles is None and label in reference.error


def test_counter_ion_label_without_attachment_template_excludes():
    # The official table's OTf- template for no bonds has no attachment point, so the label stays.
    reference = carbon_reference("x", record(["C", "N", "[OTf]"], [[0, 1, 1]], charges=[None, 1, -1]))
    assert reference.smiles is None and "OTf-" in reference.error


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"symbols": ["C", "[Rα]"]}, "Greek"),
        ({"symbols": ["C", "[?]"]}, "question mark"),
        ({"bonds": [[0, 1, 13]]}, "bond type"),
        ({"attach_points": [1, None]}, "attachment"),
        ({"brackets": [{"alias": "n", "atoms": [0], "display_rects": []}]}, "brackets"),
    ],
)
def test_graphs_smiles_cannot_express_are_excluded(change, reason):
    rec = {**record(["C", "C"], [[0, 1, 1]]), **change}
    assert any(reason in r for r in exclusion_reasons(rec))
    reference = carbon_reference("x", rec)
    assert reference.smiles is None and reason in reference.error


def test_greek_letter_excludes():
    assert carbon_reference("x", record(["C", "[Rβ]"], [[0, 1, 1]])).smiles is None


def test_plain_graph_has_no_exclusion_reasons():
    assert exclusion_reasons(record(["C", "C"], [[0, 1, 1]])) == []


def test_unsanitizable_graph_gives_rdkit_reason():
    rec = record(["C"] * 6, [[0, i, 1] for i in range(1, 6)])
    reference = carbon_reference("x", rec)
    assert reference.smiles is None and "valence" in reference.error.lower()


def test_label_atoms_leaves_real_atoms_alone():
    assert label_atoms("C[NH3+].[Cl-]") == ("C[NH3+].[Cl-]", [])
    assert label_atoms("C[R2]C[Alkyl]") == ("C[2*]C*", [])
    assert label_atoms("C[OTf-]") == ("C[OTf-]", ["OTf-"])


def test_expand_keeps_listed_labels_as_tokens():
    expanded, missing = expand_abbreviations("CC[Z]", load_templates())
    assert "[Z]" in expanded and missing == []


def test_expand_reports_labels_without_a_template():
    assert expand_abbreviations("CC[Qwerty]", load_templates())[1] == ["Qwerty"]
