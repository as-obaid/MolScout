import shutil
from pathlib import Path

import pytest
from rdkit import Chem

from molscout.data import biovista_truth as bt
from molscout.scoring import canonical_smiles

FIXTURE = Path(__file__).parent / "fixtures" / "biovista"
MANIFEST = FIXTURE / "manifest.csv"

# 221_6oac ligand 9: its label is the naive paste of the groups over the backbone, and group ring digit 1 collides
# with the triazine's open ring 1, so the label is another molecule than the RDKit molzip join of the parts.
PASTE_KEY = "Clc1nc(N1[C@@H](C)COCC1)nc(N1[C@@H](C)COCC1)n1"
PASTE_BACKBONE = "Clc1nc([*:1])nc([*:2])n1"
PASTE_GROUPS = "[*:1]N1[C@@H](C)COCC1;[*:2]N1[C@@H](C)COCC1"
JOIN = "C[C@H]1COCCN1c1nc(Cl)nc(N2CCOC[C@@H]2C)n1"
# 221_6oac ligand 8: the label uses ring digit 2, so it is the join already.
CONSISTENT = ("Clc1nc(N2CCOCC2)nc(N2CCOCC2)n1", PASTE_BACKBONE, "[*:1]N1CCOCC1;[*:2]N1CCOCC1")
# 221_6oac ligand 33: the backbone text reuses ring digit 1 itself, so the join is garbled; the label is right and
# is not the naive paste.
BROKEN_BACKBONE = (
    "NC1=NC=C(C2=NC(=NC(=N2)N2CCOCC2)N2C3CCC2COC3)C(=C1)C(F)(F)F",
    "FC(F)(F)c1cc(N)ncc1c1nc(N1CCOCC1)nc([*:1])n1",
    "[*:1]N1C2COCC1CC2",
)
# The paste is the label, but the group alone does not parse: its leading F would take two bonds.
UNPARSABLE_GROUP = ("FC(F)(F)Cc1ccccc1", "[*:1]c1ccccc1", "[*:1]FC(F)(F)C")
# The three patterns the rule must leave, each a label that is the naive paste and differs from the join:
# 26_6st3 writes groups attachment-last (`[*:1]c1ccccc1C` is benzyl when pasted at the backbone's start), so the
# label is likely right and the join (o-tolyl) wrong;
ATTACHMENT_LAST = ("c1ccccc1CNC(=O)c1cnc(N2N=CC=C2)nc1(O)", "[*:1]NC(=O)c1cnc(N2N=CC=C2)nc1(O)", "[*:1]c1ccccc1C")
# 498_6h52 16d pastes `[*:1]OC` at the backbone's start, which bonds through its last-written atom (CH2OH, not OMe);
LEADING_PASTE = ("OCc1ccccc1", "[*:1]c1ccccc1", "[*:1]OC")
# 221_6oac ligands 24 to 30: the backbone itself closes ring digit 1 into a c-n-c-N 4-ring, so label and join are
# both garbled.
SELF_GARBLED = ("Clc1nc(N1CCOCC1)nc(N1[C@@H](C)COCC1)n1", "Clc1nc(N1CCOCC1)nc([*:1])n1", "[*:1]N1[C@@H](C)COCC1")
# One paper, 7_gggg, with every kind of row: CSV lines 2 to 6, and only line 2 is repaired.
LABEL_ROWS = (
    (PASTE_KEY, "9", PASTE_BACKBONE, PASTE_GROUPS),
    (CONSISTENT[0], "8", *CONSISTENT[1:]),
    (BROKEN_BACKBONE[0], "33", *BROKEN_BACKBONE[1:]),
    (UNPARSABLE_GROUP[0], "40", *UNPARSABLE_GROUP[1:]),
    ("c1ccncc1", "41", "NA", "NA"),
)


def labelled_paper(directory, rows=LABEL_ROWS, *, header="smiles,ligand,backbone,groups"):
    """A BioVista root holding one scored paper, 7_gggg, labelled with `rows`; returns (root, manifest)."""
    labels = directory / bt.LABELS
    labels.mkdir(parents=True)
    (labels / "7_gggg_structure.csv").write_text("\n".join([header, *(",".join(row) for row in rows)]) + "\n")
    manifest = directory / "manifest.csv"
    manifest.write_text(
        "paper_id,pdb_id,structures,status,oa_version,sha256\n"
        f"7_gggg,gggg,{len(rows)},ok,publishedVersion,{'a' * 64}\n"
    )
    return directory, manifest


@pytest.fixture
def root(tmp_path):
    shutil.copytree(FIXTURE, tmp_path / "biovista")
    return tmp_path / "biovista"


@pytest.fixture(scope="module")
def truth():
    return bt.load_biovista_truth(FIXTURE, MANIFEST)


def test_scored_papers_are_ok_with_structures_in_index_order():
    papers = bt.scored_papers(MANIFEST)
    assert [p.paper_id for p in papers] == ["1_aaaa", "2_bbbb", "3_cccc", "6_ffff"]
    assert papers[0] == bt.ScoredPaper("1_aaaa", "aaaa", "publishedVersion", papers[0].sha256, 3)


def test_scored_papers_reject_bad_manifests(tmp_path):
    bad = tmp_path / "m.csv"
    bad.write_text("paper_id,pdb_id\n1_aaaa,aaaa\n")
    with pytest.raises(ValueError, match="status"):
        bt.scored_papers(bad)
    text = MANIFEST.read_text().replace(",test,3,", ",test,four,", 1)
    bad.write_text(text)
    with pytest.raises(ValueError, match="1_aaaa"):
        bt.scored_papers(bad)


def test_references_split_into_drawn_and_enumerated(truth):
    assert dict(truth.references) == {"1_aaaa": ("CCO", "c1ccccc1", "CCCCl"), "2_bbbb": ("CCN",), "3_cccc": ("CCCl",)}
    assert dict(truth.drawn) == {"1_aaaa": ("CCO", "c1ccccc1"), "2_bbbb": ("CCN",)}
    assert dict(truth.enumerated) == {"1_aaaa": ("CCCCl",), "2_bbbb": (), "3_cccc": ("CCCl",)}
    assert truth.labels == 5  # the rows of the kept papers


def test_unreadable_rows_name_the_csv_line_and_reason(truth):
    assert dict(truth.unreadable) == {
        "6_ffff:2": "NA: the label has no SMILES",
        "6_ffff:3": "RDKit cannot parse it; text after whitespace is part of the SMILES",
    }


def test_papers_with_an_unreadable_label_are_dropped(truth):
    assert dict(truth.dropped) == {"6_ffff": "has an unreadable label"}
    assert [p.paper_id for p in truth.papers] == ["1_aaaa", "2_bbbb", "3_cccc"]
    assert all("6_ffff" not in view for view in (truth.references, truth.drawn, truth.enumerated))


def test_benchmark_papers_are_the_scored_papers_without_an_unreadable_label(truth):
    assert bt.benchmark_papers(FIXTURE, MANIFEST) == truth.papers


def test_one_unreadable_label_drops_a_paper_that_has_readable_ones(root):
    path = root / bt.LABELS / "1_aaaa_structure.csv"
    path.write_text(path.read_text().replace("c1ccccc1,", "C1CC,"))
    truth = bt.load_biovista_truth(root, MANIFEST)
    assert list(truth.dropped) == ["1_aaaa", "6_ffff"]
    assert truth.unreadable["1_aaaa:3"] == "RDKit cannot parse it"
    assert list(truth.references) == ["2_bbbb", "3_cccc"]
    assert truth.labels == 2
    assert truth.groups() == {"all": frozenset({"2_bbbb", "3_cccc"}), "without_submitted": frozenset({"3_cccc"})}
    assert [p.paper_id for p in bt.benchmark_papers(root, MANIFEST)] == ["2_bbbb", "3_cccc"]


def test_groups(truth):
    assert truth.groups() == {
        "all": frozenset({"1_aaaa", "2_bbbb", "3_cccc"}),
        "without_submitted": frozenset({"1_aaaa", "3_cccc"}),
    }
    assert truth.drawn_groups() == {
        "all": frozenset({"1_aaaa", "2_bbbb"}),
        "without_submitted": frozenset({"1_aaaa"}),
    }


def test_the_tab_in_the_label_is_kept():
    text = (FIXTURE / bt.LABELS / "6_ffff_structure.csv").read_text()
    assert "CC(C)O\t67" in text


def test_changed_row_count_names_the_paper(root):
    path = root / bt.LABELS / "3_cccc_structure.csv"
    path.write_text(path.read_text() + "CCBr,8,NA,NA\n")
    with pytest.raises(ValueError, match="3_cccc"):
        bt.load_biovista_truth(root, MANIFEST)


def test_missing_label_file_and_columns(root):
    (root / bt.LABELS / "3_cccc_structure.csv").write_text("smiles,ligand,groups\nCCCl,7a,R=Cl\n")
    with pytest.raises(ValueError, match="backbone"):
        bt.load_biovista_truth(root, MANIFEST)
    (root / bt.LABELS / "3_cccc_structure.csv").unlink()
    with pytest.raises(ValueError, match="3_cccc"):
        bt.load_biovista_truth(root, MANIFEST)


def test_changed_row_count_of_a_dropped_paper_names_it(root):
    path = root / bt.LABELS / "6_ffff_structure.csv"
    path.write_text(path.read_text() + "CCBr,10,NA,NA\n")
    with pytest.raises(ValueError, match="6_ffff"):
        bt.load_biovista_truth(root, MANIFEST)


def test_paper_without_readable_label_is_dropped(root):
    (root / bt.LABELS / "3_cccc_structure.csv").write_text("smiles,ligand,backbone,groups\nNA,7a,CC*,R=Cl\n")
    truth = bt.load_biovista_truth(root, MANIFEST)
    assert list(truth.dropped) == ["3_cccc", "6_ffff"]
    assert "3_cccc" not in truth.references


def test_checksum_changes_with_a_label_file(root):
    before = bt.reference_set_sha256(root, MANIFEST)
    assert before == bt.reference_set_sha256(FIXTURE, MANIFEST)
    path = root / bt.LABELS / "3_cccc_structure.csv"
    path.write_text(path.read_text().replace("CCCl", "CCCCl"))
    assert bt.reference_set_sha256(root, MANIFEST) != before


def test_checksum_covers_the_labels_of_dropped_papers(root):
    # Fixing a dropped paper's label would bring it back, so the reference set must change with it.
    before = bt.reference_set_sha256(root, MANIFEST)
    path = root / bt.LABELS / "6_ffff_structure.csv"
    path.write_text(path.read_text().replace("NA,8,", "CCO,8,"))
    assert bt.reference_set_sha256(root, MANIFEST) != before


def test_blank_backbone_is_enumerated_not_drawn(tmp_path):
    labels = tmp_path / bt.LABELS
    labels.mkdir(parents=True)
    (labels / "1_aaaa_structure.csv").write_text("smiles,backbone,groups\nCCO,NA,NA\nCCN,,\nCCC, ,\nCCCl,CC*,R=Cl\n")
    manifest = tmp_path / "manifest.csv"
    lines = MANIFEST.read_text().splitlines()
    manifest.write_text("\n".join([lines[0], lines[1].replace(",3,", ",4,", 1)]) + "\n")
    truth = bt.load_biovista_truth(tmp_path, manifest)
    assert dict(truth.drawn) == {"1_aaaa": ("CCO",)}
    assert dict(truth.enumerated) == {"1_aaaa": ("CCN", "CCC", "CCCl")}


# Rebuilt labels: loading applies repair_label to every label


def test_a_naive_paste_label_is_repaired_to_the_molzip_join():
    assert canonical_smiles(PASTE_KEY) == "C[C@H]1COCCN=C2N=C3N=C(Cl)N3[C@@H](C)COCCN21"  # not the drawn molecule
    assert bt.repair_label(PASTE_KEY, PASTE_BACKBONE, PASTE_GROUPS) == JOIN
    assert bt.repair_label(f" {PASTE_KEY} ", f" {PASTE_BACKBONE} ", PASTE_GROUPS.replace(";", " ; ")) == JOIN


def joined(backbone, groups):
    return canonical_smiles(Chem.MolToSmiles(Chem.molzip(Chem.MolFromSmiles(".".join([backbone, *groups.split(";")])))))


def is_naive_paste_unlike_its_join(smiles, backbone, groups):
    """Conditions (a) to (c): the label is the naive paste, the join parses, and the join is another molecule."""
    pasted = backbone
    for group in groups.split(";"):
        tag = group[: group.index("]") + 1]
        pasted = pasted.replace(tag, group[len(tag):])
    key = canonical_smiles(smiles)
    return key == canonical_smiles(pasted) and joined(backbone, groups) not in (None, key)


def test_the_broken_backbone_row_is_a_mismatch_the_rule_must_leave():
    smiles, backbone, groups = BROKEN_BACKBONE
    assert joined(backbone, groups) != canonical_smiles(smiles)  # molzip garbles it, the label is right
    assert bt.repair_label(smiles, backbone, groups) is None


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(ATTACHMENT_LAST, id="26_6st3-attachment-last"),
        pytest.param(LEADING_PASTE, id="498_6h52-leading-paste"),
        pytest.param(SELF_GARBLED, id="221_6oac-self-garbled-backbone"),
    ],
)
def test_the_doubtful_pastes_are_left_as_they_are(row):
    assert is_naive_paste_unlike_its_join(*row)  # they pass (a) to (c)
    assert bt.repair_label(*row) is None


def test_d_the_paste_with_renumbered_group_digits_must_be_the_join():
    # A salt whose second part starts at its attachment: the paste bonds through the group's last atom (CH2OH),
    # the join through its first (OMe). The backbone string does not start with it, and it has no small ring.
    flipped = ("Cl.OCc1ccccc1", "Cl.[*:1]c1ccccc1", "[*:1]OC")
    assert is_naive_paste_unlike_its_join(*flipped)
    assert bt.repair_label(*flipped) is None
    # The same salt with a digit collision and nothing else: renumbering the group's digit gives the join.
    collided = ("Cl.c1ccc(N1CCOCC1)cc1", "Cl.c1ccc([*:1])cc1", "[*:1]N1CCOCC1")
    assert bt.repair_label(*collided) == canonical_smiles("Cl.c1ccc(N2CCOCC2)cc1") == joined(*collided[1:])


def test_e_no_attachment_may_start_the_backbone():
    # A digit collision that renumbering fixes, but the backbone starts at [*:1], so its paste direction is unclear.
    leading = ("CNC(=O)c1ccc(N1CCOCC1)cc1", "[*:1]NC(=O)c1ccc([*:2])cc1", "[*:1]C;[*:2]N1CCOCC1")
    assert is_naive_paste_unlike_its_join(*leading)
    assert canonical_smiles("CNC(=O)c1ccc(N2CCOCC2)cc1") == joined(*leading[1:])  # (d) holds
    assert bt.repair_label(*leading) is None
    # The same molecule with the attachment written later in the backbone is repaired.
    inner = ("O=C(NC)c1ccc(N1CCOCC1)cc1", "O=C(N[*:1])c1ccc([*:2])cc1", "[*:1]C;[*:2]N1CCOCC1")
    assert bt.repair_label(*inner) == joined(*leading[1:])


def test_f_a_backbone_that_garbles_itself_is_left():
    smiles, backbone, groups = SELF_GARBLED
    assert canonical_smiles("Clc1nc(N1CCOCC1)nc(N2[C@@H](C)COCC2)n1") == joined(backbone, groups)  # (d) holds
    assert not bt.ATTACHMENT.match(backbone)  # (e) holds
    assert bt.repair_label(smiles, backbone, groups) is None
    # Benzocyclobutene's 4-ring has two aromatic atoms: a real ring, so its digit collision is repaired.
    fused = ("c1c(N1CCOCC1)cc2c(c1)CC2", "c1c([*:1])cc2c(c1)CC2", "[*:1]N1CCOCC1")
    assert bt.repair_label(*fused) == joined(*fused[1:]) == canonical_smiles("c1c(N3CCOCC3)cc2c(c1)CC2")


@pytest.mark.parametrize(
    ("smiles", "backbone", "groups"),
    [
        pytest.param(*CONSISTENT, id="label-is-the-join"),
        pytest.param("OC(=O)c1ccccc1", "[*:1]c1ccccc1", "[*:1]C(=O)O", id="paste-is-the-join"),
        pytest.param(*UNPARSABLE_GROUP, id="group-does-not-parse"),
        pytest.param("CCCCl", "CC*", "R=CCl", id="group-without-its-attachment"),
        pytest.param("CCCCl", "CC[*:1]", "NA", id="no-groups"),
        pytest.param("CCCCl", "CC[*:1]", "", id="blank-groups"),
        pytest.param("CCCCl", "CC[*:1]", "[*:1]CCl;[*:1]CCl", id="one-attachment-twice"),
        pytest.param("CCCCl", "CC[*:1]C[*:2]", "[*:1]CCl", id="an-attachment-without-a-group"),
        pytest.param("CCCCl", "CC[*:1]", "[*:1]C[*:3]Cl", id="a-group-with-two-attachments"),
        pytest.param("CCCCl", "CC[*:1]", "[*:1]C Cl", id="whitespace-in-a-group"),
        pytest.param(PASTE_KEY, "NA", PASTE_GROUPS, id="drawn"),
        pytest.param(PASTE_KEY, "", PASTE_GROUPS, id="blank-backbone"),
        pytest.param("C1CC", PASTE_BACKBONE, PASTE_GROUPS, id="unreadable-label"),
    ],
)
def test_other_labels_are_left_as_they_are(smiles, backbone, groups):
    assert bt.repair_label(smiles, backbone, groups) is None


def test_loading_rebuilds_only_the_pasted_label_and_records_it(tmp_path):
    root, manifest = labelled_paper(tmp_path)
    truth = bt.load_biovista_truth(root, manifest)
    assert dict(truth.repairs) == {"7_gggg:2": bt.Repair(original=PASTE_KEY, repaired=JOIN)}
    labelled = tuple(row[0] for row in LABEL_ROWS)
    assert truth.references["7_gggg"] == (JOIN, *labelled[1:])  # the join is scored, not the paste
    assert truth.enumerated["7_gggg"] == (JOIN, *labelled[1:4])
    assert dict(truth.drawn) == {"7_gggg": ("c1ccncc1",)}
    assert truth.labels == len(LABEL_ROWS)


def test_the_fixture_has_no_label_to_rebuild(truth):
    assert dict(truth.repairs) == {}  # its groups (R=Cl) carry no [*:n]


def test_label_files_need_the_groups_column(tmp_path):
    rows = tuple((row[0], row[2]) for row in LABEL_ROWS)
    root, manifest = labelled_paper(tmp_path, rows, header="smiles,backbone")
    with pytest.raises(ValueError, match="groups"):
        bt.load_biovista_truth(root, manifest)


def test_the_checksum_covers_the_rebuilt_labels(tmp_path, monkeypatch):
    root, manifest = labelled_paper(tmp_path)
    rebuilt = bt.reference_set_sha256(root, manifest)
    assert len(rebuilt) == 64 and bt.reference_set_sha256(root, manifest) == rebuilt
    monkeypatch.setattr(bt, "repair_label", lambda *_: None)  # the same label files with no row rebuilt
    assert bt.reference_set_sha256(root, manifest) != rebuilt
