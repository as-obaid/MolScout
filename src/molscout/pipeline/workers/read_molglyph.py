"""Read gated crops with MolGlyph. Runs in the molglyph environment (BioMiner's MolScribe fork).

MolGlyph writes a MolParser E-SMILES caption, "<smiles><sep><groups>", where each group
"<a>i:text</a>" names what is drawn at dummy atom i of the SMILES ("R[1]", "X", "OMe", "<dum>").
BioMiner (interface.py, molglyph_utils.Translator) expands the groups that are abbreviations in
its abbrevs.csv, leaves the others as "*" and then numbers the stars [1*], [2*] by label text,
which loses the labels. Here the same table is used, with BioMiner's rules (only a dummy with one
single bond takes a group; "Ar" is never expanded, as it may be any aryl), and every other dummy
becomes [*:n] with its label in slots ("1" -> "R1"), numbered as in _readers_common.

Differences from BioMiner: a group is an abbreviation when its whole text is a symbol of the
table, so "t-Bu" and "CH(CH3)2" expand (BioMiner's group pattern keeps only [A-Za-z0-9] and drops
them); labels in any script form ("R[1]", "R^1") are kept. As in BioMiner, a caption with
ring-bound groups (<r>, <c>) has no fixed attachment and gives no molecule. A caption without
<sep> (decoding ran to the 512-token limit) is kept as SMILES for the checks to reject.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _readers_common as common  # noqa: E402

READER = "molglyph"
SEP = "<sep>"
GROUP = re.compile(r"<a>(\d+):(.*?)</a>")
NOT_ABBREVIATIONS = {"Ar"}  # BioMiner: "Ar" might be a variable aromatic ring
DUMMY = "<dum>"


def load_abbrevs(path) -> dict[str, str]:
    """symbol -> SMILES from BioMiner's abbrevs.csv (name,smiles,atoms,symbol,label,labelW); the
    SMILES's first atom bonds to the drawing, as in BioMiner's merge_group."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return {row["symbol"]: row["smiles"] for row in csv.DictReader(handle)}


def caption_to_read(caption: str, abbrevs: dict[str, str]) -> tuple[str, dict[str, str]]:
    """A MolGlyph caption as (smiles with [*:n], slots); NoMolecule for ring-bound groups."""
    from rdkit import Chem, RDLogger

    if SEP not in caption:
        return common.smiles_read(caption)
    core, groups = caption.split(SEP, 1)
    if not core.strip():
        raise common.NoMolecule("empty SMILES before <sep>")
    if "<r>" in groups or "<c>" in groups:
        raise common.NoMolecule("ring-bound group (<r>/<c>) has no fixed attachment")
    RDLogger.DisableLog("rdApp.*")
    try:
        mol = Chem.MolFromSmiles(core)
    finally:
        RDLogger.EnableLog("rdApp.*")
    if mol is None:
        return core, {}
    mol = Chem.RWMol(mol)
    expanded = []
    for index, text in list(_groups(groups, mol)):
        if text == DUMMY:
            continue  # an attachment point: label "*"
        smiles = abbrevs.get(text) if text not in NOT_ABBREVIATIONS else None
        if smiles and _attach(mol, index, smiles):
            expanded.append(index)
        else:
            Chem.SetAtomAlias(mol.GetAtomWithIdx(index), text)
    for index in sorted(expanded, reverse=True):
        mol.RemoveAtom(index)
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        raise common.NoMolecule(f"invalid after expanding groups: {exc}") from exc
    return common.assign_slots(mol)


def _groups(groups: str, mol):
    """(atom index, text) of each <a> group on a dummy atom, first group per atom."""
    seen = set()
    for index, text in GROUP.findall(groups):
        index = int(index)
        if index in seen or index >= mol.GetNumAtoms() or mol.GetAtomWithIdx(index).GetAtomicNum() != 0:
            continue
        seen.add(index)
        yield index, text.strip()


def _attach(mol, index: int, smiles: str) -> bool:
    """Replace dummy atom index by the group (its first atom bonds where the dummy did), as
    BioMiner's merge_group; the dummy is removed by the caller. False when it cannot be done."""
    from rdkit import Chem

    dummy = mol.GetAtomWithIdx(index)
    if dummy.GetDegree() != 1:
        return False
    bond = dummy.GetBonds()[0]
    if bond.GetBondType() != Chem.BondType.SINGLE:
        return False
    group = Chem.MolFromSmiles(smiles)
    if group is None:
        return False
    neighbor = bond.GetOtherAtomIdx(index)
    offset = mol.GetNumAtoms()
    for atom in group.GetAtoms():
        mol.AddAtom(atom)
    for b in group.GetBonds():
        mol.AddBond(b.GetBeginAtomIdx() + offset, b.GetEndAtomIdx() + offset, b.GetBondType())
    mol.RemoveBond(index, neighbor)
    mol.AddBond(neighbor, offset, Chem.BondType.SINGLE)
    return True


def add_arguments(parser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True, help="molglyph_large.pt")
    parser.add_argument("--abbrevs", type=Path, required=True, help="BioMiner/commons/abbrevs.csv")
    parser.add_argument("--device", default="cuda")


def load_reader(args, logger):
    """MolGlyph loaded as benchmarks/tools/structure_readers/molglyph/run.py loads it."""
    abbrevs = load_abbrevs(args.abbrevs)
    import torch
    from molscribe import MolGlyph

    model = MolGlyph(str(args.checkpoint), device=torch.device(args.device))
    logger.info("MolGlyph %s on %s, %d abbreviations", args.checkpoint, args.device, len(abbrevs))

    def predict(crop: Path):
        # return_confidence=True fails in MolGlyph (its tokenizer gives no atom indices).
        caption = model.predict_image_file(str(crop), return_confidence=False)
        return caption_to_read(caption, abbrevs)

    return predict


def main(argv=None) -> int:
    return common.reader_main(READER, __doc__.split("\n\n")[0], add_arguments, load_reader, argv)


if __name__ == "__main__":
    sys.exit(main())
