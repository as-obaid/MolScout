"""The vote for a drawing cut from an R-group table cell: an R group with one attachment point, or a whole molecule.

Readers write an R group's attachment (a wavy or dashed bond) differently. On BioVista and on RDKit drawings
MolGlyph writes a dummy atom `*`; MolScribe and MolNexTR mostly read the bond as a bond to one more carbon (a
methyl where the attachment is), sometimes as a short chain or not at all, and sometimes leave a radical there. So
each read gives candidate fragments:

- a read with exactly one dummy atom gives that fragment: the reader drew the attachment;
- a read without dummies gives one fragment for each methyl carbon (the attachment read as a carbon, which becomes
  `*`) and one for each radical atom (the attachment left open, where `*` is bonded), with radicals cleared;
- with the `attachments` fix, it also gives one for each ethyl, n-propyl or isopropyl group (the attachment read as
  a short chain, which becomes `*` as a whole) and one for each heavy atom with a hydrogen (the attachment left
  out, `*` in place of the hydrogen);
- a read of several pieces keeps its largest when every other piece is under half its size (a stray word);
- a read with two or more dummies, or two large pieces, gives nothing.

The fragment that at least one reader drew and that the most readers give as a candidate (compared by
connectivity) wins; a tie gives no vote. `agree` counts the readers that give it. With the `stereo` fix its stereo
is kept when they all give the same stereo; with `partial_stereo` also when those that give any stereo agree on it,
the others giving none. When no reader drew an attachment, the readers' plain vote may give a whole molecule
(`fallback`, vote.vote_with_reason): some tables draw a whole compound in an R column. A plain vote that keeps
dummies gives no vote.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from rdkit import Chem, rdBase

from molscout.pipeline.schema import Read, VotedRead, fix_enabled
from molscout.scoring.smiles import canonical_smiles, stereo_stripped_smiles

NO_ATTACHMENT = "no reader drew one attachment point and the readers do not agree on a molecule"
STRAY_SHARE = 0.5  # a second piece this large next to the largest makes a read ambiguous
SLOTS = {"1": "*"}
CHAIN_ATOMS = (2, 3)  # an ethyl, n-propyl or isopropyl group may stand for the attachment (the attachments fix)


def candidates(read: Read, *, fixes: Any = None) -> tuple[bool, dict[str, str]]:
    """(whether the reader drew the attachment, {connectivity key: fragment SMILES with [*:1]}) for one read."""
    if read.error or not read.smiles:
        return False, {}
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(read.smiles)
        mol = _main_piece(mol) if mol is not None else None
        if mol is None:
            return False, {}
        dummies = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
        if len(dummies) == 1:
            return True, _keyed([_written(mol, dummies[0])])
        if dummies:
            return False, {}
        found = [*_from_radicals(mol), *_from_methyls(mol)]
        if fix_enabled(fixes, "attachments"):
            found += [*_from_chains(mol), *_from_hydrogens(mol)]
        return False, _keyed(found)


def vote_cell(
    reads: Sequence[Read],
    *,
    stereo: bool = True,
    fallback: Callable[..., tuple[VotedRead | None, str | None]],
    fixes: Any = None,
) -> tuple[VotedRead | None, str | None]:
    """The agreed fragment of one cell drawing, else a whole molecule, else None and why."""
    if not reads:
        return None, "no reads"
    ballots = {read.reader: candidates(read, fixes=fixes) for read in reads}
    drawn = {key for was_drawn, found in ballots.values() if was_drawn for key in found}
    if not drawn:
        voted, _ = fallback(list(reads), stereo=stereo, fixes=fixes)
        if voted is None or "*" in (voted.smiles or ""):
            return None, NO_ATTACHMENT
        return voted, None
    support = {key: sorted(reader for reader, (_, found) in ballots.items() if key in found) for key in drawn}
    most = max(len(readers) for readers in support.values())
    leaders = [key for key, readers in support.items() if len(readers) == most]
    if len(leaders) > 1:
        return None, f"tie between {len(leaders)} fragments"
    readers = support[leaders[0]]
    given = [ballots[reader][1][leaders[0]] for reader in readers]
    drawn_by = [ballots[reader][1][leaders[0]] for reader in readers if ballots[reader][0]]
    counts = Counter(drawn_by)
    chosen = min(drawn_by, key=lambda smiles: (-counts[smiles], drawn_by.index(smiles)))
    stereo_kept = stereo and len(set(given)) == 1
    carrying = {smiles for smiles in given if carries_stereo(smiles)}
    if stereo and not stereo_kept and fix_enabled(fixes, "partial_stereo") and len(carrying) == 1:
        chosen, stereo_kept = carrying.pop(), True  # every read with stereo gives this one SMILES
    smiles = chosen if stereo_kept else stereo_stripped_smiles(chosen)
    return VotedRead(reads[0].region_id, smiles, None, len(readers), stereo_kept, len(readers) / len(reads),
                     list(readers), slots=dict(SLOTS)), None


def attachments(smiles: str | None) -> int:
    """The number of dummy atoms in a SMILES (0 for None or text RDKit cannot read)."""
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles) if smiles else None
    return 0 if mol is None else sum(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms())


def carries_stereo(smiles: str | None) -> bool:
    """Whether a canonical SMILES has tetrahedral or double-bond stereo."""
    return any(mark in (smiles or "") for mark in "@/\\")


def _main_piece(mol: Chem.Mol) -> Chem.Mol | None:
    pieces = sorted(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False), key=lambda m: -m.GetNumAtoms())
    if len(pieces) > 1 and pieces[1].GetNumAtoms() >= STRAY_SHARE * pieces[0].GetNumAtoms():
        return None
    return pieces[0]


def _written(mol: Chem.Mol, dummy: int) -> str | None:
    """Canonical SMILES of mol with atom `dummy` as the attachment [*:1]."""
    work = Chem.RWMol(mol)
    atom = work.GetAtomWithIdx(dummy)
    atom.SetAtomicNum(0)
    atom.SetIsotope(0)
    atom.SetFormalCharge(0)
    atom.SetIsAromatic(False)
    atom.SetNoImplicit(True)
    atom.SetNumExplicitHs(0)
    atom.SetNumRadicalElectrons(0)
    atom.SetAtomMapNum(1)
    for other in work.GetAtoms():
        if other.GetIdx() != dummy:
            other.SetAtomMapNum(0)
    if Chem.SanitizeMol(work, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
        return None
    return canonical_smiles(Chem.MolToSmiles(work))


def _without_radicals(mol: Chem.Mol, keep: int | None = None) -> Chem.RWMol:
    """mol with every radical cleared and its hydrogens filled in, except atom `keep`'s."""
    work = Chem.RWMol(mol)
    for atom in work.GetAtoms():
        if atom.GetNumRadicalElectrons() and atom.GetIdx() != keep:
            atom.SetNumRadicalElectrons(0)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)
    return work


def _from_radicals(mol: Chem.Mol) -> list[str | None]:
    """One fragment per radical atom: `*` bonded where the reader left the bond open."""
    found = []
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() != 1:
            continue
        work = _without_radicals(mol, keep=atom.GetIdx())
        work.GetAtomWithIdx(atom.GetIdx()).SetNumRadicalElectrons(0)
        dummy = work.AddAtom(Chem.Atom(0))
        work.AddBond(atom.GetIdx(), dummy, Chem.BondType.SINGLE)
        found.append(_written(work, dummy))
    return found


def _from_methyls(mol: Chem.Mol) -> list[str | None]:
    """One fragment per methyl carbon, which becomes `*`: the attachment bond read as a bond to a carbon."""
    work = _without_radicals(mol)
    if Chem.SanitizeMol(work, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
        return []
    return [_written(work, atom.GetIdx()) for atom in work.GetAtoms()
            if atom.GetAtomicNum() == 6 and atom.GetDegree() == 1 and atom.GetTotalNumHs() == 3
            and atom.GetFormalCharge() == 0 and not atom.GetIsAromatic()]


def _from_chains(mol: Chem.Mol) -> list[str | None]:
    """One fragment per ethyl, n-propyl or isopropyl group, which becomes `*` as a whole: the attachment bond read
    as a short chain."""
    work = _without_radicals(mol)
    if Chem.SanitizeMol(work, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
        return []
    found = []
    for bond in work.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
            continue
        ends = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for near, far in (ends, ends[::-1]):
            chain = _chain(work, near, far)
            if chain:
                edit = Chem.RWMol(work)
                for index in sorted(chain - {far}, reverse=True):
                    edit.RemoveAtom(index)
                found.append(_written(edit, far - sum(index < far for index in chain - {far})))
    return found


def _chain(mol: Chem.Mol, near: int, far: int) -> set[int] | None:
    """The atoms past the bond near-far, from `far` on, when they are 2 or 3 saturated acyclic carbons."""
    seen, todo = {far}, [far]
    while todo:
        atom = mol.GetAtomWithIdx(todo.pop())
        if not (atom.GetAtomicNum() == 6 and not atom.GetIsAromatic() and not atom.IsInRing()
                and atom.GetFormalCharge() == 0 and atom.GetDegree() + atom.GetTotalNumHs() == 4):
            return None
        for other in atom.GetNeighbors():
            if other.GetIdx() != near and other.GetIdx() not in seen:
                seen.add(other.GetIdx())
                todo.append(other.GetIdx())
        if len(seen) > max(CHAIN_ATOMS):
            return None
    return seen if len(seen) in CHAIN_ATOMS else None


def _from_hydrogens(mol: Chem.Mol) -> list[str | None]:
    """One fragment per heavy atom with a hydrogen, `*` in place of one: the attachment bond left out. The atom's
    tetrahedral stereo is cleared, since the reader drew it without the attachment."""
    work = _without_radicals(mol)
    if Chem.SanitizeMol(work, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
        return []
    found = []
    for atom in work.GetAtoms():
        if atom.GetAtomicNum() <= 1 or not atom.GetTotalNumHs():
            continue
        edit = Chem.RWMol(work)
        target = edit.GetAtomWithIdx(atom.GetIdx())
        target.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        if target.GetNumExplicitHs():
            target.SetNumExplicitHs(target.GetNumExplicitHs() - 1)
        dummy = edit.AddAtom(Chem.Atom(0))
        edit.AddBond(atom.GetIdx(), dummy, Chem.BondType.SINGLE)
        found.append(_written(edit, dummy))
    return found


def _keyed(fragments: Sequence[str | None]) -> dict[str, str]:
    out: dict[str, str] = {}
    for smiles in fragments:
        key = stereo_stripped_smiles(smiles) if smiles else None
        if key is not None:
            out.setdefault(key, smiles)
    return out
