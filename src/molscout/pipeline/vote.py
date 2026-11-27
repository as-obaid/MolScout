"""Combine several readers' SMILES for one region, and vote every region of a paper.

A read takes part when it has no error and its SMILES parses. Reads are grouped by
connectivity: the InChIKey's first block for a molecule, and for an R-group core (dummy atoms
`[*:n]`, which InChI cannot read) the canonical SMILES without stereo and with every dummy
bare. The largest group wins and `agree` is its size; two or more groups of the largest size
are a tie and give no result.

A core's slot labels must agree as well. Within the winning group the reads are compared again
with each `[*:n]` standing for its label from `Read.slots` (`enumerate.label_key` spelling), so
readers that number the slots differently but draw the same labels in the same places agree.
The largest such subgroup wins and gives `agree`; a tie there is "slot labels disagree". With
the `slot_labels` fix a tie gives the subgroup of the reader READER_PRIORITY ranks first, records
each other tied labelling as an alternative ("slot_labels") and the skeleton group's size as
`skeleton_agree`, so a table's columns can settle the labels (`link.choose_core`).

The result's SMILES and `slots` come from one read of the winning group, never mixed: the
(SMILES, slots) pair written most often, ties going to the reader whose name sorts first.

Stereo (the `stereo` fix): kept only when every winning read has the same stereo, otherwise
removed and `stereo_kept` is False. With the `partial_stereo` fix the reads without any stereo
are left out of that test, and the SMILES comes from a read with stereo. With `stereo=False` it
is removed from every vote.

An attachment drawn as a wavy bond is a dummy to MolGlyph, but often a short carbon chain or
nothing to MolScribe and MolNexTR. With the `attachments` fix, when the vote gives a plain
molecule or a tie, each drawn core that plain reads support is recorded as an alternative
("attachment"). A plain read supports a core when its connectivity is the core's with each dummy
read as H, methyl, ethyl, n-propyl or isopropyl (every fill for up to two dummies, H or methyl
for three or four, none for more). The alternative's `agree` counts the core's reads with the
same labels and the plain reads that support it. The result itself does not change.

`confidence = agree / number of readers asked`, failed reads included, so 2 of 3 readers
agreeing gives 0.67 whether the third read failed or disagreed.

`vote_paper` writes one voted.jsonl record per region of regions.jsonl. A region the gate
rejected, or one without a vote, gets `smiles` None and its reason (a tie keeps its
alternatives). With the `vote` fix off a region takes the read of one reader (the primary
reader), so the ladder's lower rungs see a single reader. A drawing cut from a table cell
(Region.kind "cell") is voted by `cell_vote.vote_cell`: an R group with exactly one attachment
point `[*:1]`, or a whole molecule.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from molscout.pipeline.cell_vote import NO_ATTACHMENT, carries_stereo, vote_cell
from molscout.pipeline.enumerate import label_key
from molscout.pipeline.schema import Read, Region, VotedRead, fix_enabled, need, read_jsonl, write_jsonl
from molscout.scoring.smiles import canonical_smiles, stereo_stripped_smiles

CONNECTIVITY_CHARS = 14
READER_PRIORITY = ("molglyph", "molscribe", "molnextr")
NOT_STRUCTURE = "not a structure"
NO_READS = "no reads"
NO_USABLE_READ = "no usable read"
LABELS_DISAGREE = "slot labels disagree"
CELL = "cell"  # Region.kind of a drawing cut from a table cell
ATTACHMENT, SLOT_LABELS = "attachment", "slot_labels"  # the kinds of VotedRead.alternatives
ALTERNATIVE_FIXES = {ATTACHMENT: "attachments", SLOT_LABELS: "slot_labels"}
# What a plain read may have where a core has a dummy: H, or a carbon with more carbons, each given as the atom it
# bonds to (0 the dummy's own carbon, n the n-th one added).
FILLS = {"H": None, "methyl": (), "ethyl": (0,), "n-propyl": (0, 1), "isopropyl": (0, 0)}
FILL_LIMITS = ((2, tuple(FILLS)), (4, ("H", "methyl")))  # (most dummies, the fills tried for each)
_CATEGORIES = {
    NOT_STRUCTURE: "not_structure",
    NO_READS: "no_reads",
    NO_USABLE_READ: "no_usable_read",
    LABELS_DISAGREE: "slot_labels_disagree",
    NO_ATTACHMENT: "cell_no_attachment",
}


@dataclass(frozen=True)
class _Ballot:
    reader: str
    smiles: str  # canonical, with the reader's own slot numbers
    skeleton: str  # connectivity, dummies bare
    labelled: str  # connectivity, each dummy standing for its label
    stereo: str  # `labelled` with stereo (the full InChIKey for a molecule)
    slots: dict[str, str] = field(default_factory=dict)

    @property
    def is_core(self) -> bool:
        return "*" in self.smiles

    @property
    def pair(self) -> tuple[str, tuple[tuple[str, str], ...]]:
        return self.smiles, tuple(sorted(self.slots.items()))


def vote(reads: list[Read], *, stereo: bool = True, fixes: Any = None) -> VotedRead | None:
    """The agreed read for one region, or None on a tie or when no read is usable."""
    return vote_with_reason(reads, stereo=stereo, fixes=fixes)[0]


def vote_with_reason(
    reads: list[Read], *, stereo: bool = True, fixes: Any = None
) -> tuple[VotedRead | None, str | None]:
    """`vote`, plus why there is no result: "no reads", "no usable read",
    "tie between N structures" or "slot labels disagree".

    `fixes` switches the attachments, slot_labels and partial_stereo rules (all on for None).
    Raises ValueError when the reads come from more than one region or a reader appears twice.
    """
    voted, reason, _ = _vote(reads, stereo, fixes)
    return voted, reason


def vote_record(region_id: str, reads: list[Read], *, stereo: bool = True, fixes: Any = None) -> VotedRead:
    """One region's voted.jsonl record: `vote`'s result, or no SMILES and the reason; a tie keeps its alternatives."""
    voted, reason, alternatives = _vote(reads, stereo, fixes)
    return voted if voted is not None else _no_vote(region_id, reason or NO_USABLE_READ, alternatives)


def alternative_reads(read: VotedRead, fixes: Any = None) -> list[VotedRead]:
    """Each alternative core of a voted read whose fix is on, as a read of the same region (for link and
    scheme_variants): its own SMILES, slots, agree, readers and confidence."""
    kinds = {kind for kind, fix in ALTERNATIVE_FIXES.items() if fix_enabled(fixes, fix)}
    return [replace(read, smiles=alternative["smiles"], inchikey=None, agree=alternative["agree"],
                    confidence=alternative.get("confidence", read.confidence), readers=list(alternative["readers"]),
                    slots=dict(alternative["slots"]), reason=None, alternatives=[])
            for alternative in read.alternatives if alternative.get("kind") in kinds]


def _vote(reads: list[Read], stereo: bool, fixes: Any) -> tuple[VotedRead | None, str | None, list[dict[str, Any]]]:
    """The result or why there is none, and the alternatives (a result carries them too)."""
    if not reads:
        return None, NO_READS, []
    _check(reads)
    numbers = _label_numbers(reads)
    ballots = [ballot for read in reads if (ballot := _ballot(read, numbers)) is not None]
    if not ballots:
        return None, NO_USABLE_READ, []
    region_id, asked, partial = reads[0].region_id, len(reads), fix_enabled(fixes, "partial_stereo")
    masked = fix_enabled(fixes, "attachments")
    leaders = _largest(ballots, lambda ballot: ballot.skeleton)
    if len(leaders) > 1:
        found = _attachment_cores(region_id, ballots, asked, stereo, partial) if masked else []
        return None, f"tie between {len(leaders)} structures", found
    labelled = _largest(leaders[0], lambda ballot: ballot.labelled)
    if len(labelled) > 1 and not fix_enabled(fixes, "slot_labels"):
        return None, LABELS_DISAGREE, []
    first, *others = sorted(labelled, key=_priority)
    voted = _result(region_id, first, asked, stereo, partial)
    if others:
        found = [_alternative(_result(region_id, group, asked, stereo, partial), SLOT_LABELS) for group in others]
        return replace(voted, skeleton_agree=len(leaders[0]), alternatives=found), None, found
    if not first[0].is_core and masked:
        found = _attachment_cores(region_id, ballots, asked, stereo, partial)
        return replace(voted, alternatives=found), None, found
    return voted, None, []


def vote_paper(
    paper_dir: Path | str,
    *,
    fixes: Any = None,
    readers: Sequence[str] | None = None,
    primary_reader: str = READER_PRIORITY[0],
) -> dict[str, int]:
    """Vote every region of one paper into voted.jsonl; return the counts.

    Only the reads of `readers` take part (all readers when None). With the `vote` fix off each
    region takes `primary_reader`'s read, else the next reader of READER_PRIORITY.

    Counts: `in` regions = `out` (regions with a voted SMILES) + every `dropped_<reason>`
    (not_structure, no_reads, no_usable_read, tie, slot_labels_disagree, cell_no_attachment);
    `cores` counts the voted R-group cores. A paper with table-cell drawings adds `cells` (its cell
    regions), `cell_fragments` (voted with one attachment point) and `cell_molecules` (voted as a
    whole molecule). Missing regions.jsonl, or reads.jsonl when a region is a structure, raises
    FileNotFoundError; reads of a region regions.jsonl lacks raise ValueError.
    """
    paper_dir = Path(paper_dir)
    regions = read_jsonl(need(paper_dir / "regions.jsonl"), Region)
    by_region = _reads_by_region(paper_dir, regions)
    if readers is not None:
        by_region = {rid: [read for read in reads if read.reader in readers] for rid, reads in by_region.items()}
    stereo = fix_enabled(fixes, "stereo")
    pick = None if fix_enabled(fixes, "vote") else (primary_reader, *READER_PRIORITY)
    records = [_vote_region(region, by_region.get(region.region_id, []), stereo, pick, fixes) for region in regions]
    write_jsonl(paper_dir / "voted.jsonl", records)
    cells = {region.region_id for region in regions if region.kind == CELL}
    counts = _counts([record for record in records if record.region_id not in cells])
    for name, value in _counts([record for record in records if record.region_id in cells]).items():
        if name != "cores":
            counts[name] = counts.get(name, 0) + value
    if cells:
        voted = [record.smiles for record in records if record.region_id in cells and record.smiles]
        counts.update(cells=len(cells), cell_fragments=sum("*" in smiles for smiles in voted),
                      cell_molecules=sum("*" not in smiles for smiles in voted))
    return counts


def _vote_region(
    region: Region, reads: list[Read], stereo: bool, pick: tuple[str, ...] | None, fixes: Any
) -> VotedRead:
    if not region.is_structure:
        return _no_vote(region.region_id, NOT_STRUCTURE)
    given = reads if pick is None else _first_reader(reads, pick)
    if region.kind != CELL or not given:
        return vote_record(region.region_id, given, stereo=stereo, fixes=fixes)
    _check(given)
    voted, reason = vote_cell(given, stereo=stereo, fallback=vote_with_reason, fixes=fixes)
    return voted if voted is not None else _no_vote(region.region_id, reason or NO_USABLE_READ)


def _first_reader(reads: list[Read], order: tuple[str, ...]) -> list[Read]:
    """The read of the reader `order` ranks first (readers it lacks after, by name), alone."""
    rank: dict[str, int] = {}
    for reader in order:
        rank.setdefault(reader, len(rank))
    return sorted(reads, key=lambda read: (rank.get(read.reader, len(rank)), read.reader))[:1]


def _no_vote(region_id: str, reason: str, alternatives: Sequence[dict[str, Any]] = ()) -> VotedRead:
    return VotedRead(region_id, None, None, 0, False, 0.0, [], reason=reason, alternatives=list(alternatives))


def _reads_by_region(paper_dir: Path, regions: Sequence[Region]) -> dict[str, list[Read]]:
    path = paper_dir / "reads.jsonl"
    if not path.is_file() and not any(region.is_structure for region in regions):
        return {}
    by_region: dict[str, list[Read]] = {}
    for read in read_jsonl(need(path), Read):
        by_region.setdefault(read.region_id, []).append(read)
    unknown = sorted(set(by_region) - {region.region_id for region in regions})
    if unknown:
        raise ValueError(f"{path}: reads of regions not in regions.jsonl: {unknown}")
    return by_region


def _counts(records: Sequence[VotedRead]) -> dict[str, int]:
    voted = [record for record in records if record.smiles]
    counts = Counter({"in": len(records), "out": len(voted), "cores": sum("*" in record.smiles for record in voted)})
    for record in records:
        if not record.smiles:
            counts[f"dropped_{_category(record.reason or NO_USABLE_READ)}"] += 1
    return dict(counts)


def _category(reason: str) -> str:
    return "tie" if reason.startswith("tie between") else _CATEGORIES[reason]


def _largest(ballots: list[_Ballot], key: Callable[[_Ballot], str]) -> list[list[_Ballot]]:
    """The groups of the largest size, in order of first appearance."""
    groups: dict[str, list[_Ballot]] = {}
    for ballot in ballots:
        groups.setdefault(key(ballot), []).append(ballot)
    largest = max(len(group) for group in groups.values())
    return [group for group in groups.values() if len(group) == largest]


def _priority(group: list[_Ballot]) -> tuple[int, str]:
    """How early READER_PRIORITY ranks a group's best reader (readers it lacks after, by name)."""
    return min((READER_PRIORITY.index(b.reader) if b.reader in READER_PRIORITY else len(READER_PRIORITY), b.reader)
               for b in group)


def _alternative(voted: VotedRead, kind: str) -> dict[str, Any]:
    return {"smiles": voted.smiles, "slots": dict(voted.slots), "agree": voted.agree, "readers": list(voted.readers),
            "confidence": voted.confidence, "kind": kind}


def _attachment_cores(
    region_id: str, ballots: list[_Ballot], asked: int, stereo: bool, partial: bool
) -> list[dict[str, Any]]:
    """Each drawn core (by its labels) that plain reads support once its dummies are read as FILLS, most agreeing
    first: SMILES and slots from the core's reads, agree and readers counting the plain reads too."""
    plain = [ballot for ballot in ballots if not ballot.is_core]
    cores: dict[str, list[_Ballot]] = {}
    for ballot in ballots if plain else []:
        if ballot.is_core:
            cores.setdefault(ballot.labelled, []).append(ballot)
    found = []
    for group in cores.values():
        keys = _filled_keys(group[0].smiles)
        members = {ballot.reader for ballot in group} | {ballot.reader for ballot in plain if ballot.skeleton in keys}
        if len(members) > len(group):
            readers = [ballot.reader for ballot in ballots if ballot.reader in members]
            core = replace(_result(region_id, group, asked, stereo, partial), agree=len(readers),
                           confidence=len(readers) / asked, readers=readers)
            found.append(_alternative(core, ATTACHMENT))
    return sorted(found, key=lambda alternative: -alternative["agree"])


def _filled_keys(smiles: str) -> set[str]:
    """The connectivity (InChIKey first block) of a core with each dummy read as one of FILLS, within FILL_LIMITS."""
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    dummies = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0] if mol is not None else []
    fills = next((names for most, names in FILL_LIMITS if len(dummies) <= most), ())
    keys = set()
    for combination in product(fills, repeat=len(dummies)) if dummies else ():
        filled = _filled(mol, dict(zip(dummies, combination, strict=True)))
        key = _inchikey(filled)[:CONNECTIVITY_CHARS] if filled is not None else ""
        if key:
            keys.add(key)
    return keys


def _filled(mol: Chem.Mol, fills: dict[int, str]) -> Chem.Mol | None:
    """mol with dummy atom i read as FILLS[fills[i]]; None when that does not make a molecule."""
    work = Chem.RWMol(mol)
    for index, name in fills.items():
        atom = work.GetAtomWithIdx(index)
        atom.SetAtomMapNum(0)
        atom.SetIsotope(0)
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
        atom.SetAtomicNum(1 if FILLS[name] is None else 6)
        added = [index]
        for parent in FILLS[name] or ():
            added.append(work.AddAtom(Chem.Atom(6)))
            work.AddBond(added[parent], added[-1], Chem.BondType.SINGLE)
    with rdBase.BlockLogs():
        return Chem.MolFromSmiles(Chem.MolToSmiles(work))  # parsing folds each H into its neighbour


def _stereo_reads(group: list[_Ballot], partial: bool) -> list[_Ballot]:
    """The reads whose stereo the result keeps: all of them when they have the same stereo; with `partial`, those
    with any stereo when they have the same; else none."""
    if len({ballot.stereo for ballot in group}) == 1:
        return group
    carrying = [ballot for ballot in group if carries_stereo(ballot.smiles)] if partial else []
    return carrying if carrying and len({ballot.stereo for ballot in carrying}) == 1 else []


def _result(region_id: str, group: list[_Ballot], asked: int, stereo: bool, partial: bool = False) -> VotedRead:
    kept = _stereo_reads(group, partial) if stereo else []
    stereo_kept = bool(kept)
    pool = kept or group
    counts = Counter(ballot.pair for ballot in pool)
    chosen = min(pool, key=lambda ballot: (-counts[ballot.pair], ballot.reader))
    smiles = chosen.smiles if stereo_kept else stereo_stripped_smiles(chosen.smiles)
    if chosen.is_core:
        inchikey = None
    else:
        inchikey = chosen.stereo if stereo_kept else _inchikey(Chem.MolFromSmiles(smiles))
    readers = [ballot.reader for ballot in group]
    return VotedRead(region_id, smiles, inchikey, len(group), stereo_kept, len(group) / asked, readers,
                     slots=dict(chosen.slots))


def _ballot(read: Read, numbers: dict[str, int]) -> _Ballot | None:
    if read.error or not read.smiles:
        return None
    smiles = canonical_smiles(read.smiles)
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    dummies = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if not dummies:
        inchikey = _inchikey(mol)
        skeleton = inchikey[:CONNECTIVITY_CHARS]
        return _Ballot(read.reader, smiles, skeleton, skeleton, inchikey) if inchikey else None
    given = read.slots or {}
    slots = {str(atom.GetAtomMapNum()): given[str(atom.GetAtomMapNum())]
             for atom in dummies if str(atom.GetAtomMapNum()) in given}
    labels = {atom.GetIdx(): numbers[label_key(slots.get(str(atom.GetAtomMapNum()), "*"))] for atom in dummies}
    return _Ballot(read.reader, smiles, skeleton=_core_key(mol, {}, stereo=False),
                   labelled=_core_key(mol, labels, stereo=False), stereo=_core_key(mol, labels, stereo=True),
                   slots=slots)


def _label_numbers(reads: Sequence[Read]) -> dict[str, int]:
    """One number per label spelling (label_key) used in this region; "*" for an unlabelled slot."""
    keys = {label_key(label) for read in reads for label in (read.slots or {}).values()} | {"*"}
    return {key: number for number, key in enumerate(sorted(keys), start=1)}


def _core_key(mol: Chem.Mol, labels: dict[int, int], *, stereo: bool) -> str:
    """Canonical SMILES of a core whose dummies carry `labels` (atom index -> number) as isotopes."""
    work = Chem.Mol(mol)
    for atom in work.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomMapNum(0)
            atom.SetIsotope(labels.get(atom.GetIdx(), 0))
    text = Chem.MolToSmiles(work)
    key = canonical_smiles(text) if stereo else stereo_stripped_smiles(text)
    return key or text


def _check(reads: list[Read]) -> None:
    regions = {read.region_id for read in reads}
    if len(regions) > 1:
        raise ValueError(f"reads from {len(regions)} regions: {sorted(regions)}")
    repeated = [reader for reader, count in Counter(read.reader for read in reads).items() if count > 1]
    if repeated:
        raise ValueError(f"more than one read from {repeated}")


def _inchikey(mol: Chem.Mol) -> str:
    with rdBase.BlockLogs():
        return Chem.MolToInchiKey(mol)
