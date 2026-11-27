"""Link and build: label drawings, build one molecule per table row, add named compounds.

Reads ``layout.jsonl``, ``regions.jsonl`` and ``voted.jsonl`` (plus ``pages.jsonl`` for page sizes
when present) and returns `Molecule` records and their counts (`link_counts`). `finalize.finalize` then
applies what every joiner mode shares: the junk and printed-formula checks, the relevance rule and the
dedupe.

Statuses here are about building only. A molecule that builds is ``accepted``; one built from a core
with a problem (a failed core check, too few agreeing readers) or a name that contradicts the drawn
or table structure for its label is ``review``; anything that cannot be built is ``abstained`` with
``smiles`` None and its reasons. Every table row gives at least one record (fix 1). Regions the gate
rejected give no record here; `vote.vote_paper` counts them.

A table's core is renumbered by the labels of its voted read (`enumerate.align_slots`) before it is
checked and built, so slot i of the aligned core is the table's i-th variable and a label drawn twice
gets a copy of the fragment at each place. A core with a single unlabelled slot fills a table with a
single variable. With the core check on (fix 2) a core whose labels match the columns wins over a
nearer one that does not.

A voted read's alternative cores (`VotedRead.alternatives`) are candidates too, right after the read's
own core. With the attachments fix that is a drawn core plain reads support with its slots read as H or
a short chain; when a table uses it, the drawing's record is abstained as that table's core, so its
plain reading is not exported as a compound. With the slot_labels fix it is another labelling of a core
whose readers split on the labels; when only one labelling passes the core check, every reader of the
skeleton (`skeleton_agree`) counts toward MIN_CORE_AGREE.

An R group drawn in a table cell is read from the drawing the cells stage cut for the row
(`table_cells.column_drawings`): its vote (`cell_vote`) gives a fragment with one attachment point,
which fills the slot, or a whole molecule at least as large as the core (and WHOLE_MIN_ATOMS), which
is the row's molecule.
A drawing that MinerU dropped reaches its row by position when the PDF's text places the rows, also
in place of a lone letter such as "N" that MinerU kept from it. A table that prints each compound whole
in a column of drawings, with no variable column, gives one molecule per row from the drawing's vote
(`compound_tables`). Cell drawings give no drawn record, and a page drawing that covers a drawing used
for a row (a whole-compound row that was built) is abstained: the row holds its structure.
A drawn core no table uses is filled from the definitions its figure prints beside it ("20, R = Boc";
`scheme_variants`), one record per compound, with the tables fix.

With the legends fix, an R cell no rule reads that holds a code such as "M0" is filled from the page drawing
labelled with that code (`labels.link_codes`): one on the table's page first, then the page before, then the page
after. A drawing with one attachment point gives that fragment; a plain molecule gives the fragment
`cell_vote.candidates` makes of its methyl (the attachment read as a carbon) when there is exactly one. Only
substituent slots take a legend, and a value that starts with a digit is a compound number, never a code. The
row notes the drawing; a drawing read by fewer than MIN_CORE_AGREE readers sends it to review, as a drawn cell
does. A drawing used this way is abstained: it is a fragment, not a compound.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache, partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rdkit import Chem
from rdkit.rdBase import BlockLogs

from molscout.pipeline import enumerate as enumeration
from molscout.pipeline import rgroups, table_cells
from molscout.pipeline.cell_vote import attachments, candidates
from molscout.pipeline.compound_tables import compound_molecules, find_compound_tables
from molscout.pipeline.labels import link_codes, link_regions, load_ocr_lines, plain_text
from molscout.pipeline.names import TEXT_TYPES as NAME_TYPES
from molscout.pipeline.names import named_compounds
from molscout.pipeline.scheme_variants import scheme_molecules
from molscout.pipeline.schema import (
    LayoutBlock,
    Molecule,
    OcrLine,
    PageRecord,
    Read,
    Region,
    VotedRead,
    fix_enabled,
    need,
    read_jsonl,
)
from molscout.pipeline.tables import RGroupTable, cell_labels, find_rgroup_tables
from molscout.pipeline.vote import alternative_reads

if TYPE_CHECKING:
    from molscout.pipeline.opsin import Opsin

__all__ = ["CoreChoice", "PaperInputs", "choose_core", "fix_enabled", "link_and_build", "link_paper", "load_inputs"]

ACCEPTED, REVIEW, ABSTAINED = "accepted", "review", "abstained"
MIN_CORE_AGREE = 2
NAME_CONFIDENCE = 0.9
EMPTY_CELL = {"", "–", "—", "-", "−"}
COVERED = 0.5  # a page drawing this much inside a cell drawing used for a row is that drawing
WHOLE_MIN_ATOMS = 15  # a cell drawing without an attachment is a whole molecule from this size (and the core's)
LEGEND_PAGES = (0, -1, 1)  # a legend drawing on the table's page first, then on the page before, then after
_SLOT_RE = re.compile(r"\[\*:(\d+)\]")
_REGION_PAGE_RE = re.compile(r"^p(\d+)-")
_UNLABELLED = {"", "*"}
_REMNANT = re.compile(r"^(?:N|O|S|NH|HN)$")  # a letter MinerU kept from a drawing it dropped


@dataclass(frozen=True)
class PaperInputs:
    """Everything link-and-build reads for one paper (`cells` and `pdf` only for drawn table cells)."""

    paper_id: str
    blocks: list[LayoutBlock]
    regions: dict[str, Region]
    voted: list[VotedRead]
    page_sizes: dict[int, tuple[float, float]] = field(default_factory=dict)
    cells: list[table_cells.Cell] = field(default_factory=list)
    pdf: Path | None = None

    def drawing_regions(self) -> list[Region]:
        """The page drawings: every region but those cut from table cells."""
        return [region for region in self.regions.values() if region.kind != table_cells.KIND]


def load_inputs(paper_dir: Path, pdf: Path | None = None) -> PaperInputs:
    """Read one paper's stage outputs; a missing required file raises FileNotFoundError."""
    blocks = read_jsonl(need(paper_dir / "layout.jsonl"), LayoutBlock)
    regions = read_jsonl(need(paper_dir / "regions.jsonl"), Region)
    voted = read_jsonl(need(paper_dir / "voted.jsonl"), VotedRead)
    pages_path = paper_dir / "pages.jsonl"
    pages = read_jsonl(pages_path, PageRecord) if pages_path.is_file() else []
    return PaperInputs(
        paper_id=paper_dir.name,
        blocks=blocks,
        regions={r.region_id: r for r in regions},
        voted=voted,
        page_sizes={p.page: (p.width_pt, p.height_pt) for p in pages},
        cells=table_cells.read_cells(paper_dir),
        pdf=None if pdf is None else Path(pdf),
    )


def _molecule(
    base: dict[str, Any], label: str | None, smiles: str, status: str, reasons: list[str], confidence: float
) -> Molecule:
    return Molecule(**base, label=label, smiles=smiles, status=status, reasons=list(reasons), confidence=confidence)


def _abstained(base: dict[str, Any], label: str | None, reasons: list[str]) -> Molecule:
    return Molecule(**base, label=label, smiles=None, status=ABSTAINED, reasons=list(reasons), confidence=0.0)


# --- the core of a table ----------------------------------------------------------------------


@dataclass(frozen=True)
class CoreChoice:
    """The core drawing paired with a table.

    `core` is its SMILES aligned to the table (slot i is the i-th variable), `unfilled` the labels
    of core slots no column fills, `problems` what sends the table's rows to review.
    """

    read: VotedRead | None
    core: str | None = None
    unfilled: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _box_gap(a: Sequence[float], b: Sequence[float]) -> float:
    dx = max(0.0, b[0] - a[2], a[0] - b[2])
    dy = max(0.0, b[1] - a[3], a[1] - b[3])
    return math.hypot(dx, dy)


def _core_candidates(inputs: PaperInputs, table: RGroupTable, fixes: Any = None) -> list[VotedRead]:
    """Drawings with numbered R slots on the table's page (nearest first), then the page before. A drawing's
    slot_labels alternatives (`vote.alternative_reads`) follow its own read. Attachment alternatives (a drawing the
    readers split on, an attachment point or a carbon) come after every drawn core: R-group drawings beside a table
    are read that way too, and the nearest of them must not stand in for the table's core."""
    labellings = {"attachments": False, "slot_labels": fix_enabled(fixes, "slot_labels")}
    attachments = {"attachments": fix_enabled(fixes, "attachments"), "slot_labels": False}
    ranked = []
    for read in inputs.voted:
        region = inputs.regions.get(read.region_id)
        if region is None or region.kind == table_cells.KIND:
            continue
        own = [read] if read.smiles and _SLOT_RE.search(read.smiles) else []
        tiers = ((0, [*own, *alternative_reads(read, labellings)]), (1, alternative_reads(read, attachments)))
        for tier, cores in tiers:
            cores = [core for core in cores if _SLOT_RE.search(core.smiles or "")]
            if region.page == table.page:
                ranked += [((tier, 0, _box_gap(region.bbox, table.bbox)), core) for core in cores]
            elif region.page == table.page - 1:
                ranked += [((tier, 1, -region.bbox[3]), core) for core in cores]
    return [read for _, read in sorted(ranked, key=lambda pair: pair[0])]


def _slot_labels(read: VotedRead, variables: Sequence[str]) -> dict[str, str]:
    """The read's slot labels; a lone unlabelled slot stands for a table's lone variable."""
    numbers = _SLOT_RE.findall(read.smiles or "")
    slots = {n: (read.slots or {}).get(n, "*") for n in numbers}
    if len(variables) == 1 and len(numbers) == 1 and enumeration.label_key(slots[numbers[0]]) in _UNLABELLED:
        return {numbers[0]: variables[0]}
    return slots


def _aligned(read: VotedRead, variables: Sequence[str]) -> CoreChoice | None:
    slots = _slot_labels(read, variables)
    core = enumeration.align_slots(read.smiles or "", slots, list(variables))
    if core is None:
        return None
    columns = {enumeration.label_key(name) for name in variables}
    unfilled = sorted({label for label in slots.values() if enumeration.label_key(label) not in columns})
    return CoreChoice(read, core, unfilled)


def _check_problem(check: enumeration.CoreCheck, choice: CoreChoice, variables: Sequence[str]) -> str:
    missing = [variables[n - 1] for n in sorted(check.missing)]
    return (
        f"core check failed: no slot for {', '.join(missing) or 'none'}; "
        f"slots no column fills: {', '.join(choice.unfilled) or 'none'}"
    )


def choose_core(inputs: PaperInputs, table: RGroupTable, fixes: Any) -> CoreChoice:
    """Pair `table` with a core; with the core check on, a core whose labels match the columns wins."""
    variables = list(table.variables)
    options = [c for read in _core_candidates(inputs, table, fixes) if (c := _aligned(read, variables)) is not None]
    if not options:
        return CoreChoice(read=None)
    choice, problems, agree = options[0], [], options[0].read.agree
    if fix_enabled(fixes, "core_check"):
        checked = [(option, enumeration.check_core(option.core, variables)) for option in options]
        choice, check = next(((o, k) for o, k in checked if k.ok), checked[0])
        if not check.ok:
            problems.append(_check_problem(check, choice, variables))
        agree = _core_agree(choice, checked, fixes)
    if fix_enabled(fixes, "vote") and agree < MIN_CORE_AGREE:
        problems.append(f"core read by {agree} agreeing reader(s), fewer than {MIN_CORE_AGREE}")
    return replace(choice, problems=problems)


def _core_agree(choice: CoreChoice, checked: Sequence[tuple[CoreChoice, enumeration.CoreCheck]], fixes: Any) -> int:
    """The readers agreeing on the chosen core. When they split on its slot labels (`skeleton_agree`, the
    slot_labels fix) and only one labelling of the drawing passes the check, every reader of the skeleton counts."""
    read = choice.read
    if not read.skeleton_agree or not fix_enabled(fixes, "slot_labels"):
        return read.agree
    passing = [option for option, check in checked if check.ok and option.read.region_id == read.region_id]
    return read.skeleton_agree if len(passing) == 1 and passing[0] is choice else read.agree


# --- table rows -------------------------------------------------------------------------------


def _fragments(table: RGroupTable, opsin: Opsin | None) -> dict[str, rgroups.Fragment | None]:
    """Every variable cell of the table read once, with all names sent to OPSIN in one batch."""
    cells = list(dict.fromkeys(row.get(var, "").strip() for row in table.rows for var in table.variables))
    cells = [cell for cell in cells if cell not in EMPTY_CELL and not cell.startswith("[image")]
    return dict(zip(cells, rgroups.to_fragments(cells, opsin), strict=True))


@dataclass(frozen=True)
class Legend:
    """A table value filled from the page drawing labelled with it (the legends fix): the code as the table prints
    it, the drawing, the fragment it gives and its vote."""

    code: str
    region_id: str
    fragment: rgroups.Fragment
    vote: VotedRead


@dataclass
class RowFill:
    """How one row fills its core: slot -> fragment, why a slot cannot be filled, what sends the row to review,
    notes for its reasons, the cell drawings and legends it uses, and a whole molecule drawn in its cell."""

    assignment: dict[int, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    review: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    used: list[table_cells.Cell] = field(default_factory=list)
    whole: VotedRead | None = None
    legends: list[Legend] = field(default_factory=list)


@dataclass(frozen=True)
class DrawnCells:
    """A table's drawn variable columns (variable -> each row's drawing) and the paper's votes by region."""

    columns: dict[str, table_cells.ColumnDrawings] = field(default_factory=dict)
    votes: dict[str, VotedRead] = field(default_factory=dict)
    vote_fix: bool = True


def _assignment(
    table: RGroupTable, index: int, choice: CoreChoice, fragments: Mapping[str, Any], drawn: DrawnCells,
    legends: Mapping[tuple[str, str], Legend] | None = None,
) -> RowFill:
    """How row `index` fills the core's slots; `legends` ((variable, value) -> legend) fills values no rule reads."""
    row = table.rows[index]
    present = {int(n) for n in _SLOT_RE.findall(choice.core or "")}
    legends = legends or {}
    fill = RowFill()
    for number, var in enumerate(table.variables, start=1):
        value = row.get(var, "").strip()
        column = drawn.columns.get(var)
        drawing = column.by_row.get(index) if column else None
        remnant = bool(_REMNANT.fullmatch(value))
        if table_cells.image_sources(value) or (drawing is not None and (value in EMPTY_CELL or remnant)):
            if number not in present:
                fill.problems.append(f"no slot in the core for {var}")
            else:
                _fill_drawn(fill, number, var, value, drawing, column, index, drawn, choice)
        elif value in EMPTY_CELL:
            why = column.problems.get(index) if column else None
            fill.problems.append(f"{var} cell is empty" + (f" ({why})" if why else ""))
        elif number not in present:
            fill.problems.append(f"no slot in the core for {var}")
        elif (fragment := fragments.get(value)) is None and (legend := legends.get((var, value))) is not None:
            _fill_legend(fill, number, var, value, legend, drawn.vote_fix)
        elif fragment is None:
            fill.problems.append(f"cannot read {var} = '{value}'")
        else:
            fill.assignment[number] = fragment
            if column is not None and remnant:
                fill.review.append(f"{var} = '{value}' sits in a column of drawings: it may be a letter of a "
                                   "drawing MinerU dropped")
    if choice.unfilled:
        fill.problems.append(f"core slot(s) {', '.join(choice.unfilled)} have no table column")
    return fill


def _fill_drawn(
    fill: RowFill, number: int, var: str, value: str, drawing: table_cells.Cell | None,
    column: table_cells.ColumnDrawings | None, index: int, drawn: DrawnCells, choice: CoreChoice,
) -> None:
    """Fill slot `number` from the drawing the cells stage cut for the row's `var` cell."""
    if drawing is None:
        why = column.problems.get(index) if column else None
        fill.problems.append(f"{var} is drawn in the table, but no drawing of it was found ({why or 'none was cut'})")
        return
    vote = drawn.votes.get(drawing.region_id)
    if vote is None or not vote.smiles:
        why = vote.reason if vote is not None else "no vote record"
        fill.problems.append(f"{var} is drawn in the table, but its drawing {drawing.region_id} gave no vote ({why})")
        return
    fill.used.append(drawing)
    count = attachments(vote.smiles)
    if count == 1 and (fragment := rgroups.drawn_fragment(vote.smiles)) is not None:
        fill.assignment[number] = fragment
    elif count == 0 and _heavy_atoms(vote.smiles) >= max(WHOLE_MIN_ATOMS, _heavy_atoms(choice.core)):
        fill.whole = vote
    else:
        fill.problems.append(f"{var} is drawn in the table, but its drawing {drawing.region_id} has no attachment "
                             f"point ({vote.smiles})")
        return
    if drawn.vote_fix and vote.agree < MIN_CORE_AGREE:
        fill.review.append(f"{var} drawing {drawing.region_id} read by {vote.agree} agreeing reader(s)")
    beside = table_cells.text_beside_images(value)
    if beside and not _REMNANT.fullmatch(beside):
        fill.review.append(f"{var} cell also holds the text '{beside}'")
    own = set(table_cells.image_sources(value))
    if drawing.src not in own:
        fill.notes.append(f"{var} drawing {drawing.region_id} placed in the row by its position")


def _heavy_atoms(smiles: str | None) -> int:
    """Atoms other than hydrogen and dummies."""
    with BlockLogs():
        mol = Chem.MolFromSmiles(smiles) if smiles else None
    return 0 if mol is None else sum(atom.GetAtomicNum() > 1 for atom in mol.GetAtoms())


# --- legends: R cells that name a fragment drawn elsewhere -----------------------------------------------


def _fill_legend(fill: RowFill, number: int, var: str, value: str, legend: Legend, vote_fix: bool) -> None:
    """Fill slot `number` from the drawing labelled with the row's `var` value (the legends fix)."""
    fill.assignment[number] = legend.fragment
    fill.legends.append(legend)
    if vote_fix and legend.vote.agree < MIN_CORE_AGREE:
        fill.review.append(f"{var} drawing {legend.region_id} read by {legend.vote.agree} agreeing reader(s)")
    fill.notes.append(f"{var} = {value} from drawing {legend.region_id}")


def _legend_key(text: str) -> str:
    """A table value or the text by a drawing in one spelling: trimmed as `labels.label_of_text` trims a label,
    then `enumerate.label_key` (no spaces or markup, case folded), so "(M0)", "M<sub>0</sub>" and "m0" are M0."""
    return enumeration.label_key(plain_text(text).strip("()[]").rstrip(".:;,").strip())


def _substituent_slots(core: str | None) -> set[int]:
    """The core's slot numbers whose every copy hangs off one atom: substituents, not linkers or ring atoms."""
    with BlockLogs():
        mol = Chem.MolFromSmiles(core) if core else None
    degrees: dict[int, set[int]] = {}
    for atom in mol.GetAtoms() if mol is not None else ():
        if atom.GetAtomicNum() == 0:
            degrees.setdefault(atom.GetAtomMapNum(), set()).add(atom.GetDegree())
    return {number for number, found in degrees.items() if found == {1}}


def _legend_fragment(vote: VotedRead | None) -> rgroups.Fragment | None:
    """The substituent a legend drawing gives: its one attachment point; for a plain molecule, the fragment
    `cell_vote.candidates` makes of its methyl (or radical) when it makes exactly one; else None. The chain and
    hydrogen candidates of the `attachments` fix are left out: every plain molecule has several."""
    if vote is None or not vote.smiles:
        return None
    count = attachments(vote.smiles)
    if count == 1:
        return rgroups.drawn_fragment(vote.smiles)
    plain = Read(vote.region_id, "vote", vote.smiles)
    found = candidates(plain, fixes={"attachments": False})[1] if count == 0 else {}
    return rgroups.drawn_fragment(next(iter(found.values()))) if len(found) == 1 else None


def _legend_rank(table: RGroupTable, region: Region) -> tuple[int, float, str]:
    """The table's page first (nearest the table), then the page before, then the page after."""
    gap = _box_gap(region.bbox, table.bbox) if region.page == table.page else 0.0
    return LEGEND_PAGES.index(region.page - table.page), gap, region.region_id


def _legends(
    inputs: PaperInputs, table: RGroupTable, choice: CoreChoice, fragments: Mapping[str, Any], ocr: Sequence[OcrLine]
) -> dict[tuple[str, str], Legend]:
    """(variable, value) -> its legend: each value of a substituent slot that no rule reads and that does not start
    with a digit, filled from the page drawing labelled with it (`_legend_rank` picks one) when that drawing gives a
    fragment (`_legend_fragment`)."""
    slots = _substituent_slots(choice.core)
    wanted = {
        (var, value)
        for row in table.rows
        for number, var in enumerate(table.variables, start=1)
        if number in slots and (value := row.get(var, "").strip()) in fragments and fragments[value] is None
        and not value[:1].isdigit()
    }
    codes = {key: value for _, value in sorted(wanted) if (key := _legend_key(value))}
    if not codes:
        return {}
    named = link_codes(inputs.drawing_regions(), inputs.blocks, lambda text: codes.get(_legend_key(text)),
                       inputs.page_sizes, ocr)
    votes = {read.region_id: read for read in inputs.voted}
    found = {}
    for key, code in codes.items():
        drawings = [inputs.regions[rid] for rid, label in named.items() if label == code]
        drawings = [region for region in drawings if region.page - table.page in LEGEND_PAGES]
        region = min(drawings, key=partial(_legend_rank, table), default=None)
        fragment = _legend_fragment(votes.get(region.region_id)) if region is not None else None
        if fragment is not None:
            found[key] = Legend(code, region.region_id, fragment, votes[region.region_id])
    return {(var, value): found[key] for var, value in wanted if (key := _legend_key(value)) in found}


def _row_molecules(
    paper_id: str, table: RGroupTable, index: int, choice: CoreChoice, fragments: Mapping[str, Any],
    drawn: DrawnCells | None = None, legends: Mapping[tuple[str, str], Legend] | None = None,
) -> tuple[list[Molecule], list[table_cells.Cell]]:
    """The record(s) for one table row (one per label when the row's label range was not split), and the cell
    drawings the row used; `legends` fills values that name a drawing (`_legends`)."""
    row = table.rows[index]
    printed = table.label_of(row)
    labels = cell_labels(printed)
    base = {"paper_id": paper_id, "page": table.page, "bbox": list(table.bbox), "source": "table",
            "table_id": table.table_id, "row": index}
    if len(labels) > 1:
        reason = f"label range {printed} not split: its cells do not hold one value per label"
        return [_abstained(base, label, [reason]) for label in labels], []
    label = labels[0] if labels else (printed or None)
    if choice.read is None:
        return [_abstained(base, label, [f"no R-group core drawing found for table {table.table_id}"])], []
    drawn = drawn or DrawnCells()
    fill = _assignment(table, index, choice, fragments, drawn, legends)
    if fill.whole is not None and not fill.problems:
        reasons = ["the R cell holds a whole molecule", *fill.review, *fill.notes]
        status = REVIEW if fill.review else ACCEPTED
        return [_molecule(base, label, fill.whole.smiles, status, reasons, fill.whole.confidence)], fill.used
    problems = fill.problems
    smiles = None if problems else enumeration.build(choice.core, fill.assignment)
    if not problems and not smiles:
        problems = ["RDKit could not build the molecule from the core and fragments"]
    if problems:
        return [_abstained(base, label, [*problems, *choice.problems])], fill.used
    status = REVIEW if choice.problems or fill.review else ACCEPTED
    confidence = min([choice.read.confidence, *(drawn.votes[cell.region_id].confidence for cell in fill.used),
                      *(legend.vote.confidence for legend in fill.legends)])
    reasons = [*choice.problems, *fill.review, *fill.notes]
    return [_molecule(base, label, smiles, status, reasons, confidence)], fill.used


@lru_cache(maxsize=16)
def _page_words(pdf: str, page: int) -> tuple[table_cells.Word, ...]:
    return tuple(table_cells.pdf_words(pdf, page))


def drawn_cells(inputs: PaperInputs, table: RGroupTable, fixes: Any) -> DrawnCells:
    """Each drawn variable column of a table (one with an image in any row) with the drawing of each row."""
    columns = {}
    labels = [found[0] if len(found := cell_labels(table.label_of(row))) == 1 else None for row in table.rows]
    words = None if inputs.pdf is None else (lambda: _page_words(str(inputs.pdf), table.page))
    for var in table.variables:
        values = [row.get(var, "") for row in table.rows]
        if any(table_cells.image_sources(value) for value in values):
            columns[var] = table_cells.column_drawings(values, table.table_id, table.bbox, labels, inputs.cells, words)
    votes = {read.region_id: read for read in inputs.voted} if columns else {}
    return DrawnCells(columns, votes, fix_enabled(fixes, "vote"))


def table_molecules(
    inputs: PaperInputs, table: RGroupTable, opsin: Opsin | None, fixes: Any, ocr: Sequence[OcrLine] = ()
) -> tuple[list[Molecule], str | None, list[tuple[table_cells.Cell, Molecule]], dict[str, str]]:
    """All records for one table, the region_id of the core used (None without a core), each cell drawing
    a row used with that row's record, and each page drawing used as a legend (region_id -> its code; the legends
    fix, which reads `ocr` for the codes printed in figures)."""
    choice = choose_core(inputs, table, fixes)
    fragments = _fragments(table, opsin) if choice.read is not None else {}
    legends_on = choice.read is not None and fix_enabled(fixes, "legends")
    legends = _legends(inputs, table, choice, fragments, ocr) if legends_on else {}
    drawn = drawn_cells(inputs, table, fixes) if choice.read is not None else DrawnCells()
    records: list[Molecule] = []
    used: list[tuple[table_cells.Cell, Molecule]] = []
    for i in range(len(table.rows)):
        molecules, cells = _row_molecules(inputs.paper_id, table, i, choice, fragments, drawn, legends)
        records += molecules
        used += [(cell, molecules[0]) for cell in cells]
    if len({m.row for m in records}) != len(table.rows):
        raise RuntimeError(f"table {table.table_id}: {len(table.rows)} rows but records for fewer rows")
    drawings = {legend.region_id: legend.code for legend in legends.values()}
    return records, (choice.read.region_id if choice.read else None), used, drawings


# --- drawings and names -----------------------------------------------------------------------


def _page_of(region_id: str) -> int | None:
    m = _REGION_PAGE_RE.match(region_id)
    return int(m[1]) if m else None


def _box_overlap(inner: Sequence[float], outer: Sequence[float]) -> float:
    """The share of box `inner` that lies inside box `outer`."""
    width = min(inner[2], outer[2]) - max(inner[0], outer[0])
    height = min(inner[3], outer[3]) - max(inner[1], outer[1])
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return width * height / area if width > 0 and height > 0 and area > 0 else 0.0


def covered_drawings(inputs: PaperInputs, used: Sequence[tuple[table_cells.Cell, Molecule]]) -> dict[str, str]:
    """Page drawing region_id -> why it is abstained: it covers a cell drawing a table row used."""
    covered = {}
    for region in inputs.drawing_regions():
        for cell, row in used:
            if cell.bbox is not None and cell.page == region.page and _box_overlap(region.bbox, cell.bbox) >= COVERED:
                covered[region.region_id] = (f"drawn in table {row.table_id} row {row.row} ({row.label}), "
                                             f"read there as {cell.region_id}")
                break
    return covered


def drawn_molecules(
    inputs: PaperInputs,
    links: Mapping[str, str],
    core_tables: Mapping[str, list[str]],
    covered: Mapping[str, str] | None = None,
    scheme_cores: Mapping[str, list[str]] | None = None,
) -> list[Molecule]:
    """One record per structure region: labelled molecules, cores and drawings without a vote. A drawing cut from a
    table cell gives none (its row holds it); a page drawing in `covered` is abstained with the reason given. A core
    in `scheme_cores` names the scheme compounds filled from it (`scheme_variants`)."""
    out = []
    for read in inputs.voted:
        region = inputs.regions.get(read.region_id)
        if region is not None and (not region.is_structure or region.kind == table_cells.KIND):
            continue
        base = {"paper_id": inputs.paper_id, "page": region.page if region else _page_of(read.region_id),
                "bbox": list(region.bbox) if region else None, "source": "drawn", "table_id": None, "row": None}
        label = links.get(read.region_id)
        missing = [] if region else ["region record missing"]
        used = core_tables.get(read.region_id)  # also a plain or tied read whose alternative core a table used
        schemes = (scheme_cores or {}).get(read.region_id)
        if covered and read.region_id in covered:
            out.append(_abstained(base, label, [covered[read.region_id], *missing]))
        elif used:
            out.append(_abstained(base, label, [f"R-group core for table {', '.join(used)}", *missing]))
        elif schemes:
            out.append(_abstained(base, label, [f"R-group core for scheme compounds {', '.join(schemes)}", *missing]))
        elif not read.smiles:
            reason = f"no vote: {read.reason or 'readers did not agree on a structure'}"
            out.append(_abstained(base, label, [reason, *missing]))
        elif "*" in read.smiles:
            out.append(_abstained(base, label, ["open R-group slots and no table uses this core", *missing]))
        else:
            out.append(_molecule(base, label, read.smiles, ACCEPTED, missing, read.confidence))
    return out


def _skeleton(smiles: str) -> str | None:
    """InChIKey connectivity block, None when RDKit cannot read the SMILES."""
    with BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToInchiKey(mol)[:14] if mol is not None else None


def _locate(name: str, blocks: Sequence[LayoutBlock]) -> tuple[int | None, list[float] | None]:
    for block in blocks:
        if block.type in NAME_TYPES and name in plain_text(block.text or ""):
            return block.page, list(block.bbox)
    return None, None


def name_molecules(inputs: PaperInputs, opsin: Opsin | None, known: Mapping[str, list[str]]) -> list[Molecule]:
    """Molecules from systematic names via OPSIN (one batch). A name that contradicts a drawn or
    table molecule with the same label goes to review; one that repeats it is kept for finalize's dedupe."""
    pairs = named_compounds(inputs.blocks)
    if not pairs:
        return []
    parsed = opsin.to_smiles([name for _, name in pairs]) if opsin is not None else [None] * len(pairs)
    if len(parsed) != len(pairs):
        raise RuntimeError(f"OPSIN returned {len(parsed)} results for {len(pairs)} names")
    out = []
    for (label, name), smiles in zip(pairs, parsed, strict=True):
        page, bbox = _locate(name, inputs.blocks)
        base = {"paper_id": inputs.paper_id, "page": page, "bbox": bbox, "source": "name", "table_id": None, "row": None}
        if not smiles:
            reason = "OPSIN not available" if opsin is None else f"OPSIN could not parse '{name}'"
            out.append(_abstained(base, label, [reason]))
            continue
        others = known.get(label, [])
        if others and _skeleton(smiles) not in {_skeleton(s) for s in others}:
            reasons = [f"name differs from the drawn or table structure for {label}"]
            out.append(_molecule(base, label, smiles, REVIEW, reasons, NAME_CONFIDENCE))
        else:
            out.append(_molecule(base, label, smiles, ACCEPTED, [], NAME_CONFIDENCE))
    return out


def _known_structures(molecules: Sequence[Molecule]) -> dict[str, list[str]]:
    known: dict[str, list[str]] = {}
    for m in molecules:
        if m.label and m.smiles and m.status != ABSTAINED:
            known.setdefault(m.label, []).append(m.smiles)
    return known


def _rgroup_rows(
    inputs: PaperInputs, opsin: Opsin | None, fixes: Any, ocr: Sequence[OcrLine] = ()
) -> tuple[list[Molecule], dict[str, list[str]], list[tuple[table_cells.Cell, Molecule]], dict[str, str]]:
    """The records of every R-group table, the tables each core drawing fills, the cell drawings rows used, and
    why each page drawing used as a legend is abstained (region_id -> reason)."""
    records: list[Molecule] = []
    core_tables: dict[str, list[str]] = {}
    used: list[tuple[table_cells.Cell, Molecule]] = []
    legend_tables: dict[str, tuple[str, list[str]]] = {}
    for table in find_rgroup_tables(inputs.blocks) if fix_enabled(fixes, "tables") else []:
        built, core_id, cells, legends = table_molecules(inputs, table, opsin, fixes, ocr)
        records.extend(built)
        used.extend(cells)
        if core_id:
            core_tables.setdefault(core_id, []).append(table.table_id)
        for region_id, code in legends.items():
            legend_tables.setdefault(region_id, (code, []))[1].append(table.table_id)
    reasons = {rid: f"legend fragment {code} for table {', '.join(ids)}" for rid, (code, ids) in legend_tables.items()}
    return records, core_tables, used, reasons


def _compound_rows(inputs: PaperInputs, fixes: Any) -> tuple[list[Molecule], list[tuple[table_cells.Cell, Molecule]]]:
    """The records of every whole-compound table and the cell drawings its built rows used."""
    found = find_compound_tables(inputs.blocks) if fix_enabled(fixes, "tables") else []
    votes = {read.region_id: read for read in inputs.voted} if found else {}
    records: list[Molecule] = []
    used: list[tuple[table_cells.Cell, Molecule]] = []
    for table in found:
        words = None if inputs.pdf is None else partial(_page_words, str(inputs.pdf), table.table.page)
        built, cells = compound_molecules(inputs.paper_id, table, inputs.cells, votes, words, fix_enabled(fixes, "vote"))
        records.extend(built)
        used.extend(cells)
    return records, used


def link_counts(
    molecules: Sequence[Molecule], compound: Sequence[Molecule], schemes: Sequence[Molecule] = (), legends: int = 0
) -> dict[str, int]:
    """`in` records = `out` (with a SMILES) + `dropped_<source>` (abstained); `whole_compound_rows` counts the
    records of whole-compound tables (`compound`), `whole_compound_molecules` those with a SMILES,
    `scheme_variants` the records filled from scheme definitions (`schemes`) and `legend_drawings` the page drawings
    used as legend fragments (`legends`)."""
    counts = {"in": len(molecules), "out": sum(bool(m.smiles) for m in molecules)}
    for source in ("drawn", "table", "name"):
        counts[f"dropped_{source}"] = sum(m.source == source and not m.smiles for m in molecules)
    counts["whole_compound_rows"] = len(compound)
    counts["whole_compound_molecules"] = sum(bool(m.smiles) for m in compound)
    counts["scheme_variants"] = len(schemes)
    counts["legend_drawings"] = legends
    return counts


def link_paper(
    paper_dir: Path | str, opsin: Opsin | None, fixes: Any = None, pdf: Path | str | None = None
) -> tuple[list[Molecule], dict[str, int]]:
    """All molecules of one paper and their counts (`link_counts`): drawn, then scheme variants, then R-group table
    rows, then whole-compound table rows, then named compounds. `pdf` (the paper's PDF) places the rows of tables
    with drawn cells; without it each row reads only the image in its own cell. A page drawing used as a legend
    fragment (the legends fix) is abstained, as is one that covers a cell drawing a row used."""
    inputs = load_inputs(Path(paper_dir), None if pdf is None else Path(pdf))
    ocr = load_ocr_lines(paper_dir)  # figure text from the ocr_labels stage, when it ran
    links = {}
    if fix_enabled(fixes, "labels"):
        links = link_regions(inputs.drawing_regions(), inputs.blocks, inputs.page_sizes, ocr)
    table_records, core_tables, used, legends = _rgroup_rows(inputs, opsin, fixes, ocr)
    schemes, scheme_cores = scheme_molecules(inputs, ocr, opsin, core_tables, fixes) if fix_enabled(fixes, "tables") \
        else ([], {})
    compound, compound_used = _compound_rows(inputs, fixes)
    covered = {**legends, **covered_drawings(inputs, [*used, *compound_used])}
    drawn = drawn_molecules(inputs, links, core_tables, covered, scheme_cores)
    known = _known_structures([*drawn, *schemes, *table_records, *compound])
    named = name_molecules(inputs, opsin, known) if fix_enabled(fixes, "names") else []
    molecules = [*drawn, *schemes, *table_records, *compound, *named]
    return molecules, link_counts(molecules, compound, schemes, len(legends))


def link_and_build(
    paper_dir: Path | str, opsin: Opsin | None, fixes: Any = None, pdf: Path | str | None = None
) -> list[Molecule]:
    """`link_paper`'s molecules without the counts."""
    return link_paper(paper_dir, opsin, fixes, pdf)[0]
