"""Whole-compound tables: a label column beside a column of drawn whole compounds, one molecule per row.

Some tables print each compound whole in a "Structure" column, beside its label and often its activity, with no
R-group core. A labelled table (tables.labelled_tables) without variable columns is one when a column other than
the label and activity columns holds MinerU's images for at least DRAWN_SHARE of its compounds. A compound whose
label cell spans two grid rows is one row: a row that repeats the label above it and adds text only where the
rows above have none continues them. A row without a label or an image is not a compound.

Each compound gives one record, source "table", from the vote of the drawing the cells stage cut for it
(`table_cells.column_drawings`, which places a drawing by the row it sits in when the PDF's text places the
rows):

- a whole molecule is the row's molecule: accepted, or review when fewer than MIN_AGREE readers agree (with the
  vote fix). Finalize then applies the relevance rule; a row of a table with activity columns names its
  label as tested;
- a fragment with an attachment point, a drawing without a vote, or no drawing abstains with the reason;
- a label range that was not split abstains once per label: one drawing cannot be several compounds;
- when more than FRAGMENT_SHARE of the column's voted drawings are fragments, the column holds R groups under a
  header that was not read as a variable, and every row abstains with that reason.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from molscout.pipeline.cell_vote import attachments
from molscout.pipeline.schema import LayoutBlock, Molecule, VotedRead
from molscout.pipeline.table_cells import Cell, Word, column_drawings, image_sources
from molscout.pipeline.tables import RGroupTable, cell_labels, labelled_tables

DRAWN_SHARE = 0.5  # a column of drawings holds an image for at least this share of the compounds
FRAGMENT_SHARE = 0.5  # more voted drawings than this share are fragments: the column holds R groups
MIN_AGREE = 2
ACCEPTED, REVIEW, ABSTAINED = "accepted", "review", "abstained"


@dataclass(frozen=True)
class CompoundRow:
    """One compound: its first row in `RGroupTable.rows`, its label as printed and its drawing cell's text."""

    index: int
    label: str
    drawing: str


@dataclass(frozen=True)
class CompoundTable:
    """A whole-compound table, the header of its column of drawings and its compounds."""

    table: RGroupTable
    column: str
    rows: list[CompoundRow]


def find_compound_tables(blocks: Sequence[LayoutBlock]) -> list[CompoundTable]:
    """The labelled tables without variable columns that have a column of drawings."""
    found = []
    for table in labelled_tables(blocks):
        if table.variables:
            continue
        merged = [(index, row) for index, row in compound_rows(table) if table.label_of(row) or _has_image(row)]
        column = drawing_column(table, [row for _, row in merged])
        if column is not None:
            rows = [CompoundRow(index, table.label_of(row), row[column]) for index, row in merged]
            found.append(CompoundTable(table, column, rows))
    return found


def compound_rows(table: RGroupTable) -> list[tuple[int, dict[str, str]]]:
    """Each compound's first row index and its cells, a compound printed over several grid rows joined."""
    out: list[tuple[int, dict[str, str]]] = []
    for index, row in enumerate(table.rows):
        label = table.label_of(row)
        if out and label and label == table.label_of(out[-1][1]) and _fits(out[-1][1], row):
            out[-1] = (out[-1][0], {key: value or row[key] for key, value in out[-1][1].items()})
        else:
            out.append((index, dict(row)))
    return out


def _fits(above: Mapping[str, str], row: Mapping[str, str]) -> bool:
    """True when `row` has text only where `above` has none or the same."""
    return all(not value or not above.get(key) or above[key] == value for key, value in row.items())


def _has_image(row: Mapping[str, str]) -> bool:
    return any(image_sources(value) for value in row.values())


def drawing_column(table: RGroupTable, rows: Sequence[Mapping[str, str]]) -> str | None:
    """The column that holds an image for the most compounds, when that is at least DRAWN_SHARE of them; never the
    label or an activity column."""
    taken = {table.headers[table.label_col], *table.activity_cols} if table.headers else set()
    drawn = {header: sum(bool(image_sources(row.get(header, ""))) for row in rows)
             for header in table.headers if header not in taken}
    best = max(drawn, key=lambda header: drawn[header], default=None)
    return best if best is not None and drawn[best] and drawn[best] >= DRAWN_SHARE * len(rows) else None


# --- one record per compound -----------------------------------------------------------------------


def compound_molecules(
    paper_id: str,
    found: CompoundTable,
    cells: Sequence[Cell],
    votes: Mapping[str, VotedRead],
    words: Callable[[], Sequence[Word]] | None = None,
    vote_fix: bool = True,
) -> tuple[list[Molecule], list[tuple[Cell, Molecule]]]:
    """The records of one whole-compound table, and each drawing a built row used with that row's record.

    `cells` are the cells stage's drawings and `votes` the paper's votes by region; `words()` gives the PDF words
    of the table's page (None: no PDF).
    """
    table = found.table
    labels = [row.label or None for row in found.rows]
    drawings = column_drawings([row.drawing for row in found.rows], table.table_id, table.bbox, labels, cells, words)
    records: list[Molecule] = []
    used: list[tuple[Cell, Molecule]] = []
    voted = [vote.smiles for cell in drawings.by_row.values() if (vote := votes.get(cell.region_id)) and vote.smiles]
    fragments = sum(bool(attachments(smiles)) for smiles in voted)
    r_groups = bool(voted) and fragments > FRAGMENT_SHARE * len(voted)
    for k, row in enumerate(found.rows):
        base = {"paper_id": paper_id, "page": table.page, "bbox": list(table.bbox), "source": "table",
                "table_id": table.table_id, "row": row.index}
        labels = cell_labels(row.label)
        if r_groups:
            reason = (f"column {found.column} holds R-group fragments ({fragments} of {len(voted)} voted drawings), "
                      "not whole compounds")
            records += [_abstained(base, label, reason) for label in labels or [row.label or None]]
            continue
        cell = drawings.by_row.get(k)
        if len(labels) > 1:
            reason = f"label range {row.label} not split: one drawing for {len(labels)} labels"
            records += [_abstained(base, label, reason) for label in labels]
            continue
        label = labels[0] if labels else (row.label or None)
        if cell is None:
            why = drawings.problems.get(k)
            records.append(_abstained(base, label, f"no drawing in column {found.column}" + (f" ({why})" if why else "")))
            continue
        records.append(_row_molecule(base, label, row, found.column, cell, votes.get(cell.region_id), vote_fix))
        if records[-1].smiles:
            used.append((cell, records[-1]))
    return records, used


def _row_molecule(
    base: dict[str, Any], label: str | None, row: CompoundRow, column: str, cell: Cell, vote: VotedRead | None,
    vote_fix: bool,
) -> Molecule:
    """The record of a row from the vote of its drawing `cell`."""
    where = f"drawing {cell.region_id}"
    if vote is None or not vote.smiles:
        return _abstained(base, label, f"{where} gave no vote ({vote.reason if vote is not None else 'no vote record'})")
    if attachments(vote.smiles):
        return _abstained(base, label, f"{where} is a fragment with an attachment point ({vote.smiles}), "
                                       "not a whole compound")
    review = [f"{where} read by {vote.agree} agreeing reader(s)"] if vote_fix and vote.agree < MIN_AGREE else []
    placed = [f"{where} placed in the row by its position"] if cell.src not in image_sources(row.drawing) else []
    reasons = [f"drawn whole in column {column} ({cell.region_id})", *review, *placed]
    return Molecule(**base, label=label, smiles=vote.smiles, status=REVIEW if review else ACCEPTED, reasons=reasons,
                    confidence=vote.confidence)


def _abstained(base: dict[str, Any], label: str | None, reason: str) -> Molecule:
    return Molecule(**base, label=label, smiles=None, status=ABSTAINED, reasons=[reason], confidence=0.0)
