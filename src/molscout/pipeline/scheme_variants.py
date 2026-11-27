"""Scheme variants: a drawn core filled from the R definitions printed beside it in its figure.

A scheme often draws one core with open R slots and prints each compound as its label and R values inside
the figure: "20, R = Boc", "R= Ph, 16b", "25 (R = H)" (`parse_definition`). Each such line the label OCR read
(ocr.jsonl) fills the nearest drawn core on its page whose slot labels are exactly the line's variables and
that no table uses, and gives one record, source "drawn", labelled as the line says. A drawing's alternative
cores (`VotedRead.alternatives`) are looked up after its own read, as for tables:

- accepted when every value reads (`rgroups.to_fragments`, names through OPSIN) and the core builds;
- review when fewer than MIN_CORE_AGREE readers agree on the core (with the vote fix);
- abstained with the reason when a value cannot be read or the core does not build.

A line names one label and at least one value; one with a label range, two labels, or "or"/"and" in a value
gives no record, nor does a line with no core to fill.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from molscout.pipeline import enumerate as enumeration
from molscout.pipeline import rgroups, table_cells
from molscout.pipeline.schema import Molecule, OcrLine, VotedRead, fix_enabled
from molscout.pipeline.vote import alternative_reads

if TYPE_CHECKING:
    from molscout.pipeline.link import PaperInputs
    from molscout.pipeline.opsin import Opsin

MIN_CORE_AGREE = 2
ACCEPTED, REVIEW, ABSTAINED = "accepted", "review", "abstained"
_LABEL = r"\d{1,3}[a-z]{0,2}"
_VARIABLE = r"R\d{0,2}'{0,2}|X|Y|Z|Ar"
_WRAPPED = re.compile(rf"(?P<label>{_LABEL})\s*[,:]?\s*\((?P<inner>.*=.*)\)")
_TOKEN = re.compile(rf"(?:(?P<label>{_LABEL})\s*[,:]?\s+)?(?P<var>{_VARIABLE})\s*=\s*(?P<value>\S.*)")
_JOINED = re.compile(r"\b(?:or|and)\b")


@dataclass(frozen=True)
class Definition:
    """One compound of a scheme: its label and the value of each variable, as printed."""

    label: str
    values: dict[str, str]


def parse_definition(text: str) -> Definition | None:
    """The compound a figure line defines ("20, R = Boc", "R= Ph, 16b", "25 (R = H)"), or None."""
    text = text.strip()
    wrapped = _WRAPPED.fullmatch(text)
    if wrapped:
        text = f"{wrapped['label']}, {wrapped['inner']}"
    label, values = None, {}
    for token in (part.strip() for part in _split(text)):
        match = _TOKEN.fullmatch(token)
        if match is None and re.fullmatch(_LABEL, token) and label is None:
            label = token
        elif match is not None and not (match["label"] and label):
            label = match["label"] or label
            values[match["var"]] = match["value"].strip()
        else:
            return None
    if label is None or not values or any(_JOINED.search(value) for value in values.values()):
        return None
    return Definition(label, values)


def _split(text: str) -> list[str]:
    """`text` split at commas and semicolons outside brackets."""
    parts, depth, current = [], 0, ""
    for char in text:
        depth += (char in "([") - (char in ")]")
        if char in ",;" and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    return [*parts, current]


def scheme_molecules(
    inputs: PaperInputs, lines: Sequence[OcrLine], opsin: Opsin | None, used: Mapping[str, Any], fixes: Any = None,
) -> tuple[list[Molecule], dict[str, list[str]]]:
    """The records of the paper's scheme variants, and each filled core's region_id with the labels it gave.

    `used` names the cores tables use (region_id -> anything); those are not filled. A drawing's alternative cores
    (`vote.alternative_reads`, with the attachments and slot_labels fixes) are looked up after its own read.
    """
    cores = [core for read in inputs.voted for core in _open_cores(read, inputs, used, fixes)]
    found = [(line, definition) for line in lines if (definition := parse_definition(line.text))]
    pairs = [(line, definition, core) for line, definition in found if (core := _nearest(line, definition, cores, inputs))]
    values = list(dict.fromkeys(value for _, definition, _ in pairs for value in definition.values.values()))
    fragments = dict(zip(values, rgroups.to_fragments(values, opsin), strict=True))
    molecules, filled = [], {}
    for line, definition, core in pairs:
        molecules.append(_variant(inputs, line, definition, core, fragments, fixes))
        filled.setdefault(core.region_id, []).append(definition.label)
    return molecules, filled


def _open_cores(read: VotedRead, inputs: PaperInputs, used: Mapping[str, Any], fixes: Any) -> list[VotedRead]:
    """The read and its alternative cores, those with labelled slots, unless a table uses the drawing."""
    region = inputs.regions.get(read.region_id)
    if region is None or region.kind == table_cells.KIND or read.region_id in used:
        return []
    return [core for core in [read, *alternative_reads(read, fixes)] if "*" in (core.smiles or "") and core.slots]


def _nearest(line: OcrLine, definition: Definition, cores: Sequence[VotedRead], inputs: PaperInputs) -> VotedRead | None:
    """The core on the line's page nearest to it whose slot labels are the definition's variables."""
    wanted = {enumeration.label_key(name) for name in definition.values}
    fits = [core for core in cores if inputs.regions[core.region_id].page == line.page
            and {enumeration.label_key(label) for label in core.slots.values()} == wanted]
    return min(fits, key=lambda core: _gap(inputs.regions[core.region_id].bbox, line.bbox), default=None)


def _gap(a: Sequence[float], b: Sequence[float]) -> float:
    dx = max(0.0, b[0] - a[2], a[0] - b[2])
    dy = max(0.0, b[1] - a[3], a[1] - b[3])
    return math.hypot(dx, dy)


def _variant(
    inputs: PaperInputs, line: OcrLine, definition: Definition, core: VotedRead,
    fragments: Mapping[str, rgroups.Fragment | None], fixes: Any,
) -> Molecule:
    region = inputs.regions[core.region_id]
    base = {"paper_id": inputs.paper_id, "label": definition.label, "page": region.page, "bbox": list(region.bbox),
            "source": "drawn", "table_id": None, "row": None}
    reasons = [f"core {core.region_id} filled from '{line.text.strip()}'"]
    variables = list(definition.values)
    unread = [f"cannot read {var} = '{definition.values[var]}'" for var in variables
              if fragments.get(definition.values[var]) is None]
    aligned = None if unread else enumeration.align_slots(core.smiles or "", core.slots, variables)
    smiles = None if aligned is None else enumeration.build(
        aligned, {number: fragments[definition.values[var]] for number, var in enumerate(variables, start=1)})
    if smiles is None:
        return Molecule(**base, smiles=None, status=ABSTAINED, reasons=[*reasons, *(unread or ["the core does not build"])])
    if fix_enabled(fixes, "vote") and core.agree < MIN_CORE_AGREE:
        reasons.append(f"core read by {core.agree} agreeing reader(s), fewer than {MIN_CORE_AGREE}")
        return Molecule(**base, smiles=smiles, status=REVIEW, reasons=reasons, confidence=core.confidence)
    return Molecule(**base, smiles=smiles, status=ACCEPTED, reasons=reasons, confidence=core.confidence)
