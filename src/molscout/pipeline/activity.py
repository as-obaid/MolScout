"""Activity: whether a paper reports biological activity, and which compound labels were tested.

A label counts as tested when it heads a table row with a value in an activity column, or when a
sentence names it next to an activity value ("12a (IC50 = 14 nM)", "compound 7 ... Ki of 3 nM").
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from molscout.pipeline.labels import (
    LABEL_LIST_PATTERN,
    LABEL_PATTERN,
    clean_text,
    expand_range,
    mentioned_labels,
)
from molscout.pipeline.tables import (
    ACTIVITY_RE,
    RGroupTable,
    cell_labels,
    labelled_tables,
    merged_table_grids,
)

if TYPE_CHECKING:
    from molscout.pipeline.schema import LayoutBlock

TEXT_TYPES = {"text", "title", "caption", "footnote"}
UNTESTED = {"", "nt", "n.t.", "not tested", "–", "—", "-", "−", "n/a", "na", "n.a."}
VALUE_WINDOW = 60
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(*\d])")
_TERM = ACTIVITY_RE.pattern
_VALUE_RE = re.compile(rf"(?:{_TERM})[^.;]{{0,{VALUE_WINDOW}}}?\d")
_VERBS = (
    r"were|was|is|are|showed|shows|show|inhibited|inhibits|exhibited|exhibits|displayed|had|has"
    r"|gave|retained|lost|demonstrated"
)
_LEADING_LABELS_RE = re.compile(rf"^(?P<labels>{LABEL_LIST_PATTERN})\s+(?:{_VERBS})\b")
_LABEL_THEN_VALUE_RE = re.compile(
    rf"(?<![\w.'])(?P<a>{LABEL_PATTERN})\s*\(\s*(?:{_TERM})"
    rf"|\(\s*(?P<b>{LABEL_PATTERN})\s*[,;:]\s*(?:{_TERM})"
)


def activity_sentences(text: str) -> list[str]:
    """Sentences of `text` that hold an activity term followed by a value."""
    sentences = _SENTENCE_SPLIT_RE.split(clean_text(text))
    return [s for s in sentences if _VALUE_RE.search(s)]


def _sentence_labels(sentence: str) -> list[str]:
    """Labels a sentence ties to its activity value."""
    labels = mentioned_labels(sentence)
    leading = _LEADING_LABELS_RE.match(sentence)
    if leading:
        labels.extend(expand_range(leading["labels"]))
    for m in _LABEL_THEN_VALUE_RE.finditer(sentence):
        labels.append(m["a"] or m["b"])
    return labels


def _text_blocks(blocks: Iterable[LayoutBlock]) -> list[str]:
    return [b.text or "" for b in blocks if b.type in TEXT_TYPES]


def reports_activity(blocks: Sequence[LayoutBlock]) -> bool:
    """True when a table header names an activity measure or a sentence gives an activity value."""
    for _, grid in merged_table_grids(blocks):
        if any(ACTIVITY_RE.search(cell) for row in grid[:3] for cell in row):
            return True
    return any(activity_sentences(t) for t in _text_blocks(blocks))


def _table_labels(table: RGroupTable) -> set[str]:
    """Labels of rows with at least one tested value in an activity column."""
    tested: set[str] = set()
    for row in table.rows:
        if any(row.get(col, "").strip().lower() not in UNTESTED for col in table.activity_cols):
            tested.update(cell_labels(table.label_of(row)))
    return tested


def tested_labels(blocks: Sequence[LayoutBlock], tables: Sequence[RGroupTable]) -> set[str]:
    """Labels that appear in an activity table row or in a sentence with an activity value."""
    by_id = {t.table_id: t for t in labelled_tables(blocks)}
    by_id.update({t.table_id: t for t in tables})
    tested: set[str] = set()
    for table in by_id.values():
        tested |= _table_labels(table)
    for body in _text_blocks(blocks):
        for sentence in activity_sentences(body):
            tested.update(_sentence_labels(sentence))
    return tested
