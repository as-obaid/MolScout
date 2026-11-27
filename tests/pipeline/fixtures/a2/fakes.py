"""Small stand-ins for the A1 schema records, used by the document-logic tests.

The document modules read records by attribute only, so these fakes carry the same field names as
`molscout.pipeline.schema` without importing it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Block:
    """A layout block with the fields of `schema.LayoutBlock`."""

    page: int
    index: int
    type: str
    bbox: list[float]
    text: str = ""
    html: str | None = None
    parent: int | None = None
    image: str | None = None


@dataclass(frozen=True)
class Region:
    """A drawing region with the fields of `schema.Region`."""

    region_id: str
    page: int
    bbox: list[float]
    crop: str | None = None
    structure_score: float = 0.95
    is_structure: bool = True


def text(page: int, index: int, bbox: list[float], body: str, kind: str = "text") -> Block:
    """A text-like block."""
    return Block(page=page, index=index, type=kind, bbox=bbox, text=body)


def table(page: int, index: int, html: str, bbox: list[float] | None = None) -> Block:
    """A table block holding `html`."""
    return Block(page=page, index=index, type="table", bbox=bbox or [50, 300, 550, 600], html=html)


def html_table(rows: list[list[str]], header_rows: int = 1) -> str:
    """Build plain table HTML from rows of cell HTML."""
    out = []
    for i, row in enumerate(rows):
        tag = "th" if i < header_rows else "td"
        out.append("<tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in row) + "</tr>")
    return "<table>" + "".join(out) + "</table>"
