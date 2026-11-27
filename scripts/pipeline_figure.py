"""Draw the pipeline figure of docs/pipeline.md: a paper PDF to molecules.csv in five stages.

    uv run python scripts/pipeline_figure.py
    uv run python scripts/pipeline_figure.py --out docs/figures

Writes pipeline-light.svg (white background) and pipeline-dark.svg (GitHub dark, #0d1117). Text and
tables (Parse) and drawings (Find drawings, Read and vote) run side by side and meet at Link and build.
Standard library only; the SVGs hold shapes and text, nothing else.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

REPO = Path(__file__).resolve().parents[1]
FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif'

# The stages as docs/pipeline.md names them: (name, tool or output file, box width). Parse spans the drawing lane.
PARSE = ("Parse", "MinerU 4.2.3")
DRAWINGS = (("Find drawings", "DECIMER Segmentation", 160), ("Read and vote", "MolGlyph, MolScribe, MolNexTR", 206))
JOINED = (("Link and build", "OPSIN + RDKit or BioMiner-Instruct 32B", 248), ("Check and export", "molecules.csv", 152))
STAGES = (PARSE[0], *(name for name, _, _ in DRAWINGS), *(name for name, _, _ in JOINED))

# Even integers, so every coordinate is a whole number.
MARGIN = 4
BOX_H, LANE_GAP, RADIUS = 56, 16, 8
INPUT_W, INPUT_H = 46, 34
FORK, STEP, MERGE = 32, 22, 32  # the gaps before the lanes, between stages and before the merge
TURN = 6  # corner radius of the connectors
HEAD = 7  # arrowhead length
NAME_SIZE, NOTE_SIZE = 15, 12


@dataclass(frozen=True)
class Theme:
    background: str
    panel: str
    border: str
    ink: str
    muted: str
    line: str
    accent: str


THEMES = {
    "light": Theme("#ffffff", "#ffffff", "#d0d7de", "#1f2328", "#59636e", "#8c959f", "#2370cd"),
    "dark": Theme("#0d1117", "#151b23", "#3d444d", "#e6edf3", "#9198a1", "#6e7681", "#58a6ff"),
}


@dataclass(frozen=True)
class Box:
    name: str
    note: str
    x: int
    y: int
    width: int
    accent: bool = False

    @property
    def right(self) -> int:
        return self.x + self.width


@dataclass(frozen=True)
class Layout:
    boxes: list[Box]
    paths: list[str]
    tips: list[tuple[int, int]]  # arrowheads, pointing right
    middle: int
    width: int
    height: int


def layout() -> Layout:
    """Where every box, connector and arrowhead goes, in SVG units."""
    top, bottom = MARGIN + BOX_H // 2, MARGIN + BOX_H * 3 // 2 + LANE_GAP
    middle = (top + bottom) // 2
    lanes_x = MARGIN + INPUT_W + FORK
    lower, x = [], lanes_x
    for name, note, width in DRAWINGS:
        lower.append(Box(name, note, x, bottom - BOX_H // 2, width))
        x += width + STEP
    parse = Box(*PARSE, lanes_x, MARGIN, lower[-1].right - lanes_x)
    joined, x = [], lower[-1].right + MERGE
    for name, note, width in JOINED:
        joined.append(Box(name, note, x, middle - BOX_H // 2, width, accent=name == JOINED[-1][0]))
        x += width + STEP
    fork, merge = MARGIN + INPUT_W + FORK // 2 - TURN, lower[-1].right + MERGE // 2 - TURN
    paths = [
        *(_elbow(MARGIN + INPUT_W, middle, fork, lane, lanes_x - HEAD + 1) for lane in (top, bottom)),
        *(f"M{a.right} {bottom}H{b.x - HEAD + 1}" for a, b in zip(lower, lower[1:])),
        *(_elbow(lower[-1].right, lane, merge, middle, joined[0].x - HEAD + 1) for lane in (top, bottom)),
        *(f"M{a.right} {middle}H{b.x - HEAD + 1}" for a, b in zip(joined, joined[1:])),
    ]
    tips = [(lanes_x, top), *((box.x, bottom) for box in lower), *((box.x, middle) for box in joined)]
    return Layout([parse, *lower, *joined], paths, tips, middle, joined[-1].right + MARGIN, bottom + BOX_H // 2 + MARGIN)


def _elbow(x0: int, y0: int, turn: int, y1: int, x1: int) -> str:
    """Right from (x0, y0) to `turn`, up or down to y1 with rounded corners, then right to x1."""
    step = TURN if y1 > y0 else -TURN
    return f"M{x0} {y0}H{turn - TURN}Q{turn} {y0} {turn} {y0 + step}V{y1 - step}Q{turn} {y1} {turn + TURN} {y1}H{x1}"


def svg(theme_name: str) -> str:
    """The figure as SVG text in one theme ("light" or "dark")."""
    theme, plan = THEMES[theme_name], layout()
    lines = "".join(f'<path d="{d}"/>' for d in plan.paths)
    heads = "".join(f'<path d="M{x} {y}l-{HEAD} -{HEAD / 2}v{HEAD}z"/>' for x, y in plan.tips)
    pdf = (
        f'<rect x="{MARGIN}" y="{plan.middle - INPUT_H // 2}" width="{INPUT_W}" height="{INPUT_H}" rx="6" '
        f'fill="{theme.panel}" stroke="{theme.border}"/>'
        f'<text x="{MARGIN + INPUT_W // 2}" y="{plan.middle + 5}" font-size="13" font-weight="600" '
        f'fill="{theme.ink}" text-anchor="middle">PDF</text>'
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{plan.width}" height="{plan.height}" '
        f"viewBox=\"0 0 {plan.width} {plan.height}\" font-family='{FONT}'>"
        f'<rect width="{plan.width}" height="{plan.height}" fill="{theme.background}"/>'
        f'<g fill="none" stroke="{theme.line}" stroke-width="1.25" stroke-linecap="round">{lines}</g>'
        f'<g fill="{theme.line}">{heads}</g>{pdf}{"".join(_box(box, theme) for box in plan.boxes)}</svg>\n'
    )


def _box(box: Box, theme: Theme) -> str:
    stroke = f'stroke="{theme.accent}" stroke-width="1.5"' if box.accent else f'stroke="{theme.border}"'
    centre = box.x + box.width // 2
    return (
        f'<rect x="{box.x}" y="{box.y}" width="{box.width}" height="{BOX_H}" rx="{RADIUS}" '
        f'fill="{theme.panel}" {stroke}/>'
        f'<text x="{centre}" y="{box.y + 23}" font-size="{NAME_SIZE}" font-weight="600" fill="{theme.ink}" '
        f'text-anchor="middle">{escape(box.name)}</text>'
        f'<text x="{centre}" y="{box.y + 41}" font-size="{NOTE_SIZE}" '
        f'fill="{theme.accent if box.accent else theme.muted}" text-anchor="middle">{escape(box.note)}</text>'
    )


def write_figures(out: Path) -> list[Path]:
    """Write pipeline-light.svg and pipeline-dark.svg into `out`; return their paths."""
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for theme in THEMES:
        path = out / f"pipeline-{theme}.svg"
        path.write_text(svg(theme), encoding="utf-8")
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "figures", help="output folder")
    args = parser.parse_args(argv)
    for path in write_figures(args.out):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
