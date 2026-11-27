"""Draw the README title image: a folder of papers, the molecules their tables define, molecules.csv.

    uv run --with playwright python scripts/hero.py
    uv run --with playwright python scripts/hero.py --out docs/figures

Writes hero-light.png (white background) and hero-dark.png (GitHub dark, #0d1117) at twice
WIDTH x HEIGHT. RDKit draws the molecules; Playwright's Chromium renders the page (install it once
with `uv run --with playwright playwright install chromium-headless-shell`). Text, time and EXIF
chunks are stripped from the PNGs.

The examples are generic public scaffolds, built row by row with RDKit's molzip as MolScout builds
R-group tables: paper_a is a 4-anilinoquinazoline with R = Cl, Br and ethynyl at the aniline meta
position; paper_b is a 2-aryl-imidazo[1,2-a]pyridine with R = F and OMe at the aryl para position.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point3D

REPO = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 1600, 480
MAX_BYTES = 400_000
MOLECULE_BOX, MOLECULE_BOND = (196, 112), 17
DROPPED_CHUNKS = {b"tEXt", b"iTXt", b"zTXt", b"eXIf", b"tIME"}


@dataclass(frozen=True)
class Series:
    """One paper's R-group table: a core with an attachment point [*:1] and the rows that fill it."""

    paper: str
    core: str
    scaffold: str
    rows: tuple[tuple[str, str, str], ...]  # label, R as printed in the table, R as a fragment
    tilt: float  # degrees; paper_a's -10 lets all three molecules fit one row height at one bond length


SERIES = (
    Series("paper_a", "[*:1]c1cccc(Nc2ncnc3ccccc23)c1", "c1cccc(Nc2ncnc3ccccc23)c1",
           (("1a", "Cl", "Cl[*:1]"), ("1b", "Br", "Br[*:1]"), ("1c", "C≡CH", "C#C[*:1]")), -10),
    Series("paper_b", "[*:1]c1ccc(-c2cn3ccccc3n2)cc1", "c1ccc(-c2cn3ccccc3n2)cc1",
           (("7a", "F", "F[*:1]"), ("7b", "OMe", "CO[*:1]")), 0),
)


@dataclass(frozen=True)
class Theme:
    background: str
    panel: str
    border: str
    ink: str
    muted: str
    bar: str
    bar_strong: str
    accent: str
    halo: tuple[float, float, float, float]


THEMES = {
    "light": Theme("#ffffff", "#ffffff", "#d0d7de", "#1f2328", "#59636e", "#e4e7eb", "#c9ced4", "#2370cd",
                   (0.137, 0.439, 0.804, 0.26)),
    "dark": Theme("#0d1117", "#151b23", "#30363d", "#e6edf3", "#9198a1", "#262c36", "#3a424d", "#58a6ff",
                  (0.345, 0.651, 1.0, 0.36)),
}


@dataclass(frozen=True)
class Compound:
    series: Series
    label: str
    r_text: str
    fragment: str
    smiles: str
    mol: Chem.Mol

    @property
    def paper(self) -> str:
        return self.series.paper


def compounds() -> list[Compound]:
    """Every table row of every paper, joined to its core and written as RDKit canonical SMILES."""
    out = []
    for series in SERIES:
        core = Chem.MolFromSmiles(series.core)
        for label, r_text, fragment in series.rows:
            smiles = Chem.MolToSmiles(Chem.molzip(core, Chem.MolFromSmiles(fragment)))
            out.append(Compound(series, label, r_text, fragment, smiles, Chem.MolFromSmiles(smiles)))
    return out


def r_atoms(compound: Compound) -> list[int]:
    """Atoms outside the scaffold: the substituent that the table row filled in."""
    match = set(compound.mol.GetSubstructMatch(Chem.MolFromSmiles(compound.series.scaffold)))
    return [atom.GetIdx() for atom in compound.mol.GetAtoms() if atom.GetIdx() not in match]


def smiles_parts(smiles: str, fragment: str) -> tuple[str, str]:
    """(R group, rest) when the SMILES starts with the R group; otherwise ("", smiles)."""
    r_group = Chem.MolToSmiles(Chem.DeleteSubstructs(Chem.MolFromSmiles(fragment), Chem.MolFromSmarts("[#0]")))
    return (r_group, smiles[len(r_group):]) if smiles.startswith(r_group) else ("", smiles)


def _rgb(hex_colour: str) -> tuple[float, float, float]:
    return tuple(int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5))  # type: ignore[return-value]


def _template(series: Series) -> Chem.Mol:
    """The core with its attachment point: long axis horizontal, then tilted by the series' angle."""
    core = Chem.MolFromSmiles(series.core)
    rdDepictor.Compute2DCoords(core, canonOrient=True)
    conformer, angle = core.GetConformer(), math.radians(series.tilt)
    for i in range(core.GetNumAtoms()):
        p = conformer.GetAtomPosition(i)
        x, y = p.x * math.cos(angle) - p.y * math.sin(angle), p.x * math.sin(angle) + p.y * math.cos(angle)
        conformer.SetAtomPosition(i, Point3D(x, y, 0.0))
    return core


def _svg(mol: Chem.Mol, theme: Theme, size: tuple[int, int], highlight: list[int], bond_length: float) -> str:
    drawer = rdMolDraw2D.MolDraw2DSVG(*size)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.setAtomPalette({-1: _rgb(theme.ink)})
    options.bondLineWidth = 2
    options.fixedBondLength = bond_length
    options.minFontSize = 16
    options.continuousHighlight = True
    options.highlightRadius = 0.5
    bonds = [b.GetIdx() for b in mol.GetBonds() if b.GetBeginAtomIdx() in highlight or b.GetEndAtomIdx() in highlight]
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight,
        highlightBonds=bonds,
        highlightAtomColors={i: theme.halo for i in highlight},
        highlightBondColors={i: theme.halo for i in bonds},
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText().split("?>", 1)[-1].strip()


def molecule_svg(compound: Compound, theme: Theme) -> str:
    """The molecule in its core's layout, so every row of a paper puts R in the same place."""
    mol = Chem.Mol(compound.mol)
    series = compound.series
    # the SMARTS form of the core lets its dummy atom match the substituent's first atom
    rdDepictor.GenerateDepictionMatching2DStructure(mol, _template(series), refPatt=Chem.MolFromSmarts(series.core))
    return _svg(mol, theme, MOLECULE_BOX, r_atoms(compound), bond_length=MOLECULE_BOND)


def core_svg(theme: Theme) -> str:
    mol = _template(SERIES[0])
    dummy = next(atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0)
    dummy.SetProp("atomLabel", "R")
    return _svg(mol, theme, (240, 84), [dummy.GetIdx()], bond_length=16)


def _arrow(theme: Theme) -> str:
    return (
        f'<svg class="arrow" width="52" height="20" viewBox="0 0 52 20">'
        f'<line x1="2" y1="10" x2="46" y2="10" stroke="{theme.muted}" stroke-width="2.5" stroke-linecap="round"/>'
        f'<path d="M38 3 L48 10 L38 17" fill="none" stroke="{theme.muted}" stroke-width="2.5" '
        f'stroke-linecap="round" stroke-linejoin="round"/></svg>'
    )


# Skeletons of the papers behind the front one, back to front. Only their top and right edges show.
BACK_PAGES = (
    '<span class="bar title w55"></span><span class="block fig right"></span>',
    '<span class="bar title w40"></span><div class="block table right">' + "<i></i>" * 6 + "</div>",
    '<span class="bar title w80"></span><span class="bar w90"></span><span class="block fig right low"></span>',
)


def _papers(theme: Theme, rows: list[Compound]) -> str:
    table = "".join(
        f'<tr><td>{c.label}</td><td class="r">{c.r_text}</td><td><span class="bar cell"></span></td></tr>'
        for c in rows
    )
    back = "".join(f'<div class="page back b{i}">{skeleton}</div>' for i, skeleton in enumerate(BACK_PAGES))
    return f"""
    <div class="stack">{back}
      <div class="page front">
        <span class="bar title w72"></span>
        <span class="bar w90"></span><span class="bar w75"></span>
        <div class="figure">{core_svg(theme)}</div>
        <span class="bar caption"></span>
        <table><tr><th>Cpd</th><th>R</th><th>IC50</th></tr>{table}</table>
        <span class="bar w60"></span>
      </div>
    </div>
    <code class="folder">papers/</code>"""


def _molecules(theme: Theme, rows: list[Compound]) -> str:
    groups = ""
    for series in SERIES:
        cells = "".join(
            f'<div class="mol"><span class="label">{c.label}</span>{molecule_svg(c, theme)}</div>'
            for c in rows
            if c.series is series
        )
        groups += f'<div class="group"><code class="paper">{series.paper}</code>{cells}</div>'
    return f'<div class="panel molecules">{groups}</div>'


def _csv(rows: list[Compound]) -> str:
    lines = '<span class="head">paper</span><span class="head">label</span><span class="head">SMILES</span>'
    previous = None
    for c in rows:
        r_group, rest = smiles_parts(c.smiles, c.fragment)
        cls = ' class="split"' if previous not in (None, c.paper) else ""
        lines += (
            f'<code{cls}><i>{c.paper}</i></code><code{cls}>{c.label}</code>'
            f"<code{cls}><b>{r_group}</b>{rest}</code>"
        )
        previous = c.paper
    return f'<div class="panel csv"><code class="file">out/molecules.csv</code><div class="grid">{lines}</div></div>'


def page_html(theme_name: str) -> str:
    theme = THEMES[theme_name]
    rows = compounds()
    front = [c for c in rows if c.series is SERIES[0]]
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{_css(theme)}</style></head><body>
<main>
  <section><h2>Papers (folder of PDFs)</h2>{_papers(theme, front)}</section>
  {_arrow(theme)}
  <section><h2>Molecules</h2>{_molecules(theme, rows)}</section>
  {_arrow(theme)}
  <section><h2>SMILES</h2>{_csv(rows)}</section>
</main></body></html>"""


PAGE_W, PAGE_H, STEP_X, STEP_Y = 280, 300, 16, 20
BODY_H = PAGE_H + 3 * STEP_Y + 30  # stack plus the folder name under it


def _css(t: Theme) -> str:
    places = "".join(
        f".b{i} {{ left: {(3 - i) * STEP_X}px; top: {i * STEP_Y}px; }}" for i in range(3)
    )
    return f"""
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ width: {WIDTH}px; height: {HEIGHT}px; background: {t.background}; overflow: hidden; }}
body {{ font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; color: {t.ink}; }}
main {{ display: flex; align-items: flex-start; justify-content: space-between; height: 100%; padding: 24px 36px; }}
section {{ display: flex; flex-direction: column; gap: 12px; }}
h2 {{ font-size: 22px; font-weight: 600; color: {t.muted}; letter-spacing: 0.01em; line-height: 27px; }}
code {{ font-family: "SF Mono", Menlo, "DejaVu Sans Mono", monospace; }}
.arrow {{ margin-top: {39 + BODY_H // 2 - 10}px; flex: none; }}
.stack {{ position: relative; width: {PAGE_W + 3 * STEP_X}px; height: {PAGE_H + 3 * STEP_Y}px; }}
.page {{ position: absolute; width: {PAGE_W}px; height: {PAGE_H}px; background: {t.panel};
        border: 1px solid {t.border}; border-radius: 6px; padding: 8px 16px; display: flex;
        flex-direction: column; gap: 6px; overflow: hidden; }}
{places}
.front {{ left: 0; top: {3 * STEP_Y}px; padding: 16px 18px; }}
.folder {{ display: block; margin-top: 8px; font-size: 18px; color: {t.muted}; text-align: center;
          width: {PAGE_W + 3 * STEP_X}px; }}
.bar {{ display: block; height: 7px; border-radius: 4px; background: {t.bar}; flex: none; }}
.bar.title {{ height: 10px; background: {t.bar_strong}; margin-bottom: 4px; }}
.w40 {{ width: 40%; }} .w55 {{ width: 55%; }} .w60 {{ width: 58%; }} .w72 {{ width: 72%; }}
.w75 {{ width: 76%; }} .w80 {{ width: 80%; }} .w90 {{ width: 92%; }}
.block {{ display: block; border-radius: 4px; flex: none; }}
.block.fig {{ width: 46%; height: 120px; background: {t.bar}; }}
.block.table {{ width: 70%; border: 1px solid {t.bar_strong}; }}
.block.table i {{ display: block; height: 23px; border-bottom: 1px solid {t.bar_strong}; }}
.block.right {{ align-self: flex-end; margin-right: -4px; }}
.block.low {{ margin-top: 24px; }}
.bar.caption {{ width: 64%; margin: 0 auto 4px; }}
.figure {{ display: flex; justify-content: center; margin-top: 2px; }}
table {{ border-collapse: collapse; font-size: 15px; }}
th, td {{ text-align: left; padding: 2px 14px 2px 0; height: 23px; }}
th {{ font-weight: 600; color: {t.muted}; border-bottom: 1px solid {t.border}; }}
td.r {{ color: {t.accent}; font-weight: 600; }}
.bar.cell {{ width: 40px; }}
.panel {{ background: {t.panel}; border: 1px solid {t.border}; border-radius: 8px; height: {BODY_H}px; }}
.molecules {{ display: flex; width: 512px; padding: 10px 0; }}
.group {{ flex: 1; display: flex; flex-direction: column; padding: 0 14px; }}
.group + .group {{ border-left: 1px solid {t.border}; }}
.paper {{ font-size: 15px; color: {t.muted}; height: 22px; }}
.mol {{ display: flex; align-items: center; gap: 6px; height: 116px; }}
.label {{ font-size: 19px; font-weight: 600; width: 28px; flex: none; }}
.csv {{ width: 476px; padding: 14px 20px; }}
.file {{ display: block; font-size: 15px; color: {t.muted}; margin-bottom: 14px; }}
.grid {{ display: grid; grid-template-columns: auto auto 1fr; align-items: center; }}
.grid > :not(:nth-child(3n)) {{ padding-right: 18px; }}
.head {{ font-size: 15px; font-weight: 600; color: {t.muted}; padding-bottom: 8px;
         border-bottom: 1px solid {t.border}; }}
.grid code {{ font-size: 17px; padding-top: 18px; padding-bottom: 18px; }}
.grid code i {{ font-style: normal; color: {t.muted}; }}
.grid code b {{ color: {t.accent}; font-weight: 600; }}
.grid code.split {{ border-top: 1px solid {t.border}; }}
"""


def strip_metadata(path: Path) -> None:
    """Drop text, time and EXIF chunks; the image data is untouched."""
    data = path.read_bytes()
    out, i = [data[:8]], 8
    while i < len(data):
        length, name = struct.unpack(">I4s", data[i : i + 8])
        if name not in DROPPED_CHUNKS:
            out.append(data[i : i + 12 + length])
        i += 12 + length
    path.write_bytes(b"".join(out))


def _shrink(path: Path) -> None:
    """Re-save with a 256-colour palette when the PNG is over MAX_BYTES."""
    if path.stat().st_size <= MAX_BYTES:
        return
    from PIL import Image

    with Image.open(path) as image:
        image.convert("RGB").quantize(colors=256, dither=Image.Dither.NONE).save(path, optimize=True)


def render(html: str, path: Path) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=2)
        page.set_content(html, wait_until="load")
        page.screenshot(path=str(path), clip={"x": 0, "y": 0, "width": WIDTH, "height": HEIGHT})
        browser.close()
    _shrink(path)
    strip_metadata(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw the README title image, light and dark.")
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "figures")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        path = args.out / f"hero-{theme}.png"
        render(page_html(theme), path)
        print(f"wrote {path} ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
