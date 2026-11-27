"""Parse stage: MinerU 4.2.3 reads each PDF and the result becomes <paper_dir>/layout.jsonl.

Runs in the mineru environment (envs/mineru/setup.sh) as a script; it imports the standard library
and MinerU only, never molscout. MinerU runs locally on every page with the models already in the
store (MINERU_MODEL_SOURCE=local, telemetry off); its remote parsing service is never used.

    python mineru_parse.py --manifest work.jsonl --log stage.log [--tier standard|basic|advanced]
                           [--backend auto|vllm|llama-cpp|lmdeploy|mlx] [--small-backend auto|onnx|torch]

Each manifest line is {"paper_id", "pdf", "paper_dir"}. Per paper the worker writes
  mineru/             MinerU's own output (middle_json.json, images/, markdown.md)
  layout.jsonl        one block per line: page (1-based), index (reading order on the page), type
                      (text, title, table, image, caption, footnote, equation), bbox (PDF points,
                      origin top-left), text, html (tables), parent (index of a caption's figure or
                      table), image (crop path relative to paper_dir, or null)
  layout_images/      the crops layout.jsonl and the table HTML refer to
and records per-paper failures in errors.jsonl, after clearing the paper's old parse errors. Every
block has a box: one without is a conversion error. MinerU's page sizes are checked against the PDF's
(pdfium's, as render.py records them); pages that differ are an error, as their boxes would be
misplaced, but the layout is kept. A paper that already has layout.jsonl is skipped unless --force.
One paper's failure, expected or not, never stops the others. Exit status 0; 4 when some papers
failed; 1 when MinerU cannot start or files cannot be locked; 2 on a bad manifest.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _readers_common as common  # noqa: E402

STAGE = "parse"
SIZE_TOLERANCE = (1.0, 0.005)  # points, fraction: MinerU rounds page sizes to whole points
MINERU_VERSION = "4.2.3"
LAYOUT_FILE = "layout.jsonl"
RAW_DIR = "mineru"
IMAGES_DIR = "layout_images"

# MinerU block types -> layout.jsonl types. Page furniture is dropped on purpose.
TEXT_TYPES = {"text": "text", "ref_text": "text", "doc_title": "title", "paragraph_title": "title",
              "page_footnote": "footnote"}
DROPPED_TYPES = frozenset({"header", "footer", "page_number", "aside_text"})
VISUAL_TYPES = {"image": "image", "chart": "image", "table": "table", "code": "text"}
BODY_TYPES = frozenset({"image_body", "chart_body", "table_body", "code_body", "algorithm_body"})
CHILD_TYPES = {f"{kind}_{role}": role for kind in ("image", "chart", "table", "code") for role in ("caption", "footnote")}
NESTED_TYPES = frozenset({"list", "index"})

IMG_SRC_RE = re.compile(r'<img\b[^>]*?\bsrc="([^"]*)"')
EQ_RE = re.compile(r"<eq>(.*?)</eq>", re.S)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
CAPTION_RE = re.compile(r"^\s*(table|figure|fig\.|scheme|chart)\s*S?\d+[A-Za-z]?(?:[.:|]|\s)", re.I)

# --- LaTeX (MinerU writes inline formulas and table <eq> cells as space-separated LaTeX) ---------

SYMBOLS = {
    "pm": "±", "mp": "∓", "times": "×", "cdot": "·", "mu": "μ", "alpha": "α", "beta": "β",
    "gamma": "γ", "delta": "δ", "Delta": "Δ", "epsilon": "ε", "kappa": "κ", "lambda": "λ",
    "pi": "π", "sigma": "σ", "tau": "τ", "omega": "ω", "Omega": "Ω", "circ": "°", "degree": "°",
    "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "approx": "≈", "sim": "~", "neq": "≠",
    "infty": "∞", "rightarrow": "→", "to": "→", "leftarrow": "←", "prime": "′", "AA": "Å",
    "%": "%", "lt": "<", "gt": ">", "textmu": "μ", "textdegree": "°",
}
STYLE_GROUP_RE = re.compile(
    r"\\(?:mathsf|mathrm|mathbf|mathit|mathbb|mathcal|boldsymbol|operatorname|textbf|textit|textrm|text|mbox)"
    r"\s*\{([^{}]*)\}")
SCRIPT_GROUP_RE = re.compile(r"\s*[_^]\s*\{([^{}]*)\}")
SCRIPT_CHAR_RE = re.compile(r"\s*[_^]\s*([^\s{}\\])")
COMMAND_RE = re.compile(r"\\([A-Za-z]+|%)")
SPACING_RE = re.compile(r"\\[,;:! ]|\\q?quad\b")
GLUE_RE = re.compile(r"[A-Za-z0-9.]")


def _join_spaced_chars(latex: str) -> str:
    """MinerU writes one token per character ('0 . 3 9 6', 'I C'): join runs of single letters, digits
    and points. Two or more spaces stay a word break."""
    out: list[str] = []
    open_run = False
    for token in latex.split(" "):
        single = len(token) == 1 and GLUE_RE.fullmatch(token) is not None
        if single and open_run:
            out[-1] += token
        else:
            out.append(token)
            open_run = single
    return " ".join(out)


def _join_group(inner: str) -> str:
    """The text of one LaTeX group: single spaces are tokenisation, wider gaps are word breaks."""
    return " ".join(part.replace(" ", "") for part in re.split(r"\s{2,}", inner.strip()))


def latex_to_text(latex: str) -> str:
    """Plain, searchable text for MinerU's inline LaTeX: IC50 μM, 0.396 ± 0.180, pKi = 7.2."""
    text = _join_spaced_chars(SPACING_RE.sub("  ", latex))
    previous = None
    while previous != text:  # innermost groups first, so nested styles and scripts unwind
        previous = text
        text = STYLE_GROUP_RE.sub(lambda m: _join_group(m.group(1)), text)
        text = SCRIPT_GROUP_RE.sub(lambda m: _join_group(m.group(1)), text)
        text = SCRIPT_CHAR_RE.sub(lambda m: m.group(1), text)
    text = COMMAND_RE.sub(lambda m: f" {SYMBOLS[m.group(1)]} " if m.group(1) in SYMBOLS else " ", text)
    text = re.sub(r"[{}$\\]", " ", text)
    text = re.sub(r"μ\s+(?=[A-Za-z])", "μ", text)
    text = re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", text))
    return re.sub(r"\s+", " ", text).strip()


# --- conversion (pure) ----------------------------------------------------------------------------


def clean_text(text: str) -> str:
    return CONTROL_RE.sub("", text).strip()


def spans_to_text(spans: list[dict[str, Any]]) -> str:
    """Concatenate MinerU inline spans (text, equation_inline, code_inline, hyperlink)."""
    parts = []
    for span in spans:
        kind, content = span.get("type"), span.get("content")
        if kind == "hyperlink":
            parts.append(spans_to_text(content or []))
        elif kind == "equation_inline":
            parts.append(latex_to_text(content or ""))
        elif isinstance(content, str):
            parts.append(content)
    return "".join(parts)


def _content_text(block: dict[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return clean_text(content)
    return clean_text(spans_to_text(content or []))


def to_points(bbox: list[float] | None, width_pt: float, height_pt: float) -> list[float]:
    """[x0, y0, x1, y1] normalised to 0-1 -> PDF points (origin top-left), 2 decimals; ValueError without one."""
    if bbox is None:
        raise ValueError("a MinerU block has no bbox")
    x0, y0, x1, y1 = bbox
    return [round(x0 * width_pt, 2), round(y0 * height_pt, 2), round(x1 * width_pt, 2), round(y1 * height_pt, 2)]


def table_html(raw_html: str, image_names: dict[str, str]) -> str:
    """Table HTML with <eq> cells as plain text and cell images pointing at layout_images/."""
    def rename(match: re.Match[str]) -> str:
        tag, start, end = match.group(0), match.start(1) - match.start(0), match.end(1) - match.start(0)
        return tag[:start] + image_names.get(match.group(1), match.group(1)) + tag[end:]

    out = EQ_RE.sub(lambda m: html.escape(latex_to_text(html.unescape(m.group(1))), quote=False), raw_html)
    return IMG_SRC_RE.sub(rename, out)


@dataclass
class _Item:
    """One future layout.jsonl record, before reading-order renumbering."""

    key: int | None
    order: int
    type: str
    bbox: list[float] | None
    text: str | None = None
    html: str | None = None
    parent: int | None = None  # the parent item's order
    image_src: str | None = None
    cell_images: list[str] = field(default_factory=list)


def _page_items(blocks: list[dict[str, Any]]) -> list[_Item]:
    items: list[_Item] = []

    def add(item_type: str, block: dict[str, Any], **kwargs: Any) -> int:
        items.append(_Item(block.get("index"), len(items), item_type, block.get("bbox"), **kwargs))
        return len(items) - 1

    def walk(block: dict[str, Any]) -> None:
        kind = block.get("type")
        if kind in DROPPED_TYPES:
            return
        if kind in TEXT_TYPES:
            add(TEXT_TYPES[kind], block, text=_content_text(block))
        elif kind == "equation":
            add("equation", block, text=clean_text(block.get("content") or ""))
        elif kind in NESTED_TYPES:
            for child in block.get("content") or []:
                walk(child)
        elif kind in VISUAL_TYPES:
            children = block.get("content") or []
            body = next((c for c in children if c.get("type") in BODY_TYPES), None)
            if body is None:
                raise ValueError(f"MinerU {kind} block {block.get('index')} has no body")
            record_type = VISUAL_TYPES[kind]
            if record_type == "text":  # code and algorithm listings
                position = add("text", block, text=_content_text(body))
            elif record_type == "table":
                raw = body.get("content") or ""
                position = add("table", block, html=raw, image_src=body.get("image_path"),
                               cell_images=list(dict.fromkeys(IMG_SRC_RE.findall(raw))))
            else:
                position = add("image", block, image_src=body.get("image_path"))
            for child in children:
                if child is body:
                    continue
                role = CHILD_TYPES.get(child.get("type"))
                if role is None:
                    raise ValueError(f"unknown MinerU block type {child.get('type')!r} inside {kind}")
                add(role, child, text=_content_text(child), parent=position)
        else:
            raise ValueError(f"unknown MinerU block type {kind!r}")

    for block in blocks:
        walk(block)
    if all(item.key is not None for item in items):
        items.sort(key=lambda item: (item.key, item.order))
    return items


def _link_text_captions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A 'Table 2.' / 'Figure 3.' text or title block next to an uncaptioned table or figure is its caption."""
    out = list(records)
    captioned = {r["parent"] for r in out if r["type"] == "caption"}
    for position, record in enumerate(out):
        match = CAPTION_RE.match(record["text"] or "") if record["type"] in ("text", "title") else None
        if not match:
            continue
        wanted = "table" if match.group(1).lower() == "table" else "image"
        # tables are captioned above, figures and schemes below
        order = (position + 1, position - 1) if wanted == "table" else (position - 1, position + 1)
        for neighbour in order:
            if 0 <= neighbour < len(out) and out[neighbour]["type"] == wanted \
                    and out[neighbour]["index"] not in captioned:
                out[position] = {**record, "type": "caption", "parent": out[neighbour]["index"]}
                captioned.add(out[neighbour]["index"])
                break
    return out


def convert_middle(middle: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """MinerU middle_json -> (layout.jsonl records, {crop path under layout_images/: MinerU image path}).

    Pages are 1-based; index is the reading order on the page, renumbered from 0 after page furniture
    (headers, footers, page numbers, margin notes) is dropped; boxes are PDF points from the page
    sizes MinerU records under extensions.docvortex_layout.
    """
    layout = (middle.get("extensions") or {}).get("docvortex_layout") or {}
    sizes = {p["page_idx"]: (p["width_pt"], p["height_pt"]) for p in layout.get("pages") or []}
    records: list[dict[str, Any]] = []
    images: dict[str, str] = {}
    for page in sorted(middle.get("pages") or [], key=lambda p: p["page_idx"]):
        page_no = page["page_idx"] + 1
        items = _page_items(page.get("blocks") or [])
        if items and page["page_idx"] not in sizes:
            raise ValueError(f"no page size for page {page_no} in extensions.docvortex_layout")
        width, height = sizes.get(page["page_idx"], (0, 0))
        new_index = {item.order: n for n, item in enumerate(items)}
        page_records = []
        for n, item in enumerate(items):
            crop = None
            if item.image_src:
                crop = f"{IMAGES_DIR}/p{page_no}-{n}{Path(item.image_src).suffix or '.jpg'}"
                images[crop] = item.image_src
            names = {}
            for k, src in enumerate(item.cell_images, start=1):
                names[src] = f"{IMAGES_DIR}/p{page_no}-{n}-c{k}{Path(src).suffix or '.jpg'}"
                images[names[src]] = src
            page_records.append({
                "page": page_no,
                "index": n,
                "type": item.type,
                "bbox": to_points(item.bbox, width, height),
                "text": item.text,
                "html": table_html(item.html, names) if item.type == "table" else None,
                "parent": new_index[item.parent] if item.parent is not None else None,
                "image": crop,
            })
        records.extend(_link_text_captions(page_records))
    return records, images


def parse_problems(middle: dict[str, Any], records: list[dict[str, Any]],
                   pdf_sizes: list[tuple[float, float]]) -> list[tuple[str, str]]:
    """Failures to record loudly: an empty parse of a non-empty PDF, pages MinerU did not return, and pages
    whose size MinerU records differently from the PDF (pdf_sizes: points per page, as shown)."""
    pdf_pages = len(pdf_sizes)
    if pdf_pages > 0 and not records:
        return [("empty_parse", f"MinerU returned no blocks for a {pdf_pages}-page PDF")]
    problems = []
    returned = {page["page_idx"] for page in middle.get("pages") or []}
    missing = [str(i + 1) for i in range(pdf_pages) if i not in returned]
    if missing:
        problems.append(("pages_missing",
                         f"MinerU returned {pdf_pages - len(missing)} of {pdf_pages} pages; missing: {', '.join(missing)}"))
    differ = size_mismatches(middle, pdf_sizes)
    if differ:
        problems.append(("page_size_mismatch",
                         f"MinerU's page sizes differ from the PDF's, so its boxes would be misplaced: {', '.join(differ)}"))
    return problems


def size_mismatches(middle: dict[str, Any], pdf_sizes: list[tuple[float, float]]) -> list[str]:
    """'page N W x H pt (PDF W x H)' for each page whose MinerU size is off by more than SIZE_TOLERANCE."""
    points, fraction = SIZE_TOLERANCE
    out = []
    for page in ((middle.get("extensions") or {}).get("docvortex_layout") or {}).get("pages") or []:
        index, size = page["page_idx"], (page["width_pt"], page["height_pt"])
        if index >= len(pdf_sizes):
            continue
        pdf = pdf_sizes[index]
        if any(abs(a - b) > max(points, fraction * b) for a, b in zip(size, pdf)):
            out.append(f"page {index + 1} {size[0]:.0f} x {size[1]:.0f} pt (PDF {pdf[0]:.0f} x {pdf[1]:.0f})")
    return out


# --- per-paper IO -----------------------------------------------------------------------------------


class Engine(Protocol):
    def page_sizes(self, pdf: Path) -> list[tuple[float, float]]: ...

    def parse(self, pdf: Path, out_dir: Path) -> None: ...


class SetupError(RuntimeError):
    """MinerU cannot run at all (not installed, wrong version, models missing): exit non-zero."""


def engine_environment(backend: str | None, small_backend: str | None) -> dict[str, str]:
    """Environment for MinerU: local models only, no telemetry, optional engine choices."""
    env = {"MINERU_TELEMETRY_DISABLED": "1", "MINERU_MODEL_SOURCE": "local"}
    if backend:
        env["MINERU_MODEL_VLM_ENGINE"] = backend
    if small_backend:
        env["MINERU_MODEL_SMALL_BACKEND"] = small_backend
    return env


class MineruEngine:
    """MinerU's local parser, loaded once for all papers of a run."""

    def __init__(self, tier: str = "standard", backend: str | None = None, small_backend: str | None = None):
        os.environ.update(engine_environment(backend, small_backend))
        os.environ.pop("MINERU_API_KEY", None)
        try:
            from mineru.model.download import verify_model_repo
            from mineru.model.registry import model_repos_for_tier
            from mineru.parser import parse
            from mineru.parser.writer import FileBasedDataWriter
            from mineru.version import __version__
        except ImportError as exc:
            raise SetupError(f"cannot import MinerU: {exc}") from exc
        if __version__ != MINERU_VERSION:
            raise SetupError(f"MinerU {__version__} installed, {MINERU_VERSION} expected")
        missing = []
        for repo in model_repos_for_tier("basic" if tier == "basic" else "standard"):
            result = verify_model_repo(repo)
            if not result.ready:
                missing.append(f"{repo.name} ({', '.join(result.missing_paths)})")
        if missing:
            raise SetupError(f"models missing: {'; '.join(missing)}; run envs/mineru/setup.sh")
        self.tier = tier
        self._parse = parse
        self._writer = FileBasedDataWriter

    def page_sizes(self, pdf: Path) -> list[tuple[float, float]]:
        """Each page's size in points as shown (after /Rotate), as render.py records it."""
        import pypdfium2

        document = pypdfium2.PdfDocument(str(pdf))
        try:
            sizes = []
            for index in range(len(document)):
                page = document[index]
                try:
                    sizes.append(tuple(page.get_size()))
                finally:
                    page.close()
            return sizes
        finally:
            document.close()

    def parse(self, pdf: Path, out_dir: Path) -> None:
        result = self._parse(pdf, tier=self.tier, ocr_mode="auto", page_range="")  # "" = every page
        result.save(self._writer(str(out_dir)))


@dataclass
class PaperResult:
    paper_id: str
    status: str  # ok, error, skipped
    pages: int = 0
    blocks: int = 0
    tables: int = 0
    seconds: float = 0.0
    errors: list[tuple[str, str]] = field(default_factory=list)


def _write_errors(paper_dir: Path, paper_id: str, problems: list[tuple[str, str]]) -> None:
    for code, message in problems:
        common.append_error(paper_dir, STAGE, paper_id, code, message=message)


def _copy_images(images: dict[str, str], raw_dir: Path, paper_dir: Path) -> list[str]:
    """Copy MinerU's crops to layout_images/; returns the crop paths MinerU referenced but did not write."""
    target = paper_dir / IMAGES_DIR
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    missing = []
    for dest, src in images.items():
        source = raw_dir / src
        if source.is_file():
            shutil.copyfile(source, paper_dir / dest)
        else:
            missing.append(dest)
    return missing


def _write_outputs(records: list[dict[str, Any]], images: dict[str, str], paper_dir: Path) -> list[tuple[str, str]]:
    """Copy the crops and write layout.jsonl; the crops MinerU referenced but did not write, as a problem."""
    missing = _copy_images(images, paper_dir / RAW_DIR, paper_dir)
    if missing:
        records = [{**r, "image": None} if r["image"] in missing else r for r in records]
    common.write_jsonl(paper_dir / LAYOUT_FILE, records)
    if not missing:
        return []
    return [("image_missing", "MinerU referenced crops it did not write: " + ", ".join(images[dest] for dest in missing))]


def _run_mineru(engine: Engine, pdf: Path, paper_dir: Path) -> dict[str, Any]:
    """MinerU into a temporary folder, renamed to mineru/ only when complete."""
    tmp = Path(tempfile.mkdtemp(prefix=".mineru-", dir=paper_dir))
    try:
        engine.parse(pdf, tmp)
        middle = json.loads((tmp / "middle_json.json").read_text(encoding="utf-8"))
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    raw = paper_dir / RAW_DIR
    if raw.exists():
        shutil.rmtree(raw)
    tmp.rename(raw)
    return middle


def process_paper(paper: dict[str, str], engine: Engine, *, force: bool = False) -> PaperResult:
    paper_id, pdf, paper_dir = paper["paper_id"], Path(paper["pdf"]), Path(paper["paper_dir"])
    layout_path = paper_dir / LAYOUT_FILE
    if layout_path.exists() and not force:
        return PaperResult(paper_id, "skipped")
    start = time.monotonic()
    paper_dir.mkdir(parents=True, exist_ok=True)
    common.clear_stage_errors(paper_dir, STAGE)
    layout_path.unlink(missing_ok=True)

    def fail(code: str, message: str) -> PaperResult:
        _write_errors(paper_dir, paper_id, [(code, message)])
        return PaperResult(paper_id, "error", seconds=time.monotonic() - start, errors=[(code, message)])

    if not pdf.is_file():
        return fail("pdf_missing", f"no PDF at {pdf}")
    try:
        pdf_sizes = engine.page_sizes(pdf)
    except Exception as exc:  # noqa: BLE001 - any reader failure is this paper's error
        return fail("pdf_unreadable", f"{type(exc).__name__}: {exc}")
    try:
        middle = _run_mineru(engine, pdf, paper_dir)
    except Exception as exc:  # noqa: BLE001
        return fail("mineru_failed", f"{type(exc).__name__}: {exc}")
    try:
        records, images = convert_middle(middle)
    except (ValueError, KeyError, TypeError) as exc:
        return fail("convert_failed", f"{type(exc).__name__}: {exc}")
    problems = parse_problems(middle, records, pdf_sizes)
    try:
        problems += _write_outputs(records, images, paper_dir) if records else []
    except OSError as exc:
        layout_path.unlink(missing_ok=True)
        return fail("write_failed", f"{type(exc).__name__}: {exc}")
    _write_errors(paper_dir, paper_id, problems)
    return PaperResult(
        paper_id, "error" if problems else "ok", pages=len(middle.get("pages") or []), blocks=len(records),
        tables=sum(r["type"] == "table" for r in records), seconds=time.monotonic() - start, errors=problems)


def guarded(paper: dict[str, str], engine: Engine, *, force: bool = False) -> PaperResult:
    """process_paper; an unexpected failure is the paper's "internal" error and the other papers go on."""
    try:
        return process_paper(paper, engine, force=force)
    except common.LockError:
        raise
    except Exception as exc:  # noqa: BLE001 - recorded in the paper's errors.jsonl and the log
        message = f"{type(exc).__name__}: {exc}"
        _write_errors(Path(paper["paper_dir"]), paper["paper_id"], [("internal", message)])
        return PaperResult(paper["paper_id"], "error", errors=[("internal", message)])


# --- CLI ----------------------------------------------------------------------------------------------


class ManifestError(ValueError):
    pass


def read_manifest(path: Path) -> list[dict[str, str]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    papers = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            paper = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{path}:{number}: not JSON: {exc}") from exc
        if not isinstance(paper, dict) or not all(isinstance(paper.get(k), str) and paper[k]
                                                  for k in ("paper_id", "pdf", "paper_dir")):
            raise ManifestError(f"{path}:{number}: needs string paper_id, pdf and paper_dir")
        papers.append(paper)
    return papers


class Log:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write(self, message: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp} {STAGE} {message}\n")
        print(f"{STAGE}: {message}", file=sys.stderr, flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--manifest", required=True, help="work.jsonl: paper_id, pdf, paper_dir per line")
    parser.add_argument("--log", required=True, help="stage log, appended to")
    parser.add_argument("--tier", default="standard", choices=("standard", "basic", "advanced"))
    parser.add_argument("--backend", choices=("auto", "vllm", "llama-cpp", "lmdeploy", "mlx"),
                        help="VLM engine for standard and advanced (MinerU picks when omitted)")
    parser.add_argument("--small-backend", choices=("auto", "onnx", "torch"),
                        help="layout, OCR, formula and table models (MinerU picks when omitted)")
    parser.add_argument("--force", action="store_true", help="re-parse papers that already have layout.jsonl")
    return parser


def main(argv: list[str] | None = None, engine: Engine | None = None) -> int:
    args = _parser().parse_args(argv)
    log = Log(Path(args.log))
    try:
        papers = read_manifest(Path(args.manifest))
    except ManifestError as exc:
        log.write(f"fatal: {exc}")
        return 2
    if engine is None:
        try:
            engine = MineruEngine(args.tier, args.backend, args.small_backend)
        except SetupError as exc:
            log.write(f"fatal: {exc}")
            return 1
    log.write(f"start: {len(papers)} papers, MinerU {MINERU_VERSION}, tier {args.tier}")
    counts = {"ok": 0, "skipped": 0, "error": 0}
    for paper in papers:
        try:
            result = guarded(paper, engine, force=args.force)
        except common.LockError as exc:
            log.write(f"fatal: {exc}")
            return 1
        counts[result.status] += 1
        detail = (f"pages={result.pages} blocks={result.blocks} tables={result.tables} "
                  f"seconds={result.seconds:.1f}")
        for code, message in result.errors:
            log.write(f"{result.paper_id} error {code}: {message}")
        log.write(f"{result.paper_id} {result.status} {detail}")
    log.write(f"done: {counts['ok']} ok, {counts['skipped']} skipped, {counts['error']} error(s)")
    return common.PARTIAL if counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
