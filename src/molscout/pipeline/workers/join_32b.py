"""The 32B joiner: BioMiner-Instruct links drawn structures to compound labels and reads R-group tables.

Runs in the biominer environment (Python 3.10, Pillow 10.4, RDKit, py2opsin) against vLLM serving
BioMiner-Instruct (serve_32b.sh); never imports molscout. Per paper it rebuilds BioMiner's structure
requests from pages, layout, regions and voted.jsonl and writes molecules_32b.jsonl (source 32b).
Requests, prompts, parsing and R-group joining follow BioMiner (jiaxianyan/BioMiner at 17c6161:
runner/extractor.py, runner/mllm.py, commons/ocsr.py, commons/mineru_pdf.py; MIT licence, Copyright (c)
2025 jiaxianyan): boxes with a voted SMILES get global indices by page and top edge; a SMILES with "*"
is "part" (scaffold or drawn R-group), else "full". Per page and kind, red numbered boxes are drawn on
the 200 dpi page; when every box lies in a figure or table area (block and caption, enlarged by 1/32)
each area is one image. A "full" image shows at most 4 boxes, cropped without areas to their extent
enlarged by 1/8; a "part" image shows all. BioMiner's own prompts (gemini-series, updated_v3_with_index)
are read from its clone and checked by sha256. Full answers [{index, identifier}] label drawings; part
answers [{scaffold, R-group: {name: index or text}, identifier}] are joined with RDKit's molzip, text
groups through BioMiner's formula table, its OPSIN cache, then OPSIN.

Departures from BioMiner, none affecting a well-formed answer: requests carry seed 0; failed, refused,
empty or unparsable answers are retried and then recorded in errors.jsonl (BioMiner uses []); a cut
answer is kept and recorded as truncated; entries that are malformed or name indices the image did not
show are not used and their counts go to errors.jsonl; R-group names map to slots through the voted
read's slot labels (voted.jsonl "slots") when they match, else in BioMiner's sorted order; unbuilt rows
are written as abstained (smiles null) with the reason. A table row is placed in its layout table when
its identifier labels one of the table's rows (_join_layout.py): table_id p{page}-t{index}, the table's
page and bbox; otherwise table_id 32b-<request>, the scaffold's page and bbox. Rows count from 0 per
table, in answer order. A molecule is accepted when 32b gave an identifier, else review; relevance is
finalize's. Also written: join_32b_requests.jsonl (per request: attempts, error, tokens, raw answer).
Logs hold counts and the server's vLLM version and model, never SMILES. Exit 0; 4 when some papers
failed (their errors are in errors.jsonl); 1 on a setup error (BioMiner's files, no server, no file
locks); 2 on a bad manifest; 3 when the server stopped answering mid-run (those papers get no
molecules_32b.jsonl).
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import http.client
import io
import json
import logging
import os
import re
import sqlite3
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _join_layout as layout  # noqa: E402
import _readers_common as common  # noqa: E402
from _join_layout import page_areas, plan_requests  # noqa: E402

RDLogger.DisableLog("rdApp.*")  # RDKit's messages quote SMILES; nothing here may print one

STAGE = "join_32b"
SERVED_MODEL = "local-biominer-instruct"
DEFAULT_URL = "http://127.0.0.1:8613/v1"
# BioMiner's system message for image requests (runner/mllm.py, call_api_iamge), verbatim
VISION_SYSTEM = ("You are a expert with certian knowledge of bioinfomratics and chemistry. "
                 "Please help finish these domain related paper-reading tasks.")
MAX_TOKENS = 4096  # BioMiner's max_tokens; temperature 0
BIOMINER_DPI = 200  # pdf2image's default, at which BioMiner renders pages
FONT_SIZE, LINE_WIDTH, RED = 25, 2, (255, 0, 0)  # draw_bbox_xywh
# files read from the BioMiner clone at 17c6161, with their sha256
ASSETS = {
    "full": ("BioMiner/commons/prompts/gemini-series-prompt/prompt_for_structure_full_updated_v3_with_index.txt",
             "f5ba3255dbe292d26017f6ab8657cef5316460e41a3e91f61d453e04ff4299e7"),
    "part": ("BioMiner/commons/prompts/gemini-series-prompt/prompt_for_structure_part_updated_v3_with_index.txt",
             "be069f8d32090e4186a66a58941c73ec760bbeb555efb4ae2085a719ca059a4f"),
    "formulas": ("BioMiner/commons/1d_r_groups/group_chem_formula_merge.log",
                 "81bbb01c1b8877f7d9c213b5e91d03147dd98d91af894b46574f800ed7184d3e"),
    "font": ("BioMiner/commons/arial.ttf", "82afb35eda3a52edb10106bcc04af93646384421ded538d38792c1444d816022"),
    "opsin_cache": ("BioMiner/commons/1d_r_groups/iupac_cache.db",
                    "5ba0c343e86cd957ed288eba0eb0233df5ec8b778bc2ad9d6ba3c0b0db044604"),
}
OPTIONAL_ASSETS = {"font", "opsin_cache"}  # may be absent only when the hashes are not checked (tests)
# names BioMiner writes out itself (convert_molminer_substitue_iupac_for_zip)
SPECIAL_GROUPS = {"TBDMS": "[Si](*)(C)(C)C(C)(C)C", "TBDPS": "[Si](*)(C1=CC=CC=C1)(C1=CC=CC=C1)C(C)(C)C",
                  "nitro": "[N+](*)(=O)[O-]", "hydroxyl": "O(*)[H]"}
TRANSPORT_ERRORS = {"http", "timeout", "connection", "bad_response"}
LOST = {"server_down", "aborted"}
NO_LABEL = {"", "none", "null", "n/a", "na", "-", "unknown"}
SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉", "01234567890123456789")

log = logging.getLogger(STAGE)


# ---------------------------------------------------------------- inputs

@dataclass(frozen=True)
class Assets:
    prompts: dict
    formulas: dict
    font: Path | None
    opsin_cache: dict


@dataclass(frozen=True)
class Box:
    index: int
    region_id: str
    page: int
    bbox: list  # PDF points, as regions.jsonl gives it
    xywh: tuple  # normalized to the page, as BioMiner works
    smiles: str
    slots: dict  # "1" -> "R1", from the voted read
    confidence: float | None
    kind: str  # full or part


@dataclass(frozen=True)
class Paper:
    paper_dir: Path
    pages: dict
    boxes: list
    areas: dict  # page -> figure and table areas, or None
    tables: list  # the layout's table blocks (_join_layout.Table)


def load_assets(src, verify=True):
    """BioMiner's prompts, formula table, font and OPSIN cache from its clone; SystemExit when unusable."""
    src = Path(src)
    if not (src / "BioMiner").is_dir():
        raise SystemExit(f"{src} is not a BioMiner clone (no BioMiner/ in it); pass --biominer-src")
    paths = {}
    for name, (relative, digest) in ASSETS.items():
        path = src / relative
        if not path.is_file():
            if name in OPTIONAL_ASSETS and not verify:
                paths[name] = None
                continue
            raise SystemExit(f"{path} is missing from the BioMiner clone")
        if verify and hashlib.sha256(path.read_bytes()).hexdigest() != digest:  # each under 1 MB
            raise SystemExit(f"{path} does not match its pinned sha256 (BioMiner 17c6161)")
        paths[name] = path
    prompts = {kind: paths[kind].read_text(encoding="utf-8").strip() for kind in ("full", "part")}
    formulas = json.loads(paths["formulas"].read_text(encoding="utf-8"))
    cache = read_opsin_cache(paths["opsin_cache"]) if paths["opsin_cache"] else {}
    return Assets(prompts, formulas, paths["font"], cache)


def read_opsin_cache(path):
    """BioMiner's name -> radical SMILES cache (sqlite), opened read-only: the clone is never written."""
    uri = Path(path).resolve().as_uri() + "?mode=ro&immutable=1"
    with contextlib.closing(sqlite3.connect(uri, uri=True)) as connection:
        return dict(connection.execute("SELECT iupac, result FROM cache WHERE result IS NOT NULL"))


def load_paper(paper_dir):
    """The paper's pages, structure boxes, layout areas and tables. Missing or malformed inputs raise."""
    paper_dir = Path(paper_dir)
    pages = {int(p["page"]): p for p in common.read_jsonl(paper_dir / "pages.jsonl")}
    voted = {v["region_id"]: v for v in common.read_jsonl(paper_dir / "voted.jsonl") if v.get("smiles")}
    boxes = make_boxes(common.read_jsonl(paper_dir / "regions.jsonl"), voted, pages)
    blocks_all = common.read_jsonl(paper_dir / "layout.jsonl")
    blocks = defaultdict(list)
    for block in blocks_all:
        blocks[int(block["page"])].append(block)
    areas = {page: page_areas(blocks[page], float(p["width_pt"]), float(p["height_pt"])) for page, p in pages.items()}
    return Paper(paper_dir, pages, boxes, areas, layout.layout_tables(blocks_all))


def make_boxes(regions, voted, pages):
    """Structure regions with a voted SMILES RDKit reads, indexed 0.. by page, top edge and left edge."""
    found = []
    for region in regions:
        vote = voted.get(region["region_id"])
        if not region.get("is_structure") or vote is None or Chem.MolFromSmiles(vote["smiles"]) is None:
            continue
        page = int(region["page"])
        width, height = float(pages[page]["width_pt"]), float(pages[page]["height_pt"])
        x0, y0, x1, y1 = (float(v) for v in region["bbox"])
        found.append((page, y0 / height, x0 / width, region, vote, (x0 / width, y0 / height, (x1 - x0) / width,
                                                                     (y1 - y0) / height)))
    found.sort(key=lambda item: item[:3])
    return [Box(index, region["region_id"], page, [float(v) for v in region["bbox"]], xywh, vote["smiles"],
                slots_for(vote), vote.get("confidence"), "part" if "*" in vote["smiles"] else "full")
            for index, (page, _, _, region, vote, xywh) in enumerate(found)]


def slots_for(vote):
    """The voted read's slot labels ("1" -> "R1"); ValueError unless they are its SMILES's [*:n]."""
    slots = vote.get("slots") or {}
    if {str(n) for n in slots} != set(re.findall(r"\[\*:(\d+)\]", vote["smiles"])):
        raise ValueError(f"voted.jsonl: the slots of {vote['region_id']} are not its SMILES's numbered R slots")
    return {str(n): str(label) for n, label in slots.items()}


def prompt_text(template, indices):
    return template.replace("<index></index>", str(list(indices)))


def page_image(paper, page, dpi=BIOMINER_DPI):
    """The rendered page, resized to dpi (BioMiner draws its 25 px labels on 200 dpi pages)."""
    info = paper.pages[page]
    path = Path(info["image"])
    path = path if path.is_absolute() else paper.paper_dir / path
    size = (round(float(info["width_pt"]) / 72 * dpi), round(float(info["height_pt"]) / 72 * dpi))
    with Image.open(path) as image:
        image = image.convert("RGB")
    return image if image.size == size else image.resize(size, Image.LANCZOS)


def load_font(path):
    # load_default(size=...) needs Pillow 10.1 or later; the biominer environment has 10.4.0 (checked
    # 2026-11-25), and runs with the asset check on always have BioMiner's arial.ttf.
    return ImageFont.truetype(str(path), FONT_SIZE) if path else ImageFont.load_default(size=FONT_SIZE)


def render_request(request, page, boxes, font):
    """The request's PNG: red boxes and white-backed red indices (draw_bbox_xywh), then the crop."""
    font = font or load_font(None)
    image = page.copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for index in request.indices:
        x, y, w, h = boxes[index].xywh
        left, top, right, bottom = x * width, y * height, (x + w) * width, (y + h) * height
        corners = [(left, top), (right, top), (right, bottom), (left, bottom), (left, top)]
        for start, end in zip(corners, corners[1:]):
            draw.line([start, end], fill=RED, width=LINE_WIDTH)
        x0, y0, x1, y1 = draw.textbbox((left, top), str(index), font=font)
        draw.rectangle((x0 - 5, y0 - 5, x1 + 5, y1 + 5), fill="white")
        draw.text((left, top), str(index), font=font, fill=RED)
    if request.crop is not None:
        x, y, w, h = request.crop
        image = image.crop((int(x * width), int(y * height), int((x + w) * width), int((y + h) * height)))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------- the endpoint

class ChatError(Exception):
    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind


class Client:
    """BioMiner-Instruct behind vLLM's OpenAI-compatible API. No proxy: 127.0.0.1 never goes to the cluster's."""

    def __init__(self, url, model=SERVED_MODEL, timeout=900.0, retries=2, backoff=10.0, sleep=time.sleep,
                 max_tokens=MAX_TOKENS):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout
        self.retries, self.backoff, self.sleep, self.max_tokens = retries, backoff, sleep, max_tokens
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def healthy(self, timeout=10.0):
        """True when the server lists the served model."""
        try:
            with self.opener.open(f"{self.url}/models", timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            return any(model.get("id") == self.model for model in body.get("data", []))
        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError, AttributeError):
            return False

    def _get(self, url):
        with self.opener.open(url, timeout=10.0) as response:
            return json.loads(response.read().decode("utf-8"))

    def identity(self):
        """The server's vLLM version and the served model's weights folder and context length, for the log."""
        found = {}
        try:
            found["vllm"] = self._get(f"{self.url.rsplit('/v1', 1)[0]}/version").get("version")
            model = next(m for m in self._get(f"{self.url}/models").get("data", []) if m.get("id") == self.model)
            found.update(root=model.get("root"), max_model_len=model.get("max_model_len"))
        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError, AttributeError,
                StopIteration) as exc:
            found["error"] = f"{type(exc).__name__}: {exc}"
        return found

    def chat(self, messages):
        payload = {"model": self.model, "messages": messages, "temperature": 0, "seed": 0,
                   "max_tokens": self.max_tokens}
        request = urllib.request.Request(f"{self.url}/chat/completions", data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise ChatError("http", f"HTTP {exc.code}") from exc
        except TimeoutError as exc:
            raise ChatError("timeout", f"no answer within {self.timeout:g} s") from exc
        except urllib.error.URLError as exc:
            kind = "timeout" if isinstance(exc.reason, TimeoutError) else "connection"
            raise ChatError(kind, str(exc.reason)) from exc
        except (http.client.HTTPException, OSError) as exc:
            raise ChatError("connection", f"{type(exc).__name__}: {exc}") from exc
        try:
            body = json.loads(raw)
            choice = body["choices"][0]
            message = choice["message"]
            return {"content": message.get("content"), "refusal": message.get("refusal"),
                    "finish_reason": choice.get("finish_reason"), "usage": body.get("usage")}
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ChatError("bad_response", f"not a chat completion ({len(raw)} bytes)") from exc


@dataclass(frozen=True)
class Answer:
    request_id: str
    items: list | None
    error: str | None
    message: str | None
    attempts: int
    text: str | None
    usage: dict | None
    seconds: float


def vision_messages(prompt, png):
    """BioMiner's image request (call_api_iamge): its system message, then the image and the prompt."""
    image = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(png).decode()}",
                                                "detail": "high"}}
    return [{"role": "system", "content": VISION_SYSTEM},
            {"role": "user", "content": [image, {"type": "text", "text": prompt}]}]


def ask(client, request_id, prompt, png, abort=None):
    """One request with retries. Every failure ends as an Answer with an error kind, never as []."""
    messages, start = vision_messages(prompt, png), time.monotonic()
    items = error = message = text = usage = None
    attempts = 0
    for attempt in range(1, client.retries + 2):
        if abort is not None and abort.is_set():
            error, message = "aborted", "the server stopped answering earlier in this run"
            break
        if attempt > 1:
            client.sleep(client.backoff * 2 ** (attempt - 2))
        attempts = attempt
        try:
            reply = client.chat(messages)
        except ChatError as exc:
            items, error, message, text = None, exc.kind, str(exc), None
            continue
        text, usage = reply["content"], reply["usage"]
        if reply["refusal"]:
            items, error, message = None, "refused", str(reply["refusal"])[:300]
            continue
        items, error = parse_answer(text)
        if reply["finish_reason"] == "length":  # a retry at temperature 0 would be cut the same way
            error, message = "truncated", f"the answer reached max_tokens ({client.max_tokens}); rows after the cut are lost"
            break
        if error is None:
            message = None
            break
        message = {"empty": "no answer, or an empty list"}.get(error, "the answer is not a JSON list")
    if error in TRANSPORT_ERRORS and not client.healthy():
        error, message = "server_down", f"{message}; the server no longer lists {client.model}"
        if abort is not None:
            abort.set()
    return Answer(request_id, items, error, message, attempts, text, usage, round(time.monotonic() - start, 3))


# ---------------------------------------------------------------- answers (BioMiner's parsing)

def _loads(text):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def repair(text):
    """correct_output_failed_for_json: a long answer that lost its last '}'."""
    text = text.strip()
    if text[-2:] == "}]" and text[-3:] != "}}]":
        return text[:-1] + "}]"
    if text[-3:] == "}]}":
        return text[:-2] + "}]"
    return None


def parse_answer(text):
    """(items, error): process_api_output (the fenced block, 'json' removed), json.loads, then repair()."""
    if text is None or not text.strip():
        return None, "empty"
    body = text
    if "`" in body:
        parts = body.replace("json", "").split("```")
        if len(parts) < 2:
            return None, "unparsable"
        body = parts[1]
    data = _loads(body)
    if data is None:
        data = _loads(repair(body) or "")
    if isinstance(data, dict) and isinstance(data.get("data"), list):  # the gpt-series answer shape
        data = data["data"]
    if not isinstance(data, list):
        return None, "unparsable"
    return (data, None) if data else ([], "empty")


# ---------------------------------------------------------------- molecules

def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol) if mol is not None else None


def as_index(value):
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def clean_label(identifier):
    if identifier is None or isinstance(identifier, (dict, list)):
        return None
    label = str(identifier).strip()
    return None if label.lower() in NO_LABEL else label


def norm_slot(name):
    return re.sub(r"<[^>]+>|[\s^_{}()]", "", str(name)).translate(SUPERSCRIPTS).upper()


def is_float(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def with_slot(smiles, slot):
    """A one-attachment group with its dummy mapped to slot, or (None, why)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "does not parse"
    dummies = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummies) != 1:
        return None, f"has {len(dummies)} attachment points, not 1"
    dummies[0].SetAtomMapNum(slot)
    dummies[0].SetIsotope(0)
    return Chem.MolToSmiles(mol), None


def from_radical(smiles, slot):
    """OPSIN's radical SMILES of a substituent name, its one radical atom bonded to [*:slot]."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "OPSIN's SMILES does not parse"
    radicals = [atom for atom in mol.GetAtoms() if atom.GetNumRadicalElectrons()]
    if len(radicals) != 1 or radicals[0].GetNumRadicalElectrons() != 1:
        return None, "is not a substituent with one attachment point"
    editable = Chem.RWMol(mol)
    editable.GetAtomWithIdx(radicals[0].GetIdx()).SetNumRadicalElectrons(0)
    dummy = Chem.Atom(0)
    dummy.SetAtomMapNum(slot)
    editable.AddBond(radicals[0].GetIdx(), editable.AddAtom(dummy), Chem.BondType.SINGLE)
    try:
        Chem.SanitizeMol(editable)
    except Exception as exc:  # RDKit raises several sanitization errors; the reason is kept
        return None, f"cannot be attached ({type(exc).__name__})"
    return Chem.MolToSmiles(editable), None


class GroupResolver:
    """A text R-group as BioMiner reads it (part_structure_data_api_putput_to_list): a box index shown in
    the image, else not a number, else the formula table, BioMiner's special names, its OPSIN cache, OPSIN."""

    def __init__(self, formulas, opsin_cache=None, opsin=None):
        self.formulas, self.cache, self.opsin = formulas, dict(opsin_cache or {}), opsin
        self.opsin_calls = 0

    def radical(self, name):
        if name not in self.cache:
            self.opsin_calls += 1
            self.cache[name] = self.opsin(name) if self.opsin else None
        return self.cache[name]

    def fragment(self, value, slot, visible):
        if isinstance(value, int) and not isinstance(value, bool):
            value = str(value)
        if not isinstance(value, str) or not value.strip():
            return None, "is empty or not text"
        value = value.strip()
        if value in visible:
            smiles, why = with_slot(visible[value], slot)
            return (smiles, None) if smiles else (None, f"box {value} {why}")
        if is_float(value):
            return None, f"'{value}' is a number, not a box shown in the image or a group"
        known = self.formulas.get(value) or SPECIAL_GROUPS.get(value)
        if known:
            smiles, why = with_slot(known, slot)
            return (smiles, None) if smiles else (None, f"'{value}' {why}")
        radical = self.radical(value)
        if radical:
            smiles, why = from_radical(radical, slot)
            return (smiles, None) if smiles else (None, f"'{value}' {why}")
        return None, f"'{value}' is not resolved (no box, formula or name OPSIN reads)"


def scaffold_slots(mol):
    """(scaffold with each dummy mapped to its slot, slot numbers) or (None, why): [*:n], [n*], or one bare *."""
    editable = Chem.RWMol(mol)
    dummies = [atom for atom in editable.GetAtoms() if atom.GetAtomicNum() == 0]
    numbers = [atom.GetAtomMapNum() or atom.GetIsotope() or None for atom in dummies]
    if len(dummies) == 1 and numbers[0] is None:
        numbers = [1]
    if not dummies or None in numbers or len(set(numbers)) != len(numbers):
        return None, "the scaffold's attachment points are missing, unnumbered or repeated"
    for atom, number in zip(dummies, numbers):
        atom.SetAtomMapNum(number)
        atom.SetIsotope(0)
    return editable.GetMol(), sorted(numbers)


def number_groups(names, slots, slot_labels):
    """R-group name -> slot: through the readers' slot labels, else BioMiner's sorted order, else why not."""
    by_label = {}
    for number, label in slot_labels.items():
        by_label.setdefault(norm_slot(label), int(number))
    mapped = {name: by_label.get(norm_slot(name)) for name in names}
    if None not in mapped.values() and sorted(mapped.values()) == slots:
        return mapped
    if len(names) == len(slots):
        return dict(zip(sorted(names), slots))
    return f"gives {len(names)} R-groups for a scaffold with {len(slots)} slots"


def build_row(scaffold, slot_labels, groups, visible, resolver):
    """(SMILES, None) for one table row, or (None, why). visible maps the image's box indices to SMILES."""
    core = Chem.MolFromSmiles(scaffold)
    if core is None:
        return None, "the scaffold's SMILES does not parse"
    core, slots = scaffold_slots(core)
    if core is None:
        return None, slots
    if not isinstance(groups, dict) or not groups:
        return None, "no R-groups given"
    numbering = number_groups(list(groups), slots, slot_labels)
    if isinstance(numbering, str):
        return None, numbering
    combined = core
    for name, value in groups.items():
        smiles, why = resolver.fragment(value, numbering[name], visible)
        if smiles is None:
            return None, f"{name} {why}"
        combined = Chem.CombineMols(combined, Chem.MolFromSmiles(smiles))
    try:
        joined = Chem.RemoveHs(Chem.molzip(combined))
    except Exception as exc:  # molzip and sanitization raise several RDKit errors; the reason is kept
        return None, f"RDKit could not join the row ({type(exc).__name__})"
    if any(atom.GetAtomicNum() == 0 for atom in joined.GetAtoms()):
        return None, "an attachment point was left unjoined"
    smiles = canonical(Chem.MolToSmiles(joined))
    return (smiles, None) if smiles else (None, "the joined molecule does not parse")


def record(paper_id, label, smiles, box, status, reasons, table=None, row=None):
    """A molecules_32b.jsonl record; a row placed in a layout table (a Table) takes the table's page and bbox."""
    placed = isinstance(table, layout.Table)
    return {"paper_id": paper_id, "label": label, "smiles": smiles, "page": table.page if placed else box.page,
            "bbox": table.bbox if placed else box.bbox, "source": "32b",
            "table_id": table.table_id if placed else table, "row": row, "status": status, "reasons": reasons,
            "confidence": box.confidence}


def answer_molecules(paper_id, request, items, boxes, resolver=None, layout_tables=(), rows=None):
    """An answer's molecules (full_ and part_structure_data_api_putput_to_list). A full entry labels a drawing
    the image showed; a part entry joins a scaffold the image showed with its R-groups, or abstains. rows
    counts each table's rows so far (from 0); pass one Counter for all of a paper's answers."""
    counts, records, rows = Counter(entries=len(items)), [], Counter() if rows is None else rows
    full = request.kind == "full"
    keys = {"index", "identifier"} if full else {"scaffold", "R-group", "identifier"}
    visible = {str(i): boxes[i].smiles for i in request.indices}
    for item in items:
        if not (isinstance(item, dict) and keys <= set(item)):
            counts["malformed"] += 1
            continue
        index = as_index(item["index" if full else "scaffold"])
        if index not in request.indices:
            counts["not_shown"] += 1
            continue
        box, label = boxes[index], clean_label(item["identifier"])
        if full:
            smiles, why, table, row = canonical(box.smiles), None, None, None
            reasons = [f"32b labelled drawing {box.region_id}"]
        else:
            smiles, why = build_row(box.smiles, box.slots, item["R-group"], visible, resolver)
            table = layout.place(label, box.page, layout_tables) or f"32b-{request.request_id}"
            key = getattr(table, "table_id", table)
            row, rows[key] = rows[key], rows[key] + 1
            reasons = [f"32b row of scaffold {box.region_id}"]
        if smiles is None:
            status = "abstained"
            reasons.append(f"not built: {why}")
        else:
            status = "accepted" if label else "review"
            reasons += [] if label else ["32b gave no identifier"]
        counts["abstained" if smiles is None else "molecules"] += 1
        records.append(record(paper_id, label, smiles, box, status, reasons, table, row))
    return records, {k: v for k, v in counts.items() if v}


# ---------------------------------------------------------------- the run

OUTPUT = "molecules_32b.jsonl"
UNUSED = {"malformed": "lack the keys the prompt asks for", "not_shown": "name a box the image did not show"}


@dataclass
class Job:
    paper_id: str
    paper_dir: Path
    error: str | None = None
    requests: list = field(default_factory=list)
    futures: list = field(default_factory=list)
    boxes: dict = field(default_factory=dict)
    tables: list = field(default_factory=list)


def submit_paper(entry, assets, client, pool, abort, font, args):
    """Load and plan one paper, render its images and queue its requests. Any failure ends in Job.error, which
    finish_paper records, so one bad paper never stops the run."""
    job = Job(entry.paper_id, entry.paper_dir)
    try:
        paper = load_paper(job.paper_dir)
        job.requests = plan_requests(job.paper_id, paper.boxes, paper.areas)
        job.boxes, job.tables = {b.index: b for b in paper.boxes}, paper.tables
        images = request_images(paper, job, font, args)
    except Exception as exc:  # the paper's input error, recorded in errors.jsonl
        job.error = common.describe(exc)
        return job
    job.futures = [pool.submit(ask, client, r.request_id, prompt_text(assets.prompts[r.kind], r.indices),
                               images[r.request_id], abort) for r in job.requests]
    return job


def request_images(paper, job, font, args):
    """Each request's PNG by request id, also kept in <paper_dir>/join_32b/ with --save-images."""
    images = {}
    for page in sorted({r.page for r in job.requests}):
        rendered = page_image(paper, page, args.image_dpi)
        for request in (r for r in job.requests if r.page == page):
            images[request.request_id] = render_request(request, rendered, job.boxes, font)
    if args.save_images and images:
        folder = job.paper_dir / STAGE
        folder.mkdir(exist_ok=True)
        for request_id, png in images.items():
            (folder / f"{request_id}.png").write_bytes(png)
    return images


def paper_failed(job, kind, message, totals):
    """A paper that gets no molecules_32b.jsonl: its error (input or internal) replaces this stage's errors."""
    common.replace_stage_errors(job.paper_dir, STAGE, [{"stage": STAGE, "paper_id": job.paper_id, "kind": kind,
                                                        "message": message}])
    (job.paper_dir / OUTPUT).unlink(missing_ok=True)
    log.error("%s: %s error (%s); no %s", job.paper_id, kind, message.split(":")[0], OUTPUT)
    totals["papers_failed"] += 1


def answer_errors(paper_id, request, answer, tally):
    """errors.jsonl entries for one request: its failure, and the counts of answer entries not used."""
    base = {"stage": STAGE, "paper_id": paper_id, "request": request.request_id}
    errors = [{**base, "kind": answer.error, "attempts": answer.attempts, "message": answer.message}] if answer.error else []
    for kind, what in UNUSED.items():
        if tally.get(kind):
            errors.append({**base, "kind": kind, "count": tally[kind], "message": f"{tally[kind]} answer entries {what}"})
    return errors


def finish_paper(job, resolver, totals):
    """Wait for a paper's answers, build its molecules and write its files. Counts only go to the log."""
    if job.error:
        paper_failed(job, "input", job.error, totals)
        return
    molecules, errors, requests, counts, rows = [], [], [], Counter(), Counter()
    for request, future in zip(job.requests, job.futures):
        answer = future.result()
        built, tally = (answer_molecules(job.paper_id, request, answer.items, job.boxes, resolver, job.tables, rows)
                        if answer.items else ([], {}))
        molecules += built
        counts.update(tally)
        errors += answer_errors(job.paper_id, request, answer, tally)
        requests.append({"request": request.request_id, "kind": request.kind, "page": request.page,
                         "indices": list(request.indices), "crop": request.crop, "attempts": answer.attempts,
                         "error": answer.error, "seconds": answer.seconds, "usage": answer.usage, "counts": tally,
                         "answer": answer.text})
        totals["prompt_tokens"] += (answer.usage or {}).get("prompt_tokens") or 0
    common.write_jsonl(job.paper_dir / f"{STAGE}_requests.jsonl", requests)
    common.replace_stage_errors(job.paper_dir, STAGE, errors)
    totals.update([f"errors_{e['kind']}" for e in errors] + ["requests"] * len(job.requests))
    if any(e["kind"] in LOST for e in errors):
        (job.paper_dir / OUTPUT).unlink(missing_ok=True)
        log.error("%s: the server stopped answering; no %s (rerun this paper)", job.paper_id, OUTPUT)
        totals["papers_lost"] += 1
        return
    common.write_jsonl(job.paper_dir / OUTPUT, molecules)
    status = Counter(m["status"] for m in molecules)
    totals.update({f"molecules_{k}": v for k, v in status.items()} | {"papers_written": 1})
    log.info("%s: %d boxes, %d requests, %d errors (%s); %d molecules (%s); answer entries %s", job.paper_id,
             len(job.boxes), len(job.requests), len(errors), ", ".join(sorted({e["kind"] for e in errors})) or "-",
             len(molecules), dict(status), dict(counts))


def finish_safely(job, resolver, totals):
    """finish_paper; an unexpected failure is the paper's internal error and the run goes on."""
    try:
        finish_paper(job, resolver, totals)
    except common.LockError:
        raise
    except Exception as exc:  # recorded in errors.jsonl and counted in papers_failed
        paper_failed(job, "internal", common.describe(exc), totals)


class Opsin:
    """py2opsin as BioMiner calls it (radicals allowed, '/' removed). A failure leaves the name unresolved; the
    first failure's message is logged and the others are counted (failures)."""

    def __init__(self, convert, scratch):
        self.convert, self.scratch, self.failures = convert, Path(scratch), 0

    def __call__(self, name):
        try:
            smiles = self.convert(chemical_name=name.replace("/", ""), output_format="SMILES", allow_radicals=True,
                                  tmp_fpath=str(self.scratch / "opsin_input.txt"))
        except Exception as exc:  # py2opsin raises on Java errors; the row then says the name is unresolved
            self.failures += 1
            if self.failures == 1:
                log.warning("OPSIN failed on an R-group name (%s); later failures are only counted",
                            common.describe(exc))
            return None
        return smiles or None


def load_opsin(scratch):
    """Opsin over py2opsin, or None outside the biominer env."""
    try:
        from py2opsin import py2opsin
    except ImportError:
        log.warning("py2opsin is not installed: R-group names outside BioMiner's tables stay unresolved")
        return None
    return Opsin(py2opsin, scratch)


def setup_logging(path):
    for handler in list(log.handlers):
        log.removeHandler(handler)
        handler.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stderr)):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True, help="work.jsonl: paper_id, pdf, paper_dir per line")
    parser.add_argument("--log", type=Path, required=True)
    store = os.environ.get("MOLSCOUT_STORE")
    parser.add_argument("--biominer-src", type=Path, default=Path(store) / "src" / "BioMiner" if store else None,
                        help="the BioMiner clone at 17c6161 (default $MOLSCOUT_STORE/src/BioMiner)")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"vLLM's OpenAI-compatible base URL (default {DEFAULT_URL})")
    parser.add_argument("--model", default=SERVED_MODEL)
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds per attempt (default 900)")
    parser.add_argument("--retries", type=int, default=2, help="retries after a failed attempt (default 2)")
    parser.add_argument("--backoff", type=float, default=10.0, help="seconds before the first retry, doubling")
    parser.add_argument("--jobs", type=int, default=8, help="requests in flight (default 8)")
    parser.add_argument("--wait", type=float, default=0.0, help="seconds to wait for the server to list the model")
    parser.add_argument("--image-dpi", type=int, default=BIOMINER_DPI)
    parser.add_argument("--save-images", action="store_true", help="keep each request's image in <paper_dir>/join_32b/")
    parser.add_argument("--skip-asset-check", action="store_true", help="tests only: do not check BioMiner's files' sha256")
    args = parser.parse_args(argv)
    if args.biominer_src is None:
        parser.error("--biominer-src is required when MOLSCOUT_STORE is not set")
    if args.jobs < 1 or args.retries < 0:
        parser.error("--jobs must be at least 1 and --retries at least 0")
    return args


def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.log)
    try:
        return run(args)
    finally:
        for handler in list(log.handlers):
            log.removeHandler(handler)
            handler.close()


def run_papers(papers, assets, client, abort, totals, args):
    """Every paper, the next paper's images rendering while this one's requests run; returns the resolver."""
    with tempfile.TemporaryDirectory(prefix="join_32b-") as scratch:
        resolver = GroupResolver(assets.formulas, assets.opsin_cache, load_opsin(scratch))
        font = load_font(assets.font)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            pending = None
            for entry in papers:
                job = submit_paper(entry, assets, client, pool, abort, font, args)
                if pending is not None:
                    finish_safely(pending, resolver, totals)
                pending = job
            if pending is not None:
                finish_safely(pending, resolver, totals)
    return resolver


def run(args):
    try:
        papers = common.read_manifest(args.manifest)
    except common.ManifestError as exc:
        log.error("bad manifest: %s", exc)
        return 2
    try:
        assets = load_assets(args.biominer_src, verify=not args.skip_asset_check)
    except (SystemExit, OSError, ValueError, sqlite3.Error) as exc:
        log.error("setup: %s", exc)
        return 1
    tools = f"{Path(sys.executable).parent}{os.pathsep}"  # java for py2opsin first on PATH, as BioMiner's run has it
    if not os.environ.get("PATH", "").startswith(tools):
        os.environ["PATH"] = tools + os.environ.get("PATH", "")
    client = Client(args.url, args.model, args.timeout, args.retries, args.backoff)
    deadline = time.monotonic() + args.wait
    while not client.healthy():
        if time.monotonic() >= deadline:
            log.error("setup: %s does not list %s (waited %g s); start serve_32b.sh first", args.url, args.model, args.wait)
            return 1
        time.sleep(5)
    log.info("server: %s", client.identity())
    abort, totals = threading.Event(), Counter()
    try:
        resolver = run_papers(papers, assets, client, abort, totals, args)
    except common.LockError as exc:
        log.error("%s", exc)
        return 1
    log.info("done: %d papers, %s; OPSIN calls %d, failures %d", len(papers), dict(sorted(totals.items())),
             resolver.opsin_calls, getattr(resolver.opsin, "failures", 0))
    if abort.is_set():
        log.error("the server stopped answering during the run; papers without %s need a rerun", OUTPUT)
        return 3
    return common.PARTIAL if totals["papers_failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
