"""The 32B joiner worker (src/molscout/pipeline/workers/join_32b.py) against a fake vLLM endpoint.

The worker is a standalone script for the biominer environment, so it is loaded by path. The fixture
paper (tests/pipeline/fixtures/b3/paper) is synthetic: one page, a figure with two drawn compounds and an
R-group table with a scaffold and one drawn R-group. The BioMiner files under fixtures/b3/biominer are
stand-ins; the real prompts are read from the BioMiner clone at run time and checked by sha256.
"""

import base64
import importlib.util
import io
import json
import logging
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image
from rdkit import Chem

REPO = Path(__file__).resolve().parents[2]
WORKER = REPO / "src" / "molscout" / "pipeline" / "workers" / "join_32b.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "b3"


def load_worker():
    spec = importlib.util.spec_from_file_location("join_32b", WORKER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["join_32b"] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


j = load_worker()


def canon(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def make_paper(tmp_path, name="paper"):
    """A copy of the fixture paper with its page image (white, at the dpi pages.jsonl states)."""
    paper_dir = tmp_path / name
    shutil.copytree(FIXTURES / "paper", paper_dir)
    (paper_dir / "pages").mkdir()
    Image.new("RGB", (850, 1100), "white").save(paper_dir / "pages" / "p1.png")
    return paper_dir


def assets():
    return j.load_assets(FIXTURES / "biominer", verify=False)


# ---------------------------------------------------------------- answer parsing (BioMiner's rules)

def test_fenced_json_answer_parses():
    text = '```json\n[{"index": "0", "identifier": "1"}]\n```'
    assert j.parse_answer(text) == ([{"index": "0", "identifier": "1"}], None)


def test_plain_json_answer_parses():
    assert j.parse_answer('[{"index": "1", "identifier": "2"}]') == ([{"index": "1", "identifier": "2"}], None)


def test_answer_missing_its_last_brace_is_repaired_as_biominer_does():
    text = '[{"scaffold": "2", "R-group": {"R1": "Cl"}, "identifier": "3a"}, {"scaffold": "2", "R-group": {"R1": "F", "identifier": "3b"}]'
    items, error = j.parse_answer(text)
    assert error is None
    assert items[1]["R-group"] == {"R1": "F", "identifier": "3b"}


def test_data_wrapped_answer_is_unwrapped():
    assert j.parse_answer('{"data": [{"index": "0", "identifier": "1"}]}') == ([{"index": "0", "identifier": "1"}], None)


def test_empty_list_is_flagged_empty_not_silently_accepted():
    assert j.parse_answer("```json\n[]\n```") == ([], "empty")


@pytest.mark.parametrize("text", ["I cannot help with that.", "```json [{oops", "", "   ", '{"index": "0"}'])
def test_text_that_is_not_a_json_list_is_unparsable(text):
    items, error = j.parse_answer(text)
    assert items is None and error in {"unparsable", "empty"}


# ---------------------------------------------------------------- loading and request planning

def test_boxes_get_global_indices_in_page_then_top_order_and_skip_non_structures(tmp_path):
    paper = j.load_paper(make_paper(tmp_path))
    assert [(b.index, b.region_id, b.kind) for b in paper.boxes] == [
        (0, "p1-r1", "full"), (1, "p1-r2", "full"), (2, "p1-r3", "part"), (3, "p1-r4", "part")]
    scaffold = paper.boxes[2]
    assert scaffold.slots == {"1": "R1", "2": "R2"}  # the voted read's slots
    assert scaffold.xywh == pytest.approx((90 / 612, 445 / 792, 160 / 612, 115 / 792))
    assert scaffold.bbox == [90.0, 445.0, 250.0, 560.0]
    assert [(t.table_id, t.page, t.bbox) for t in paper.tables] == [("p1-t5", 1, [72.0, 440.0, 540.0, 700.0])]
    assert paper.tables[0].labels == {"3a", "3b", "3c"}  # "Compd" is a generic word, no label


@pytest.mark.parametrize("slots", [{}, {"1": "R1"}, {"1": "R1", "2": "R2", "3": "R3"}])
def test_voted_slots_that_are_not_the_smiles_slots_make_the_paper_unusable(tmp_path, slots):
    paper_dir = make_paper(tmp_path)
    voted = read_jsonl(paper_dir / "voted.jsonl")
    voted[2]["slots"] = slots
    (paper_dir / "voted.jsonl").write_text("".join(json.dumps(v) + "\n" for v in voted))
    with pytest.raises(ValueError, match="p1-r3"):
        j.load_paper(paper_dir)


# ---------------------------------------------------------------- placing table rows

def table_block(page, index, rows, bbox=(72, 400, 540, 700)):
    html = "<table>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows) + "</table>"
    return {"page": page, "index": index, "type": "table", "bbox": list(bbox), "html": html}


def test_a_row_is_placed_by_the_first_cell_of_a_table_row_on_the_scaffolds_page_first():
    tables = j.layout.layout_tables([
        table_block(2, 4, [["Compd", "R", "IC50"], ["5", "Cl", "3"]]),
        table_block(3, 1, [["Cpd", "R"], ["Compound 3 (ZD1839)", "F"], ["5", "H"]]),
    ])
    assert j.layout.place("5", 2, tables).table_id == "p2-t4"
    assert j.layout.place("compound 3", 2, tables).table_id == "p3-t1"  # the table starts on the next page
    assert j.layout.place("gefitinib;ZD1839", 3, tables) is None  # first word or whole cell only
    assert j.layout.place("3", 2, tables).table_id == "p3-t1"  # the IC50 of 3 on page 2 is no label
    assert j.layout.place("5", 4, tables) is None and j.layout.place(None, 2, tables) is None


def test_figure_and_table_areas_include_their_captions_enlarged_by_a_32nd():
    paper_blocks = [json.loads(line) for line in (FIXTURES / "paper" / "layout.jsonl").read_text().splitlines()]
    areas = j.page_areas(paper_blocks, 612.0, 792.0)
    pad = 1 / 32
    table = (72 / 612 - pad, 420 / 792 - pad, (540 - 72) / 612 + 2 * pad, (700 - 420) / 792 + 2 * pad)
    figure = (72 / 612 - pad, 110 / 792 - pad, (540 - 72) / 612 + 2 * pad, (320 - 110) / 792 + 2 * pad)
    assert areas == [pytest.approx(table), pytest.approx(figure)]  # tables first, as BioMiner


def test_an_uncaptioned_figure_outside_tables_disables_layout_areas_for_the_page():
    blocks = [{"page": 1, "index": 0, "type": "image", "bbox": [10, 10, 100, 100], "parent": None}]
    assert j.page_areas(blocks, 612.0, 792.0) is None


def test_requests_follow_biominer_layout_split_and_group_rules(tmp_path):
    paper = j.load_paper(make_paper(tmp_path))
    requests = j.plan_requests("paper", paper.boxes, paper.areas)
    assert [(r.request_id, r.kind, r.page, r.indices) for r in requests] == [
        ("p1-full-0", "full", 1, (0, 1)), ("p1-part-0", "part", 1, (2, 3))]
    assert requests[0].crop == pytest.approx(paper.areas[1][1])  # the figure area
    assert requests[1].crop == pytest.approx(paper.areas[1][0])  # the table area


def box(index, y, kind="full", x=0.1, page=1):
    return j.Box(index=index, region_id=f"p{page}-r{index}", page=page, bbox=[0, 0, 1, 1], xywh=(x, y, 0.1, 0.05),
                 smiles="C" if kind == "full" else "[*:1]C", slots={}, confidence=None, kind=kind)


def test_more_than_four_full_boxes_in_one_area_are_asked_four_at_a_time():
    boxes = [box(i, 0.1 + 0.06 * i) for i in range(6)]
    requests = j.plan_requests("p", boxes, {1: [(0.0, 0.0, 1.0, 1.0)]})
    assert [r.indices for r in requests] == [(0, 1, 2, 3), (4, 5)]
    assert all(r.crop == (0.0, 0.0, 1.0, 1.0) for r in requests)


def test_without_layout_areas_full_groups_are_cropped_to_their_extent_enlarged_by_an_eighth():
    boxes = [box(0, 0.5, x=0.3), box(1, 0.6, x=0.3), box(2, 0.01, x=0.95)]
    first, = j.plan_requests("p", boxes[:2], {1: None})
    assert first.crop == pytest.approx((0.3 - 0.125, 0.5 - 0.125, 0.1 + 0.25, 0.15 + 0.25))
    clipped, = j.plan_requests("p", boxes[2:], {1: None})
    assert clipped.crop == pytest.approx((0.95 - 0.125, 0.0, 1.0 - 0.825, 0.06 + 0.125))  # kept on the page


def test_a_box_outside_every_area_sends_the_whole_page_uncropped():
    boxes = [box(0, 0.1), box(1, 0.8)]
    (request,) = j.plan_requests("p", boxes, {1: [(0.0, 0.0, 1.0, 0.5)]})
    assert request.indices == (0, 1) and request.crop is None


def test_part_boxes_of_an_area_go_in_one_request_however_many():
    boxes = [box(i, 0.05 * i, kind="part") for i in range(7)]
    (request,) = j.plan_requests("p", boxes, {1: None})
    assert request.kind == "part" and request.indices == tuple(range(7)) and request.crop is None


def test_prompt_lists_the_box_indices_as_biominer_does():
    assert j.prompt_text("Indices: <index></index>.", (0, 3)) == "Indices: [0, 3]."


def test_request_image_is_the_crop_at_200_dpi_with_red_numbered_boxes(tmp_path):
    paper = j.load_paper(make_paper(tmp_path))
    request = j.plan_requests("paper", paper.boxes, paper.areas)[0]
    page = j.page_image(paper, 1, dpi=200)
    assert page.size == (1700, 2200)
    png = j.render_request(request, page, {b.index: b for b in paper.boxes}, font=None)
    image = Image.open(io.BytesIO(png)).convert("RGB")
    x, y, w, h = request.crop
    assert image.size == (int((x + w) * 1700) - int(x * 1700), int((y + h) * 2200) - int(y * 2200))
    left = int(90 / 612 * 1700) - int(x * 1700)  # the first box's left edge, below its label
    row = int(200 / 792 * 2200) - int(y * 2200)
    assert (255, 0, 0) in {image.getpixel((left + dx, row)) for dx in (-1, 0, 1)}
    assert image.getpixel((left + 40, row)) == (255, 255, 255)  # inside the box: the page, untouched


# ---------------------------------------------------------------- the endpoint (fake vLLM)

class FakeVLLM:
    """An OpenAI-compatible /v1 on 127.0.0.1: scripted replies per prompt kind (FULL / PART)."""

    def __init__(self):
        self.replies = {"FULL": [], "PART": []}
        self.default = {"FULL": "[]", "PART": "[]"}
        self.requests = []
        self.models_up = True
        handler = self._handler()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, body):
                data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/version":
                    self._send(200, {"version": "0.11.0"})
                elif fake.models_up:
                    self._send(200, {"data": [{"id": j.SERVED_MODEL, "root": "/work/models/BioMiner-Instruct",
                                               "max_model_len": 128000}]})
                else:
                    self._send(503, {"error": "down"})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(payload)
                text = payload["messages"][1]["content"][1]["text"]
                kind = "FULL" if text.startswith("FULL") else "PART"
                reply = fake.replies[kind].pop(0) if fake.replies[kind] else fake.default[kind]
                if callable(reply):
                    reply = reply()
                if isinstance(reply, tuple):  # (status, body)
                    self._send(*reply)
                    return
                if isinstance(reply, dict):  # a whole message
                    message, finish = reply, "stop"
                else:
                    message, finish = {"role": "assistant", "content": reply}, "stop"
                if isinstance(reply, dict) and reply.get("finish_reason"):
                    finish = reply.pop("finish_reason")
                self._send(200, {"choices": [{"message": message, "finish_reason": finish}],
                                 "usage": {"prompt_tokens": 1000, "completion_tokens": 20}})

        return Handler

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def vllm():
    fake = FakeVLLM()
    yield fake
    fake.close()


def client(url, **kw):
    return j.Client(url, j.SERVED_MODEL, timeout=kw.pop("timeout", 5), retries=kw.pop("retries", 2),
                    sleep=lambda s: None, **kw)


def test_request_payload_matches_biominers_vision_call(vllm):
    vllm.replies["FULL"] = ['[{"index": "0", "identifier": "1"}]']
    answer = j.ask(client(vllm.url), "r", "FULL prompt [0]", b"png-bytes")
    assert answer.items == [{"index": "0", "identifier": "1"}] and answer.error is None and answer.attempts == 1
    payload = vllm.requests[0]
    assert payload["model"] == "local-biominer-instruct"
    assert payload["temperature"] == 0 and payload["seed"] == 0 and payload["max_tokens"] == 4096
    system, user = payload["messages"]
    assert system == {"role": "system", "content": j.VISION_SYSTEM}
    image, text = user["content"]
    assert image == {"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(b"png-bytes").decode(), "detail": "high"}}
    assert text == {"type": "text", "text": "FULL prompt [0]"}
    assert answer.usage == {"prompt_tokens": 1000, "completion_tokens": 20}


def test_server_error_is_retried_then_succeeds(vllm):
    vllm.replies["FULL"] = [(500, {"error": "boom"}), '[{"index": "0", "identifier": "1"}]']
    answer = j.ask(client(vllm.url), "r", "FULL", b"x")
    assert answer.error is None and answer.attempts == 2


@pytest.mark.parametrize("reply, kind", [
    ((500, {"error": "boom"}), "http"),
    ("", "empty"),
    ("[]", "empty"),
    ("Sorry, I cannot read this image.", "unparsable"),
    ({"role": "assistant", "content": None, "refusal": "no"}, "refused"),
    ((200, "not json"), "bad_response"),
])
def test_failed_or_empty_answers_are_retried_then_reported_with_their_kind(vllm, reply, kind):
    vllm.default["FULL"] = reply
    answer = j.ask(client(vllm.url, retries=2), "r", "FULL", b"x")
    assert answer.error == kind and answer.attempts == 3 and len(vllm.requests) == 3
    assert answer.items == ([] if kind == "empty" and reply == "[]" else None)


def test_truncated_answer_is_kept_but_reported_without_pointless_retries(vllm):
    vllm.replies["FULL"] = [{"role": "assistant", "content": '[{"index": "0", "identifier": "1"}]',
                             "finish_reason": "length"}]
    answer = j.ask(client(vllm.url), "r", "FULL", b"x")
    assert answer.items == [{"index": "0", "identifier": "1"}] and answer.error == "truncated"
    assert answer.attempts == 1


def test_timeout_is_retried_and_reported(vllm):
    vllm.default["FULL"] = lambda: time.sleep(1.0) or "[]"
    answer = j.ask(client(vllm.url, timeout=0.2, retries=1), "r", "FULL", b"x")
    assert answer.error == "timeout" and answer.attempts == 2


def test_unreachable_server_is_reported_as_server_down():
    answer = j.ask(client("http://127.0.0.1:9/v1", retries=1), "r", "FULL", b"x")
    assert answer.error == "server_down"


def test_health_check_needs_the_served_model(vllm):
    assert client(vllm.url).healthy()
    vllm.models_up = False
    assert not client(vllm.url).healthy()


# ---------------------------------------------------------------- building molecules

def resolver(opsin=None, cache=None):
    return j.GroupResolver(assets().formulas, opsin_cache=cache or {}, opsin=opsin)


SCAFFOLD = "[*:1]c1ccc([*:2])cc1"
SLOTS = {"1": "R1", "2": "R2"}


@pytest.mark.parametrize("groups, expected", [
    ({"R1": "Cl", "R2": "H"}, "Clc1ccccc1"),
    ({"R1": "Me", "R2": "OMe"}, "COc1ccc(C)cc1"),
    ({"R<sup>1</sup>": "F", "R²": "CF3"}, "Fc1ccc(C(F)(F)F)cc1"),
    ({"R1": "7", "R2": "H"}, "c1ccc(C2CC2)cc1"),  # a drawn R-group: box 7
    ({"R1": "nitro", "R2": "H"}, "O=[N+]([O-])c1ccccc1"),  # BioMiner's special names
])
def test_rows_build_from_formulas_names_and_drawn_groups(groups, expected):
    smiles, reason = j.build_row(SCAFFOLD, SLOTS, groups, {"7": "[*:1]C1CC1"}, resolver())
    assert reason is None and smiles == canon(expected)


def test_names_resolve_through_biominers_opsin_cache_then_opsin():
    calls = []

    def opsin(name):
        calls.append(name)
        return {"ethyl": "[CH2]C"}.get(name)

    rows = resolver(opsin=opsin, cache={"2-pyridyl": "N1=[C]C=CC=C1"})
    assert j.build_row(SCAFFOLD, SLOTS, {"R1": "2-pyridyl", "R2": "ethyl"}, {}, rows)[0] == canon("CCc1ccc(-c2ccccn2)cc1")
    assert j.build_row(SCAFFOLD, SLOTS, {"R1": "ethyl", "R2": "H"}, {}, rows)[0] == canon("CCc1ccccc1")
    assert calls == ["ethyl"]  # cached after the first call


@pytest.mark.parametrize("groups, reason", [
    ({"R1": "1.5", "R2": "H"}, "number"),
    ({"R1": "Cl"}, "slots"),
    ({"R1": "Cl", "R2": "H", "R3": "H"}, "slots"),
    ({"R1": "unobtainium", "R2": "H"}, "not resolved"),
    ({"R1": "carbonyl", "R2": "H"}, "attachment"),
    ({"R1": "9", "R2": "H"}, "number"),  # not a box shown in the image, and a number is no group
])
def test_rows_that_cannot_be_built_say_why(groups, reason):
    smiles, why = j.build_row(SCAFFOLD, SLOTS, groups, {}, resolver())
    assert smiles is None and reason in why


def test_without_a_slot_map_names_are_numbered_in_sorted_order_as_biominer_does():
    smiles, _ = j.build_row("[*:1]c1ccc([*:2])cc1", {}, {"Y": "OMe", "X": "Cl"}, {}, resolver())
    assert smiles == canon("COc1ccc(Cl)cc1")


def test_full_answer_becomes_labelled_drawn_molecules(tmp_path):
    paper = j.load_paper(make_paper(tmp_path))
    request = j.plan_requests("paper", paper.boxes, paper.areas)[0]
    items = [{"index": "0", "identifier": "1"}, {"index": 1, "identifier": ""},
             {"index": "3", "identifier": "9"}, {"index": "42", "identifier": "x"}, "junk"]
    records, counts = j.answer_molecules("paper", request, items, {b.index: b for b in paper.boxes})
    assert [(m["label"], m["status"], m["page"], m["bbox"], m["source"]) for m in records] == [
        ("1", "accepted", 1, [90.0, 130.0, 250.0, 280.0], "32b"),
        (None, "review", 1, [300.0, 130.0, 500.0, 280.0], "32b")]
    assert records[0]["smiles"] == canon("CC(=O)Oc1ccccc1C(=O)O") and records[0]["confidence"] == 0.9
    assert counts == {"entries": 5, "molecules": 2, "not_shown": 2, "malformed": 1}


def test_part_answer_becomes_table_rows_including_abstentions(tmp_path):
    paper = j.load_paper(make_paper(tmp_path))
    request = j.plan_requests("paper", paper.boxes, paper.areas)[1]
    items = [{"scaffold": "2", "R-group": {"R1": "Cl", "R2": "H"}, "identifier": "3a"},
             {"scaffold": "2", "R-group": {"R1": "3", "R2": "OMe"}, "identifier": "3b"},
             {"scaffold": "2", "R-group": {"R1": "1.5", "R2": "H"}, "identifier": "3c"},
             {"scaffold": "0", "R-group": {"R1": "H"}, "identifier": "x"}]
    records, counts = j.answer_molecules("paper", request, items, {b.index: b for b in paper.boxes}, resolver(),
                                         paper.tables)
    assert [(m["label"], m["status"], m["table_id"], m["row"], m["page"], m["bbox"]) for m in records] == [
        ("3a", "accepted", "p1-t5", 0, 1, [72.0, 440.0, 540.0, 700.0]),
        ("3b", "accepted", "p1-t5", 1, 1, [72.0, 440.0, 540.0, 700.0]),
        ("3c", "abstained", "p1-t5", 2, 1, [72.0, 440.0, 540.0, 700.0])]
    assert records[0]["smiles"] == canon("Clc1ccccc1") and records[1]["smiles"] == canon("COc1ccc(C2CC2)cc1")
    assert records[2]["smiles"] is None and "number" in records[2]["reasons"][-1]
    assert counts == {"entries": 4, "molecules": 2, "abstained": 1, "not_shown": 1}


def test_a_core_voted_by_vote_paper_reaches_the_joiner_with_its_slot_labels(tmp_path):
    from molscout.pipeline.vote import vote_paper

    paper_dir = make_paper(tmp_path)
    reads = {"p1-r1": ("CC(=O)Oc1ccccc1C(=O)O", {}), "p1-r2": ("c1ccc2ccccc2c1", {}),
             "p1-r3": ("[*:1]c1ccc([*:2])cc1", {"1": "R1", "2": "R2"}), "p1-r4": ("[*:1]C1CC1", {"1": "*"})}
    (paper_dir / "reads.jsonl").write_text("".join(
        json.dumps({"region_id": region, "reader": reader, "smiles": smiles, "slots": slots, "error": None}) + "\n"
        for region, (smiles, slots) in reads.items() for reader in ("molglyph", "molscribe", "molnextr")))
    vote_paper(paper_dir)
    paper = j.load_paper(paper_dir)
    scaffold = next(b for b in paper.boxes if b.region_id == "p1-r3")
    assert (scaffold.kind, scaffold.slots) == ("part", {"1": "R1", "2": "R2"})
    request = next(r for r in j.plan_requests("paper", paper.boxes, paper.areas) if r.kind == "part")
    items = [{"scaffold": str(scaffold.index), "R-group": {"R2": "OMe", "R1": "Cl"}, "identifier": "3a"}]
    (row,), _ = j.answer_molecules("paper", request, items, {b.index: b for b in paper.boxes}, resolver(),
                                   paper.tables)
    assert (row["smiles"], row["table_id"], row["row"], row["status"]) == (canon("COc1ccc(Cl)cc1"), "p1-t5", 0,
                                                                          "accepted")


def test_rows_not_in_a_layout_table_keep_the_scaffolds_place_and_count_from_0_per_table(tmp_path):
    paper = j.load_paper(make_paper(tmp_path))
    request = j.plan_requests("paper", paper.boxes, paper.areas)[1]
    boxes, rows = {b.index: b for b in paper.boxes}, j.Counter()
    items = [{"scaffold": "2", "R-group": {"R1": "Cl", "R2": "H"}, "identifier": "9x"},
             {"scaffold": "2", "R-group": {"R1": "F", "R2": "H"}, "identifier": "3b"}]
    first, _ = j.answer_molecules("paper", request, items, boxes, resolver(), paper.tables, rows)
    again, _ = j.answer_molecules("paper", request, items[1:], boxes, resolver(), paper.tables, rows)
    assert [(m["table_id"], m["row"], m["page"], m["bbox"]) for m in first] == [
        ("32b-p1-part-0", 0, 1, [90.0, 445.0, 250.0, 560.0]), ("p1-t5", 0, 1, [72.0, 440.0, 540.0, 700.0])]
    assert [(m["table_id"], m["row"]) for m in again] == [("p1-t5", 1)]


# ---------------------------------------------------------------- assets

def test_assets_are_checked_against_their_pinned_hashes():
    with pytest.raises(SystemExit, match="sha256"):
        j.load_assets(FIXTURES / "biominer", verify=True)
    loaded = assets()
    assert loaded.prompts["full"].startswith("FULL") and loaded.formulas["OMe"] == "*OC"


def test_missing_biominer_clone_is_a_setup_error(tmp_path):
    with pytest.raises(SystemExit, match="BioMiner"):
        j.load_assets(tmp_path / "nowhere", verify=False)


# ---------------------------------------------------------------- the worker end to end

FULL_REPLY = '```json\n[{"index": "0", "identifier": "1"}, {"index": "1", "identifier": "2"}]\n```'
PART_REPLY = ('```json\n[{"scaffold": "2", "R-group": {"R1": "Cl", "R2": "H"}, "identifier": "3a"}, '
              '{"scaffold": "2", "R-group": {"R1": "3", "R2": "OMe"}, "identifier": "3b"}]\n```')


def run_worker(tmp_path, vllm, papers, *extra):
    manifest = tmp_path / "work.jsonl"
    manifest.write_text("".join(json.dumps({"paper_id": p.name, "pdf": str(p / "paper.pdf"), "paper_dir": str(p)}) + "\n"
                                for p in papers))
    return j.main(["--manifest", str(manifest), "--log", str(tmp_path / "join_32b.log"), "--url", vllm.url,
                   "--biominer-src", str(FIXTURES / "biominer"), "--skip-asset-check", "--retries", "1",
                   "--backoff", "0", "--jobs", "2", *extra])


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_worker_writes_32b_molecules_and_never_prints_smiles(tmp_path, vllm, capfd):
    paper_dir = make_paper(tmp_path)
    vllm.replies = {"FULL": [FULL_REPLY], "PART": [PART_REPLY]}
    assert run_worker(tmp_path, vllm, [paper_dir]) == 0
    molecules = read_jsonl(paper_dir / "molecules_32b.jsonl")
    assert [(m["label"], m["source"], m["status"]) for m in molecules] == [
        ("1", "32b", "accepted"), ("2", "32b", "accepted"), ("3a", "32b", "accepted"), ("3b", "32b", "accepted")]
    assert set(molecules[0]) == {"paper_id", "label", "smiles", "page", "bbox", "source", "table_id", "row",
                                 "status", "reasons", "confidence"}
    assert not (paper_dir / "errors.jsonl").exists()
    log = read_jsonl(paper_dir / "join_32b_requests.jsonl")
    assert [(r["request"], r["error"], r["attempts"]) for r in log] == [("p1-full-0", None, 1), ("p1-part-0", None, 1)]
    out, err = capfd.readouterr()
    text = out + err + (tmp_path / "join_32b.log").read_text()
    for m in molecules:
        assert m["smiles"] not in text
    assert "c1" not in out and "c1" not in err


def test_failed_requests_go_to_errors_jsonl_and_the_rest_is_kept(tmp_path, vllm):
    paper_dir = make_paper(tmp_path)
    (paper_dir / "errors.jsonl").write_text(json.dumps({"stage": "segment", "paper_id": "paper", "kind": "x"}) + "\n")
    vllm.replies = {"FULL": [FULL_REPLY], "PART": ["no table here", "still no"]}
    assert run_worker(tmp_path, vllm, [paper_dir]) == 0
    assert [m["label"] for m in read_jsonl(paper_dir / "molecules_32b.jsonl")] == ["1", "2"]
    errors = read_jsonl(paper_dir / "errors.jsonl")
    assert errors[0]["stage"] == "segment"  # other stages' errors stay
    assert [(e["stage"], e["kind"], e["request"], e["attempts"]) for e in errors[1:]] == [
        ("join_32b", "unparsable", "p1-part-0", 2)]


def test_rerun_replaces_this_stages_errors(tmp_path, vllm):
    paper_dir = make_paper(tmp_path)
    vllm.default = {"FULL": "[]", "PART": "[]"}
    run_worker(tmp_path, vllm, [paper_dir])
    assert len(read_jsonl(paper_dir / "errors.jsonl")) == 2
    vllm.default = {"FULL": FULL_REPLY, "PART": PART_REPLY}
    run_worker(tmp_path, vllm, [paper_dir])
    assert not (paper_dir / "errors.jsonl").exists() or read_jsonl(paper_dir / "errors.jsonl") == []


def test_paper_without_structures_asks_nothing_and_writes_an_empty_file(tmp_path, vllm):
    paper_dir = make_paper(tmp_path)
    (paper_dir / "voted.jsonl").write_text("")
    assert run_worker(tmp_path, vllm, [paper_dir]) == 0
    assert read_jsonl(paper_dir / "molecules_32b.jsonl") == [] and vllm.requests == []


def test_missing_inputs_are_a_paper_error_not_a_crash(tmp_path, vllm):
    broken, fine = make_paper(tmp_path, "broken"), make_paper(tmp_path, "fine")
    (broken / "pages.jsonl").unlink()
    vllm.default = {"FULL": FULL_REPLY, "PART": PART_REPLY}
    assert run_worker(tmp_path, vllm, [broken, fine]) == 4
    assert not (broken / "molecules_32b.jsonl").exists()
    assert read_jsonl(broken / "errors.jsonl")[0]["kind"] == "input"
    assert len(read_jsonl(fine / "molecules_32b.jsonl")) == 4


def test_unexpected_failures_stay_with_their_paper(tmp_path, vllm, monkeypatch):
    zero, crash, fine = (make_paper(tmp_path, name) for name in ("zero", "crash", "fine"))
    (zero / "pages.jsonl").write_text(json.dumps({"page": 1, "width_pt": 0, "height_pt": 792.0,
                                                   "image": "pages/p1.png"}) + "\n")  # ZeroDivisionError
    real = j.answer_molecules

    def answer(paper_id, *args, **kw):
        if paper_id == "crash":
            raise AttributeError("'NoneType' object has no attribute 'get'")
        return real(paper_id, *args, **kw)

    monkeypatch.setattr(j, "answer_molecules", answer)
    vllm.default = {"FULL": FULL_REPLY, "PART": PART_REPLY}
    assert run_worker(tmp_path, vllm, [zero, crash, fine]) == 4
    assert [e["kind"] for e in read_jsonl(zero / "errors.jsonl")] == ["input"]
    (error,) = read_jsonl(crash / "errors.jsonl")
    assert error["kind"] == "internal" and "AttributeError" in error["message"]
    assert not (zero / "molecules_32b.jsonl").exists() and not (crash / "molecules_32b.jsonl").exists()
    assert len(read_jsonl(fine / "molecules_32b.jsonl")) == 4


def test_a_bad_manifest_is_a_setup_error(tmp_path, vllm):
    manifest = tmp_path / "work.jsonl"
    manifest.write_text(json.dumps({"paper_id": "a", "pdf": "a.pdf"}) + "\n")
    assert j.main(["--manifest", str(manifest), "--log", str(tmp_path / "j.log"), "--url", vllm.url,
                   "--biominer-src", str(FIXTURES / "biominer"), "--skip-asset-check"]) == 2
    assert "paper_dir" in (tmp_path / "j.log").read_text()


def test_unused_answer_entries_are_counted_in_errors_jsonl(tmp_path, vllm):
    paper_dir = make_paper(tmp_path)
    vllm.replies = {"FULL": ['[{"index": "0", "identifier": "1"}, {"index": "7", "identifier": "2"}, '
                             '{"identifier": "3"}, "junk"]'], "PART": [PART_REPLY]}
    assert run_worker(tmp_path, vllm, [paper_dir]) == 0
    assert len(read_jsonl(paper_dir / "molecules_32b.jsonl")) == 3
    errors = read_jsonl(paper_dir / "errors.jsonl")
    assert [(e["kind"], e["request"], e["count"]) for e in errors] == [("malformed", "p1-full-0", 2),
                                                                       ("not_shown", "p1-full-0", 1)]


def test_the_log_names_the_servers_vllm_version_and_model(tmp_path, vllm):
    vllm.default = {"FULL": FULL_REPLY, "PART": PART_REPLY}
    run_worker(tmp_path, vllm, [make_paper(tmp_path)])
    text = (tmp_path / "join_32b.log").read_text()
    assert "'vllm': '0.11.0'" in text and "/work/models/BioMiner-Instruct" in text and "128000" in text


def test_opsin_failures_are_counted_and_the_first_is_logged_once(tmp_path):
    def convert(**kw):
        raise RuntimeError("Java was not found")

    messages = []
    handler = logging.Handler()
    handler.emit = lambda record: messages.append(record.getMessage())
    j.log.addHandler(handler)
    try:
        opsin = j.Opsin(convert, tmp_path)
        assert opsin("ethyl") is None and opsin("propyl") is None
    finally:
        j.log.removeHandler(handler)
    assert opsin.failures == 2
    assert messages == [
        "OPSIN failed on an R-group name (RuntimeError: Java was not found); later failures are only counted"]


def test_biominers_opsin_cache_is_read_and_closed(tmp_path):
    import sqlite3

    path = tmp_path / "iupac_cache.db"
    with j.contextlib.closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("CREATE TABLE cache (iupac TEXT, result TEXT)")
        connection.executemany("INSERT INTO cache VALUES (?, ?)", [("ethyl", "[CH2]C"), ("bad", None)])
    assert j.read_opsin_cache(path) == {"ethyl": "[CH2]C"}
    path.unlink()  # no handle left open on it


def test_server_not_up_is_a_fatal_setup_error(tmp_path, vllm):
    vllm.models_up = False
    assert run_worker(tmp_path, vllm, [make_paper(tmp_path)]) == 1


def test_biominer_files_missing_is_a_fatal_setup_error(tmp_path, vllm):
    manifest = tmp_path / "work.jsonl"
    manifest.write_text(json.dumps({"paper_id": "a", "pdf": "a.pdf", "paper_dir": str(tmp_path / "a")}) + "\n")
    assert j.main(["--manifest", str(manifest), "--log", str(tmp_path / "j.log"), "--url", vllm.url,
                   "--biominer-src", str(tmp_path / "nowhere")]) == 1


def test_server_lost_mid_run_leaves_the_paper_unwritten_and_fails_the_run(tmp_path, vllm):
    paper_dir = make_paper(tmp_path)

    def die():
        vllm.models_up = False
        return (503, {"error": "engine dead"})

    vllm.replies = {"FULL": [die, die], "PART": [die, die]}
    vllm.default = {"FULL": (503, {}), "PART": (503, {})}
    assert run_worker(tmp_path, vllm, [paper_dir]) == 3
    assert not (paper_dir / "molecules_32b.jsonl").exists()
    assert {e["kind"] for e in read_jsonl(paper_dir / "errors.jsonl")} <= {"server_down", "aborted"}


def test_save_images_keeps_each_request_image(tmp_path, vllm):
    paper_dir = make_paper(tmp_path)
    vllm.default = {"FULL": FULL_REPLY, "PART": PART_REPLY}
    run_worker(tmp_path, vllm, [paper_dir], "--save-images")
    assert sorted(p.name for p in (paper_dir / "join_32b").iterdir()) == ["p1-full-0.png", "p1-part-0.png"]


# ---------------------------------------------------------------- serve_32b.sh

SERVE = WORKER.parent / "serve_32b.sh"
FAKE_PYTHON = """#!/bin/sh
case "$1" in
  -c) echo "0.11.0"; exit 0 ;;
  -) cat >/dev/null; echo "staged to $3" >> "$SERVE_OUT.stage"
     printf 'BIOMINER_PYTHON=%s\\nVLLM_PYTHON=%s\\nBIOMINER_INSTRUCT=%s\\n' "$4" "$5" "$6"; exit 0 ;;
esac
{ echo "args: $*"; env | grep -i -E '^(no_)?proxy|_proxy=|^HF_HUB_OFFLINE|^VLLM_' | sort; } > "$SERVE_OUT"
"""


def fake_store(tmp_path):
    store = tmp_path / "store"
    for env in ("biominer", "biominer_vllm"):
        python = store / "envs" / env / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_text(FAKE_PYTHON)
        python.chmod(0o755)
    (store / "models" / "biominer" / "BioMiner-Instruct").mkdir(parents=True)
    return store


def serve(tmp_path, *args, store=None, **env):
    environment = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "SERVE_OUT": str(tmp_path / "served.txt"), **env}
    if store is not None:
        environment["MOLSCOUT_STORE"] = str(store)
    return subprocess.run(["bash", str(SERVE), *args], capture_output=True, text=True, env=environment, timeout=30)


def test_serve_uses_the_benchmarks_proven_vllm_flags(tmp_path):
    result = serve(tmp_path, "--dry-run", "--no-stage", "--work", str(tmp_path / "w"), store=fake_store(tmp_path))
    assert result.returncode == 0, result.stderr
    command = result.stdout
    for flag in ("-m vllm.entrypoints.openai.api_server", "--served-model-name local-biominer-instruct",
                 "--trust-remote-code", "--tensor-parallel-size 1", "--gpu-memory-utilization 0.8",
                 "--max-model-len 128000", "--swap-space 32", "--port 8613", "--mm-encoder-tp-mode data",
                 "--async-scheduling", "--host 127.0.0.1", "-u http_proxy", "-u HTTPS_PROXY",
                 "no_proxy=127.0.0.1\\,localhost", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1"):
        assert flag in command, flag
    assert "BioMiner-Instruct" in command and not (tmp_path / "w").exists()


def test_serve_starts_vllm_offline_and_records_the_pythons(tmp_path):
    store, work = fake_store(tmp_path), tmp_path / "w"
    result = serve(tmp_path, "--no-stage", "--work", str(work), "--gpu-memory-utilization", "0.9", store=store,
                   http_proxy="http://proxy:3128", HTTPS_PROXY="http://proxy:3128")
    assert result.returncode == 0, result.stderr
    served = (tmp_path / "served.txt").read_text()
    assert "--gpu-memory-utilization 0.9" in served and f"--model {store}/models/biominer/BioMiner-Instruct" in served
    assert "proxy:3128" not in served and "no_proxy=127.0.0.1,localhost" in served
    assert f"VLLM_CACHE_ROOT={work}/vllm-cache" in served
    staged = (work / "staged.env").read_text()
    assert f"BIOMINER_PYTHON={store}/envs/biominer/bin/python" in staged


def test_serve_works_in_a_folder_of_its_own_job_by_default(tmp_path):
    result = serve(tmp_path, "--dry-run", store=fake_store(tmp_path), TMPDIR=str(tmp_path), SLURM_JOB_ID="4242")
    assert result.returncode == 0, result.stderr
    assert f"to {tmp_path}/molscout-32b-4242" in result.stdout


def staged_files(work):
    """A staged copy in work, as an earlier job of the same work folder left it, and its staged.env."""
    paths = {"BIOMINER_PYTHON": work / "envs/biominer/bin/python", "VLLM_PYTHON": work / "envs/vllm/bin/python",
             "BIOMINER_INSTRUCT": work / "models/BioMiner-Instruct"}
    paths["BIOMINER_INSTRUCT"].mkdir(parents=True)
    for name in ("BIOMINER_PYTHON", "VLLM_PYTHON"):
        paths[name].parent.mkdir(parents=True)
        paths[name].write_text(FAKE_PYTHON)
        paths[name].chmod(0o755)
    (work / "staged.env").write_text("".join(f"{name}={path}\n" for name, path in paths.items()))
    return paths


def test_serve_restages_over_what_an_interrupted_job_left(tmp_path):
    store, work = fake_store(tmp_path), tmp_path / "w"
    (work / "envs" / "biominer").mkdir(parents=True)  # unpacking stopped half way: no staged.env
    (work / "models" / "BioMiner-Instruct").mkdir(parents=True)
    result = serve(tmp_path, "--work", str(work), store=store)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "served.txt.stage").read_text() == f"staged to {work}\n"
    assert not (work / "envs").exists() and not (work / "models").exists()  # cleared before staging again
    assert f"BIOMINER_PYTHON={store}/envs/biominer/bin/python" in (work / "staged.env").read_text()


def test_serve_reuses_a_staged_copy_only_while_it_is_there(tmp_path):
    store, work = fake_store(tmp_path), tmp_path / "w"
    paths = staged_files(work)
    assert serve(tmp_path, "--work", str(work), store=store).returncode == 0
    assert not (tmp_path / "served.txt.stage").exists()
    assert f"--model {paths['BIOMINER_INSTRUCT']}" in (tmp_path / "served.txt").read_text()
    paths["VLLM_PYTHON"].unlink()
    assert serve(tmp_path, "--work", str(work), store=store).returncode == 0
    assert (tmp_path / "served.txt.stage").read_text() == f"staged to {work}\n"


def test_serve_without_staging_ignores_a_staged_copy(tmp_path):
    store, work = fake_store(tmp_path), tmp_path / "w"
    staged_files(work)
    assert serve(tmp_path, "--no-stage", "--work", str(work), store=store).returncode == 0
    assert f"--model {store}/models/biominer/BioMiner-Instruct" in (tmp_path / "served.txt").read_text()


def test_serve_logs_the_vllm_version_and_the_models_snapshot(tmp_path):
    store = fake_store(tmp_path)
    model = store / "models" / "biominer" / "BioMiner-Instruct"
    (model / "config.json").write_text('{"model_type": "qwen3_vl"}')

    def snapshot():
        result = serve(tmp_path, "--no-stage", "--work", str(tmp_path / "w"), store=store)
        assert result.returncode == 0, result.stderr
        line = next(line for line in result.stderr.splitlines() if "vLLM 0.11.0" in line)
        return line.rsplit("snapshot ", 1)[1]

    first = snapshot()
    assert len(first) == 16 and snapshot() == first
    (model / "config.json").write_text('{"model_type": "other"}')
    assert snapshot() != first


@pytest.mark.parametrize("args, store, message", [
    (["--dry-run"], None, "MOLSCOUT_STORE"),
    (["--dry-run", "--bogus"], "fake", "unknown argument"),
    (["--dry-run"], "empty", "does not exist"),
    (["--dry-run", "--work", ""], "fake", "--work"),
])
def test_serve_refuses_bad_setups(tmp_path, args, store, message):
    stores = {None: None, "fake": fake_store(tmp_path), "empty": tmp_path / "nothing"}
    result = serve(tmp_path, *args, store=stores[store])
    assert result.returncode != 0 and message in result.stderr
