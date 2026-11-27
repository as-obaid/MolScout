"""Fakes for pipeline tests: a config whose workers are tests/fixtures/pipeline/fake_worker.py, and in-process
steps that write plausible records without the chemistry modules."""

from __future__ import annotations

import json
import sys
import threading
from collections import Counter
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from molscout.pipeline.config import load_pipeline_config
from molscout.pipeline.inputs import Paper

FAKE_WORKER = Path(__file__).parent / "fixtures" / "pipeline" / "fake_worker.py"
WORKER_STAGES = ("parse", "segment", "read_molglyph", "read_molscribe", "read_molnextr", "join_32b")


def pipeline_yaml(
    tmp_path: Path, extra: str = "", worker_args: dict[str, list[str]] | None = None, ocr: bool = False
) -> Path:
    """A pipeline config whose every worker is the fake, with --stage and any extra arguments per stage; with `ocr`
    it names an ocr_labels worker too, so the label OCR stage runs."""
    lines = ["workers:"]
    for stage in (*WORKER_STAGES, "ocr_labels") if ocr else WORKER_STAGES:
        args = ["--stage", stage, *(worker_args or {}).get(stage, [])]
        if stage == "join_32b":
            args += ["--url", "{vllm_url}"]
        lines.append(f"  {stage}: {{python: {json.dumps(sys.executable)}, worker: {json.dumps(str(FAKE_WORKER))}, "
                     f"args: {json.dumps(args)}}}")
    jar = tmp_path / "opsin.jar"
    jar.write_text("not a jar")
    path = tmp_path / "pipeline.yaml"
    path.write_text("\n".join(lines) + f"\nopsin_jar: {jar}\n" + extra)
    return path


def fake_config(
    tmp_path: Path, extra: str = "", worker_args: dict[str, list[str]] | None = None, ocr: bool = False, **overrides
):
    return load_pipeline_config(pipeline_yaml(tmp_path, extra, worker_args, ocr), **overrides)


def make_papers(tmp_path: Path, ids=("p1", "p2")) -> list[Paper]:
    folder = tmp_path / "pdfs"
    folder.mkdir(exist_ok=True)
    papers = []
    for paper_id in ids:
        pdf = folder / f"{paper_id}.pdf"
        pdf.write_bytes(b"%PDF-1.4 " + paper_id.encode())
        papers.append(Paper(paper_id, pdf))
    return papers


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def _write(path: Path, items) -> None:
    path.write_text("".join(json.dumps(item) + "\n" for item in items))


def fake_render(paper, paper_dir, config):
    _write(paper_dir / "pages.jsonl", [
        {"page": n, "width_pt": 612.0, "height_pt": 792.0, "image": f"pages/{n}.png", "dpi": config.dpi, "vector": True}
        for n in (1, 2)
    ])


VOTE_DROPS = ("not_structure", "no_reads", "no_usable_read", "tie", "slot_labels_disagree")


def _no_vote(region_id, reason):
    return {"region_id": region_id, "smiles": None, "inchikey": None, "agree": 0, "stereo_kept": False,
            "confidence": 0.0, "readers": [], "slots": {}, "reason": reason}


def fake_vote(paper, paper_dir, config):
    """Like vote.vote_paper: one record per region (rejected or unvoted ones without SMILES, with a reason), and
    its counts: in = out + every dropped_<reason>, plus cores."""
    reads = [r for r in _records(paper_dir / "reads.jsonl") if r["reader"] in config.readers]
    by_region: dict[str, list[dict]] = {}
    for read in reads:
        by_region.setdefault(read["region_id"], []).append(read)
    voted, counts = [], Counter({"in": 0, "out": 0, "cores": 0, **{f"dropped_{d}": 0 for d in VOTE_DROPS}})
    for region in _records(paper_dir / "regions.jsonl"):
        counts["in"] += 1
        group = [r for r in by_region.get(region["region_id"], []) if r["smiles"] and not r["error"]]
        if not config.fixes.vote:
            group = [r for r in group if r["reader"] == config.primary_reader]
        if not region["is_structure"]:
            voted.append(_no_vote(region["region_id"], "not a structure"))
            counts["dropped_not_structure"] += 1
        elif not by_region.get(region["region_id"]):
            voted.append(_no_vote(region["region_id"], "no reads"))
            counts["dropped_no_reads"] += 1
        elif not group:
            voted.append(_no_vote(region["region_id"], "no usable read"))
            counts["dropped_no_usable_read"] += 1
        else:
            smiles, agree = Counter(r["smiles"] for r in group).most_common(1)[0]
            voted.append({"region_id": region["region_id"], "smiles": smiles, "inchikey": None, "agree": agree,
                          "stereo_kept": True, "confidence": agree / 3,
                          "readers": [r["reader"] for r in group if r["smiles"] == smiles], "slots": {}, "reason": None})
            counts["out"] += 1
    _write(paper_dir / "voted.jsonl", voted)
    return dict(counts)


def molecule(paper_id, label, smiles, source, status, page=1):
    return {"paper_id": paper_id, "label": label, "smiles": smiles, "page": page, "bbox": [1.0, 2.0, 30.0, 40.0],
            "source": source, "table_id": None, "row": None, "status": status, "reasons": [], "confidence": 0.5}


def fake_link(paper, paper_dir, config):
    voted = [v for v in _records(paper_dir / "voted.jsonl") if v["smiles"]]
    items = [molecule(paper.paper_id, f"{i}a", v["smiles"], "drawn", "accepted", page=i) for i, v in enumerate(voted, 1)]
    items.append(molecule(paper.paper_id, "9z", "c1ccccc1", "table", "review"))
    _write(paper_dir / config.output("link"), items)


def fake_merge(paper, paper_dir, config):
    items = _records(paper_dir / config.output("link")) + _records(paper_dir / config.output("join_32b"))
    _write(paper_dir / config.output("merge"), items)


def fake_finalize(paper, paper_dir, config):
    """Like finalize.finalize: the junk check abstains (SMILES None), and its counts: in = out + dropped_duplicate,
    plus accepted, review and abstained."""
    from molscout.pipeline.stages import joiner_output

    items = _records(paper_dir / joiner_output(config))
    if config.fixes.junk:
        items = [{**m, "status": "abstained", "smiles": None} if m["smiles"] == "CCN" else m for m in items]
    _write(paper_dir / config.output("finalize"), items)
    statuses = {status: sum(m["status"] == status for m in items) for status in ("accepted", "review", "abstained")}
    return {"in": len(items), "out": len(items), "dropped_duplicate": 0, **statuses}


FAKE_STEPS = {
    "render": fake_render, "vote": fake_vote, "link": fake_link, "merge": fake_merge, "finalize": fake_finalize,
}


def ran(out: Path, stage: str) -> list[str]:
    """The papers the fake worker ran for a stage, in order, from its log."""
    log = out / "logs" / f"{stage}.log"
    return [line.split()[1] for line in log.read_text().splitlines()] if log.is_file() else []


@contextmanager
def models_endpoint(model_id: str = "local-biominer-instruct"):
    """A local OpenAI-style server on 127.0.0.1 that lists one model at /v1/models; yields its /v1 URL."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (the stdlib's name)
            body = json.dumps({"data": [{"id": model_id}]}).encode()
            self.send_response(200 if self.path == "/v1/models" else 404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


def dead_proxy(monkeypatch) -> None:
    """Point every proxy variable at a port nothing listens on, with no no_proxy, as on a compute node."""
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
