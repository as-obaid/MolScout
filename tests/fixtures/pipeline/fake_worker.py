"""A stand-in pipeline worker with the real workers' command line; standard library only.

    fake_worker.py --manifest work.jsonl --log stage.log --stage segment [--fail P] [--empty P] [--crash] [--overwrite]

It writes plausible records for its stage into each paper folder and logs one `ran <paper>` line per paper.
--fail writes the paper's error to errors.jsonl and no output, --empty an empty output file, --crash exits 1 (a setup failure),
--overwrite makes a reader replace reads.jsonl instead of appending to it, and --exit N exits with N after the
papers (4: some papers failed; 3: the 32B server was lost). ocr_labels also writes .counts/ocr_labels.json, and
--bad-counts P makes P's counts not add up.
"""

import argparse
import json
import sys
from pathlib import Path


def records(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path, items, mode="w"):
    with path.open(mode) as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")


def output(stage, folder, paper_id, args):
    if stage == "parse":
        blocks = [
            {"page": 1, "index": 0, "type": "text", "bbox": [0, 0, 10, 10], "text": "Compound 1a", "html": None,
             "parent": None, "image": None},
            {"page": 1, "index": 1, "type": "table", "bbox": [0, 20, 10, 30], "text": "", "html": "<table></table>",
             "parent": None, "image": None},
        ]
        return "layout.jsonl", blocks
    if stage == "segment":
        pages = records(folder / "pages.jsonl")
        rejected = paper_id in args.reject
        regions = [
            {"region_id": f"p{p['page']}-r1", "page": p["page"], "bbox": [1, 2, 30, 40], "crop": None,
             "structure_score": 0.1 if rejected else 0.9, "is_structure": not rejected}
            for p in pages
        ]
        return "regions.jsonl", regions
    if stage == "ocr_labels":
        regions = [r for r in records(folder / "regions.jsonl") if r["is_structure"]]
        lines = [
            {"page": r["page"], "bbox": [10, 42, 20, 50], "text": str(n), "score": 0.99, "source_block": f"region:{r['region_id']}"}
            for n, r in enumerate(regions, 1)
        ]
        counts = {"in": len(regions), "out": len(regions), "dropped_no_text": 0, "dropped_failed": 0, "lines": len(lines)}
        if paper_id in args.bad_counts:
            counts["out"] += 1
        (folder / ".counts").mkdir(exist_ok=True)
        (folder / ".counts" / "ocr_labels.json").write_text(json.dumps(counts))
        return "ocr.jsonl", lines
    if stage.startswith("read_"):
        reader = stage.removeprefix("read_")
        smiles = "CCN" if reader == "molnextr" else "CCO"
        regions = [r for r in records(folder / "regions.jsonl") if r["is_structure"]]
        reads = [{"region_id": r["region_id"], "reader": reader, "smiles": smiles, "slots": {}, "error": None}
                 for r in regions]
        return "reads.jsonl", reads
    if stage == "join_32b":
        voted = records(folder / "voted.jsonl")
        molecules = [
            {"paper_id": paper_id, "label": f"L{i}", "smiles": v["smiles"], "page": 1, "bbox": [1, 2, 3, 4],
             "source": "32b", "table_id": None, "row": None, "status": "accepted", "reasons": [], "confidence": 0.9}
            for i, v in enumerate(voted, 1)
        ]
        return args.output or "molecules_32b.jsonl", molecules
    raise SystemExit(f"fake worker: unknown stage {stage}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--fail", action="append", default=[])
    parser.add_argument("--empty", action="append", default=[])
    parser.add_argument("--reject", action="append", default=[], help="segment: the gate rejects this paper's regions")
    parser.add_argument("--bad-counts", action="append", default=[], help="ocr_labels: counts that do not add up")
    parser.add_argument("--crash", action="store_true")
    parser.add_argument("--exit", type=int, default=0, help="the status to exit with after the papers")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--url")
    args = parser.parse_args()
    if args.crash:
        print("fake worker: crashing", file=sys.stderr)
        return 1
    with args.log.open("a") as log:
        for line in args.manifest.read_text().splitlines():
            work = json.loads(line)
            paper_id, folder = work["paper_id"], Path(work["paper_dir"])
            log.write(f"ran {paper_id}\n")
            if paper_id in args.fail:
                write(folder / "errors.jsonl", [{"paper_id": paper_id, "stage": args.stage, "error": "fake failure"}], "a")
                continue
            name, items = output(args.stage, folder, paper_id, args)
            if paper_id in args.empty:
                items = []
            reader = args.stage.startswith("read_") and not args.overwrite
            write(folder / name, items, "a" if reader else "w")
    return args.exit


if __name__ == "__main__":
    sys.exit(main())
