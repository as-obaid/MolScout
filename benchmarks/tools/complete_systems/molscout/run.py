"""MolScout, the pipeline in this repository, on whole PDFs, written as predictions.csv. Runs in the molscout environment.

The harness passes the papers (papers.csv), the dataset, the tool label, predictions.csv's path and a checkpoint
(--resume). This runs molscout.pipeline on all papers stage by stage (each model loads once) with the pipeline
config beside this script and the variant's overrides (--joiner, --fixes-only, --fixes-off, --export), then
writes the files of paper_runner's contract beside predictions.csv:

- predictions.errors.json: the papers the pipeline stopped, each as "<stage>: <message>"; they have no rows.
- predictions.timing.json: each paper's seconds, the sum of its stage times. A stage reused from an earlier run
  or from the shared folder counts the time it took when it ran, so s / paper is the configuration's cost.
- stages.json: per stage and paper, records in and out, seconds, peak GPU memory and errors, and the funnel
  counts. No SMILES and no paper text; for Internal each error is cut to its type.

--work keeps each paper's stage files across runs (default: beside the checkpoint), so a resumed or repeated run
redoes only what changed; --shared keeps render, parse, segment and the readers for every variant with the same
upstream settings. --start-vllm starts BioMiner-Instruct on vLLM (vllm_server.py) when the 32B joiner first has
work and stops it at the end, staging into a work folder of the job's own under $TMPDIR. The run exits 1 when
every paper stopped, as paper_runner does.
"""

import argparse
import contextlib
import json
import os
import re
import shutil
import signal
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vllm_server import PORT, START_SECONDS, VllmServer, command  # noqa: E402

ERROR_TYPE = re.compile(r"[A-Za-z_][\w.]*(Error|Exception|Exit|Warning|Interrupt)")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--papers", type=Path, required=True, help="CSV with columns paper_id,pdf")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tool", required=True, help='"<name> <version>"')
    parser.add_argument("--output", type=Path, required=True, help="predictions.csv to write")
    parser.add_argument("--resume", type=Path, help="the harness's checkpoint CSV; the work folder goes beside it")
    parser.add_argument("--pipeline-config", type=Path, default=HERE / "pipeline.yaml")
    parser.add_argument("--joiner", choices=("rules", "32b", "both"))
    parser.add_argument("--fixes-only", help="comma-separated fixes to keep on, or none")
    parser.add_argument("--fixes-off", default="", help="comma-separated fixes to switch off")
    parser.add_argument("--export", choices=("accepted", "accepted+review"))
    parser.add_argument("--vllm-url", help="a running vLLM server's OpenAI endpoint")
    parser.add_argument("--work", type=Path, help="the pipeline's work folder (default: beside --resume)")
    parser.add_argument("--shared", type=Path, help="the folder of upstream stage outputs shared by variants")
    parser.add_argument("--device", choices=("gpu", "cpu"), help="default: the pipeline config's (gpu)")
    parser.add_argument("--start-vllm", action="store_true", help="start BioMiner-Instruct when the 32B joiner runs")
    parser.add_argument("--vllm-port", type=int, default=PORT)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--vllm-start-seconds", type=int, default=START_SECONDS)
    return parser


def names(text):
    if text is None:
        return None
    return () if text.strip() in ("", "none") else tuple(part.strip() for part in text.split(",") if part.strip())


def main(argv=None, steps=None):
    """Run the pipeline for the harness; `steps` replaces in-process steps (tests)."""
    from molscout.pipeline.config import WORKERS_DIR, load_pipeline_config
    from molscout.pipeline.inputs import resolve_papers
    from molscout.pipeline.runner import PipelineError, run

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.work is None and args.resume is None:
        parser.error("pass --work, or --resume for a work folder beside the checkpoint")
    signal.signal(signal.SIGTERM, _exit_143)
    work = args.work or args.resume.parent / "work"
    vllm_url = f"http://127.0.0.1:{args.vllm_port}/v1" if args.start_vllm else args.vllm_url
    try:
        config = load_pipeline_config(
            args.pipeline_config, joiner=args.joiner, vllm_url=vllm_url, fixes_only=names(args.fixes_only),
            fixes_off=names(args.fixes_off) or (), export=args.export, shared=args.shared, device=args.device,
        )
        papers = resolve_papers(args.papers)
    except ValueError as exc:
        parser.error(str(exc))
    with contextlib.ExitStack() as stack:
        def before_stage(stage):
            if stage == "join_32b" and args.start_vllm:
                staging = Path(os.environ.get("TMPDIR", "/tmp")) / f"molscout-32b-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
                argv = command(WORKERS_DIR / "serve_32b.sh", staging, args.vllm_port, args.gpu_memory_utilization)
                server = VllmServer(argv, work / "logs" / "vllm.log", port=args.vllm_port,
                                    start_seconds=args.vllm_start_seconds)
                stack.enter_context(server)

        try:
            summary = run(papers, work, config, steps=steps, dataset=args.dataset, tool=args.tool,
                          before_stage=before_stage)
        except PipelineError as exc:
            print(f"molscout run.py: error: {exc}", file=sys.stderr)
            return 2
    write_outputs(summary, work, args.output, args.dataset)
    stopped = [paper for paper, record in summary.papers.items() if record["status"] == "stopped"]
    return 1 if papers and len(stopped) == len(papers) else 0


def write_outputs(summary, work, output, dataset):
    """predictions.csv, its errors and timing files, and stages.json beside it."""
    shutil.copyfile(work / "predictions.csv", output)
    stopped = {
        paper: f"{record['stopped_at']}: {record['errors'][-1]['error']}"
        for paper, record in summary.papers.items()
        if record["status"] == "stopped"
    }
    errors = {"papers": len(summary.papers), "failed": len(stopped), "errors": stopped}
    _write(output.with_name(output.stem + ".errors.json"), errors)
    timing = {"papers": len(summary.papers), "seconds": summary.seconds()}
    _write(output.with_name(output.stem + ".timing.json"), timing)
    stages = json.loads((work / "stages.json").read_text(encoding="utf-8"))
    _write(output.with_name("stages.json"), public_stages(stages) if dataset == "internal" else stages)


def public_stages(stages):
    """stages.json with each error message cut to its exception type ("error" when it has none): Internal papers are
    private, and a message may quote their text."""
    def cut(message):
        kind = str(message).split(":", 1)[0].strip()
        return kind if ERROR_TYPE.fullmatch(kind) else "error"

    papers = {}
    for paper, record in stages["per_paper"].items():
        papers[paper] = {
            **record,
            "errors": [{**error, "error": cut(error["error"])} for error in record["errors"]],
            "stages": {
                stage: {**counts, "errors": [cut(message) for message in counts["errors"]]}
                for stage, counts in record["stages"].items()
            },
        }
    return {**stages, "per_paper": papers}


def _write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _exit_143(signum, frame):
    """SIGTERM (scancel, a time limit) unwinds through finally clauses: workers and vLLM are stopped."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    sys.exit(143)


if __name__ == "__main__":
    sys.exit(main())
