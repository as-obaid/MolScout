"""The stages that run inside the molscout process: render, vote, link and build, merge, finalize and export.

Each step reads its inputs from the paper folder and writes its output file there, and may return counts
({"in", "out", "dropped", "dropped_<why>", ...}) that the runner records and checks. The chemistry and document
modules are imported when a step runs, so the runner and its tests do not need them. What each step calls:

- render: `render.render_paper(pdf, paper_dir, dpi=...)` writes pages.jsonl and the page images.
- vote: `vote.vote_paper(paper_dir, fixes=..., readers=..., primary_reader=...)` writes voted.jsonl, one record
  per region (a region the gate rejected or that got no vote has no SMILES and a reason), and returns its counts.
- link: `link.link_paper(paper_dir, opsin, fixes, pdf=...)` gives the rules molecules and their counts; drawings
  take their labels from layout.jsonl and, when the ocr_labels worker ran (after segment, in the mineru
  environment), from ocr.jsonl; the PDF's text places the rows of tables with drawn cells.
- merge (joiner `both`): `merge.merge(rules, molecules_32b)` gives the merged molecules, or the molecules and
  their counts.
- finalize (every joiner): `finalize.finalize(paper_dir, molecules, fixes=...)` applies relevance, the junk and
  formula checks and dedupe, and returns the molecules and their counts; they go to molecules.jsonl.
- export: the paper's molecules.csv (export.write_molecules_csv).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import cache
from pathlib import Path
from typing import Any

from molscout.pipeline import export
from molscout.pipeline.config import PipelineConfig
from molscout.pipeline.inputs import Paper
from molscout.pipeline.stages import joiner_output

Counts = Mapping[str, int]
Step = Callable[[Paper, Path, PipelineConfig], Counts | None]


def render(paper: Paper, paper_dir: Path, config: PipelineConfig) -> None:
    from molscout.pipeline import render as rendering

    rendering.render_paper(paper.pdf, paper_dir, dpi=config.dpi)


def vote(paper: Paper, paper_dir: Path, config: PipelineConfig) -> Counts:
    from molscout.pipeline import vote as voting

    return voting.vote_paper(
        paper_dir, fixes=config.fixes, readers=config.readers, primary_reader=config.primary_reader
    )


def link(paper: Paper, paper_dir: Path, config: PipelineConfig) -> Counts:
    jar = config.opsin_jar_file()
    if jar is None or not jar.is_file():
        raise FileNotFoundError(f"no OPSIN jar at {config.opsin_jar}; run scripts/fetch_opsin.py --dest <folder>")
    from molscout.pipeline import link as linking
    from molscout.pipeline import schema

    molecules, counts = linking.link_paper(paper_dir, _opsin(jar, config.java), config.fixes, pdf=paper.pdf)
    schema.write_jsonl(paper_dir / config.output("link"), molecules)
    return counts


def merge(paper: Paper, paper_dir: Path, config: PipelineConfig) -> Counts | None:
    from molscout.pipeline import merge as merging
    from molscout.pipeline import schema

    rules = schema.read_jsonl(paper_dir / config.output("link"), schema.Molecule)
    llm = schema.read_jsonl(paper_dir / config.output("join_32b"), schema.Molecule)
    molecules, counts = _with_counts(merging.merge(rules, llm))
    schema.write_jsonl(paper_dir / config.output("merge"), molecules)
    return counts


def finalize(paper: Paper, paper_dir: Path, config: PipelineConfig) -> Counts | None:
    from molscout.pipeline import finalize as finalizing
    from molscout.pipeline import schema

    molecules = schema.read_jsonl(paper_dir / joiner_output(config), schema.Molecule)
    molecules, counts = _with_counts(finalizing.finalize(paper_dir, molecules, fixes=config.fixes))
    schema.write_jsonl(paper_dir / config.output("finalize"), molecules)
    return counts


def export_paper(paper: Paper, paper_dir: Path, config: PipelineConfig) -> None:
    export.write_molecules_csv(paper_dir / config.output("export"), [paper_dir / config.output("finalize")])


STEPS: Mapping[str, Step] = {
    "render": render,
    "vote": vote,
    "link": link,
    "merge": merge,
    "finalize": finalize,
    "export": export_paper,
}


def _with_counts(result: Any) -> tuple[list[Any], Counts | None]:
    """(molecules, counts) from a function that returns either the molecules or both."""
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Mapping):
        return list(result[0]), result[1]
    return list(result), None


@cache
def _opsin(jar: Path, java: str) -> Any:
    from molscout.pipeline.opsin import Opsin

    return Opsin(jar, java=java)
