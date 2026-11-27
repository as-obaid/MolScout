"""The drawing workers in their real Explorer environments, on CPU, against real models.

The other worker tests stand in fakes for the models (segment's FakeModels, the readers' load_reader).
These run the workers as the runner does, in their own environments under $MOLSCOUT_STORE/envs, so they
check the loading code too. They are opt-in: set MOLSCOUT_REAL_ENVS=1 and MOLSCOUT_STORE, on a node
that has the store (a CPU job is enough).
"""

import json
import os
import subprocess
from pathlib import Path

import pytest
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw

STORE = os.environ.get("MOLSCOUT_STORE", "")
WORKERS = Path(__file__).resolve().parents[2] / "src" / "molscout" / "pipeline" / "workers"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
pytestmark = pytest.mark.skipif(os.environ.get("MOLSCOUT_REAL_ENVS") != "1" or not STORE,
                                reason="opt-in: set MOLSCOUT_REAL_ENVS=1 and MOLSCOUT_STORE")

READERS = {
    "molscribe": ["--checkpoint", f"{STORE}/models/molscribe/swin_base_char_aux_1m680k.pth"],
    "molnextr": ["--checkpoint", f"{STORE}/models/molnextr/molnextr_best.pth", "--pad-to-square"],
    "molglyph": ["--checkpoint", f"{STORE}/models/molglyph/molglyph_large.pt",
                 "--abbrevs", f"{STORE}/src/BioMiner/BioMiner/commons/abbrevs.csv"],
}


def inchikey(smiles):
    mol = Chem.MolFromSmiles(smiles or "")
    return Chem.MolToInchiKey(mol) if mol is not None else None


def run_worker(env, worker, paper_dir, *args):
    manifest = paper_dir.parent / "work.jsonl"
    manifest.write_text(json.dumps({"paper_id": paper_dir.name, "pdf": str(paper_dir / "paper.pdf"),
                                    "paper_dir": str(paper_dir)}) + "\n")
    command = [f"{STORE}/envs/{env}/bin/python", str(WORKERS / worker), "--manifest", str(manifest),
               "--log", str(paper_dir.parent / f"{env}.log"), *args, "--device", "cpu"]
    return subprocess.run(command, capture_output=True, text=True, timeout=1200, check=False)


@pytest.mark.parametrize("reader", READERS)
def test_a_reader_reads_a_drawn_molecule_in_its_environment(tmp_path, reader):
    paper_dir = tmp_path / "paper"
    (paper_dir / "regions").mkdir(parents=True)
    Draw.MolToImage(Chem.MolFromSmiles(ASPIRIN), size=(400, 400)).save(paper_dir / "regions" / "p1-r1.png")
    region = {"region_id": "p1-r1", "page": 1, "bbox": [0, 0, 96, 96], "crop": "regions/p1-r1.png",
              "structure_score": 0.0, "is_structure": True}
    (paper_dir / "regions.jsonl").write_text(json.dumps(region) + "\n")

    result = run_worker(reader, f"read_{reader}.py", paper_dir, *READERS[reader])

    assert result.returncode == 0, result.stderr[-3000:]
    (read,) = [json.loads(line) for line in (paper_dir / "reads.jsonl").read_text().splitlines()]
    assert read["reader"] == reader and read["error"] is None
    assert inchikey(read["smiles"]) == inchikey(ASPIRIN) and read["slots"] == {}


def test_segment_finds_a_drawn_molecule_on_a_page_in_its_environment(tmp_path):
    paper_dir = tmp_path / "paper"
    (paper_dir / "pages").mkdir(parents=True)
    page = Image.new("RGB", (2550, 3300), "white")  # a Letter page at 300 dpi
    page.paste(Draw.MolToImage(Chem.MolFromSmiles(ASPIRIN), size=(700, 700)), (900, 1200))
    page.save(paper_dir / "pages" / "p001.png")
    (paper_dir / "pages.jsonl").write_text(json.dumps({"page": 1, "width_pt": 612.0, "height_pt": 792.0,
                                                       "image": "pages/p001.png", "dpi": 300, "vector": True}) + "\n")

    result = run_worker("decimer_ai", "segment.py", paper_dir)

    assert result.returncode == 0, result.stderr[-3000:]
    regions = [json.loads(line) for line in (paper_dir / "regions.jsonl").read_text().splitlines()]
    found = [r for r in regions if r["is_structure"]]
    assert found, regions
    x0, y0, x1, y1 = found[0]["bbox"]  # the drawing spans 216-384 pt across and 288-456 pt down
    assert x0 < 384 and x1 > 216 and y0 < 456 and y1 > 288
    assert (paper_dir / found[0]["crop"]).is_file()
