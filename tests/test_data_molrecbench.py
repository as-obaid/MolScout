"""MolRecBench-Wild: crop IDs, references from CARBON labels, image export and scoring."""

import json
from io import BytesIO

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from molscout.cli import main
from molscout.data import molrecbench

INTS = pa.list_(pa.int64())
POINTS = pa.list_(pa.list_(pa.float64()))
SCHEMA = pa.schema(
    [
        ("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ("id", pa.string()),
        ("evaluation_subset", pa.string()),
        ("symbols", pa.list_(pa.string())),
        ("charges", INTS),
        ("radicals", INTS),
        ("valences", INTS),
        ("isotopes", INTS),
        ("attach_points", INTS),
        ("coords", POINTS),
        ("bonds", pa.list_(INTS)),
        ("brackets", pa.list_(pa.struct([("alias", pa.string()), ("atoms", INTS), ("display_rects", POINTS)]))),
    ]
)
HEADER = "dataset,item_id,smiles,page,bbox,confidence,tool,seconds"


def jpeg(mode="RGB"):
    buffer = BytesIO()
    Image.new(mode, (8, 6), 0).save(buffer, format="JPEG")
    return buffer.getvalue()


def row(sample_id, symbols, bonds, image=None):
    n = len(symbols)
    empty = [None] * n
    return {
        "image": {"bytes": image or jpeg(), "path": sample_id},
        "id": sample_id,
        "evaluation_subset": "A",
        "symbols": symbols,
        "charges": empty,
        "radicals": empty,
        "valences": empty,
        "isotopes": empty,
        "attach_points": empty,
        "coords": [[float(i), 0.0] for i in range(n)],
        "bonds": bonds,
        "brackets": [],
    }


def write_shard(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "molrecbench_wild"
    write_shard(
        root / "data" / "test-00000-of-00002.parquet",
        [
            row("p1_mol_0.jpg", ["C", "C", "O"], [[0, 1, 1], [1, 2, 1]]),
            row("p1_mol_1.jpg", ["C", "[Rα]"], [[0, 1, 1]], image=jpeg("CMYK")),
        ],
    )
    write_shard(root / "data" / "test-00001-of-00002.parquet", [row("p2_mol_0.jpg", ["O", "[Me]"], [[0, 1, 1]])])
    return root


def test_crop_id_drops_the_image_extension():
    assert molrecbench.crop_id("10.1002_anie.202400632_1_figure_0_mol_0.jpg") == "10.1002_anie.202400632_1_figure_0_mol_0"
    assert molrecbench.crop_id("a.PNG") == "a"
    assert molrecbench.crop_id("a.b") == "a.b"


def test_references_come_from_the_labels(root):
    references = molrecbench.load_references(root)
    assert references["p1_mol_0"].smiles == "CCO"
    assert references["p2_mol_0"].smiles == "CO"
    assert references["p1_mol_1"].smiles is None
    assert "Greek" in references["p1_mol_1"].error


def test_export_writes_one_png_per_crop_once(root):
    assert molrecbench.export_images(root) == 3
    images = root / "images"
    assert sorted(path.name for path in images.iterdir()) == ["p1_mol_0.png", "p1_mol_1.png", "p2_mol_0.png"]
    for path in images.iterdir():
        with Image.open(path) as image:
            assert (image.format, image.mode, image.size) == ("PNG", "RGB", (8, 6))
    assert molrecbench.export_images(root) == 0


def test_duplicate_crop_ids_are_refused(root):
    write_shard(root / "data" / "test-00002-of-00002.parquet", [row("p2_mol_0.jpg", ["C"], [])])
    with pytest.raises(ValueError, match="p2_mol_0"):
        molrecbench.read_labels(root)


def test_checksum_changes_with_a_shard(root):
    before = molrecbench.reference_set_sha256(root)
    assert molrecbench.reference_set_sha256(root) == before
    write_shard(root / "data" / "test-00001-of-00002.parquet", [row("p2_mol_0.jpg", ["N", "[Me]"], [[0, 1, 1]])])
    assert molrecbench.reference_set_sha256(root) != before


def test_no_shards_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="parquet"):
        molrecbench.shard_paths(tmp_path)


def test_molscout_score_reads_the_labels(root, tmp_path):
    predictions = tmp_path / "predictions.csv"
    rows = [("p1_mol_0", "OCC"), ("p1_mol_1", "C"), ("p2_mol_0", "CC")]
    predictions.write_text(HEADER + "\n" + "".join(f"molrecbench_wild,{i},{s},,,,T 1,0.1\n" for i, s in rows))
    out = tmp_path / "scores.json"
    assert main(["score", str(predictions), "--references", str(root), "-o", str(out)]) == 0
    report = json.loads(out.read_text())
    scores, references = report["scores"], report["inputs"]["references"]
    assert scores["items_scored"] == 2
    assert scores["accuracy"]["successes"] == 1
    assert scores["items_excluded"] == ["p1_mol_1"]
    assert references["labels"] == 3 and "files" not in references
    assert references["sha256"] == molrecbench.reference_set_sha256(root)
    assert "Greek" in references["unreadable"]["p1_mol_1"]
