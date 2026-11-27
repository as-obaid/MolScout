import importlib.util
import re
import struct
import sys
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin
from rdkit import Chem

REPO = Path(__file__).resolve().parents[1]
FIGURES = REPO / "docs" / "figures"
TEXT_CHUNKS = {b"tEXt", b"iTXt", b"zTXt", b"eXIf", b"tIME"}


def _load_script():
    spec = importlib.util.spec_from_file_location("hero", REPO / "scripts" / "hero.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


hero = _load_script()


def _chunks(path: Path) -> list[bytes]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    names, i = [], 8
    while i < len(data):
        length, name = struct.unpack(">I4s", data[i : i + 8])
        names.append(name)
        i += 12 + length
    return names


def test_compounds_are_table_rows_from_two_papers():
    compounds = hero.compounds()
    assert [(c.paper, c.label) for c in compounds] == [
        ("paper_a", "1a"), ("paper_a", "1b"), ("paper_a", "1c"), ("paper_b", "7a"), ("paper_b", "7b"),
    ]
    assert [c.smiles for c in compounds] == [
        "Clc1cccc(Nc2ncnc3ccccc23)c1",
        "Brc1cccc(Nc2ncnc3ccccc23)c1",
        "C#Cc1cccc(Nc2ncnc3ccccc23)c1",
        "Fc1ccc(-c2cn3ccccc3n2)cc1",
        "COc1ccc(-c2cn3ccccc3n2)cc1",
    ]
    for compound in compounds:
        assert compound.smiles == Chem.MolToSmiles(Chem.MolFromSmiles(compound.smiles))
        assert compound.mol.HasSubstructMatch(Chem.MolFromSmiles(compound.series.scaffold))


def test_r_atoms_are_the_substituent_only():
    symbols = [sorted(c.mol.GetAtomWithIdx(i).GetSymbol() for i in hero.r_atoms(c)) for c in hero.compounds()]
    assert symbols == [["Cl"], ["Br"], ["C", "C"], ["F"], ["C", "O"]]


def test_smiles_split_marks_the_r_group():
    assert hero.smiles_parts("Clc1cccc(Nc2ncnc3ccccc23)c1", "Cl[*:1]") == ("Cl", "c1cccc(Nc2ncnc3ccccc23)c1")
    assert hero.smiles_parts("C#Cc1cccc(Nc2ncnc3ccccc23)c1", "C#C[*:1]") == ("C#C", "c1cccc(Nc2ncnc3ccccc23)c1")
    assert hero.smiles_parts("c1ccccc1Cl", "Cl[*:1]") == ("", "c1ccccc1Cl")


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_page_shows_a_folder_of_papers_their_molecules_and_the_csv(theme):
    html = hero.page_html(theme)
    for text in ("Papers", "papers/", "Molecules", "molecules.csv", "IC50", "paper_a", "paper_b", "label",
                 "1a", "1c", "7b", "c1cccc(Nc2ncnc3ccccc23)c1", "c1ccc(-c2cn3ccccc3n2)cc1"):
        assert text in html
    assert html.count('class="page') >= 3  # a stack of PDF pages
    assert html.count("<svg") >= 8  # the core in the front paper, five molecules, two arrows
    assert hero.THEMES[theme].background in html
    assert "<script" not in html


def test_strip_metadata_removes_text_chunks(tmp_path):
    path = tmp_path / "tagged.png"
    info = PngImagePlugin.PngInfo()
    info.add_text("Software", "some renderer")
    Image.new("RGB", (8, 4), "white").save(path, pnginfo=info)
    assert b"tEXt" in _chunks(path)
    hero.strip_metadata(path)
    assert not TEXT_CHUNKS & set(_chunks(path))
    with Image.open(path) as image:
        assert image.size == (8, 4)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_committed_hero_images_are_lean(theme):
    path = FIGURES / f"hero-{theme}.png"
    with Image.open(path) as image:
        assert image.size == (2 * hero.WIDTH, 2 * hero.HEIGHT)
    assert path.stat().st_size < 400_000
    assert not TEXT_CHUNKS & set(_chunks(path))


def test_readme_and_docs_images_exist():
    for doc in (REPO / "README.md", *sorted((REPO / "docs").glob("*.md"))):
        text = doc.read_text(encoding="utf-8")
        refs = re.findall(r'(?:src|srcset)="([^"]+)"', text) + re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
        for ref in refs:
            if not ref.startswith("http"):
                assert (doc.parent / ref).is_file(), f"{doc.name} links a missing image: {ref}"
