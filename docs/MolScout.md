# MolScout

Chemical structures in research papers are published as drawings. They can be read by a
chemist but not searched, compared or computed on, so the chemistry in the literature stays
locked in figures. Optical chemical structure recognition (OCSR) tools convert these drawings
to machine-readable SMILES. Most are measured on clean, pre-cropped images, and accuracy drops
sharply on real papers: MolScribe and MolNexTR score 82–94% on patent crops but about 41% on
real journal crops from MolRecBench-Wild. Full-document systems and AI agents fare no better.
BioMiner reaches an F1 of 52.8% on BioVista, and the best autonomous Claude Code run in
Gulluoglu et al. (2026) reached about 21% precision and 11% recall.

MolScout takes a whole scientific PDF and returns canonical SMILES for every molecule drawn in
it, with page-level provenance for each one. Every result is measured against existing tools
on fixed datasets, so the gain over current methods is shown, not claimed.

## At a Glance

| | |
|:--------------|:------------------------------------------------------------------------|
| Task | Optical chemical structure recognition from whole papers |
| Input | Scientific paper PDF |
| Output | Canonical SMILES per molecule, with the page it came from |
| Scoring | Exact match of canonical SMILES under RDKit 2026.3.2 |
| Baselines | 4 structure readers, 3 complete systems, 1 AI agent ([Benchmarking](Benchmarking.md)) |
| Datasets | [USPTO](https://github.com/Kohulan/OCSR_Review), [UOB](https://github.com/Kohulan/OCSR_Review), [JPO](https://github.com/Kohulan/OCSR_Review), [CLEF](https://github.com/Kohulan/OCSR_Review), [MolRecBench-Wild](https://huggingface.co/datasets/opendatalab/MolRecBench-Wild), [BioVista](https://github.com/jiaxianyan/BioMiner#statistics-and-access-of-biovista), Internal ([Data](#data)) |
| Models | [MolScribe](https://github.com/thomas0809/MolScribe), [MolNexTR](https://github.com/CYF2000127/MolNexTR), [DECIMER](https://github.com/Kohulan/DECIMER-Image_Transformer), [DECIMER-Segmentation](https://github.com/Kohulan/DECIMER-Image-Segmentation), [MolVec](https://github.com/ncats/molvec), [MolDetv2](https://huggingface.co/UniParser/MolDetv2), [MolGlyph](https://huggingface.co/jiaxianustc/MolGlyph) |
| Language | Python 3.11+ |
| Status | Active development |
| Author | Muhammad Obaidullah |
| Affiliation | XN Sponsored Project, Applied Computer Vision, Northeastern University |
| Sponsors | Dr. Jianing Li, Purdue University<br>Dr. Anton Sinitskiy |

---

## Design Goals

- **Runs from a clone.** One command takes a folder of PDFs to a SMILES file.
- **Works on real papers.** Input is the whole messy PDF, not pre-cropped molecule images.
- **Measured against baselines.** Exact-match canonical SMILES, compared with existing tools on
  a paper split frozen before development.
- **Explains its failures.** Errors are broken out by failure class, not hidden in one score.
- **Readable by non-experts.** Results and limits are stated in plain terms.

## Scope

**In scope**

- Detecting chemical structure drawings on PDF pages
- Converting each detected structure to canonical SMILES
- Recording the page each molecule came from
- Scoring against ground-truth SMILES

**Out of scope**

- Training OCSR models from scratch
- Reaction schemes and reaction conditions
- Redistributing copyrighted paper PDFs

## Approach

## Data

| Dataset | Input | Size | Access |
|:--------|:------|:-----|:-------|
| [USPTO](https://github.com/Kohulan/OCSR_Review) | Patent crops | 5,719 images | Public |
| [UOB](https://github.com/Kohulan/OCSR_Review) | Patent and synthetic crops | 5,740 images | Public |
| [JPO](https://github.com/Kohulan/OCSR_Review) | Patent crops | 450 images | Public |
| [CLEF](https://github.com/Kohulan/OCSR_Review) | Patent crops | 961 images | Public |
| [MolRecBench-Wild](https://huggingface.co/datasets/opendatalab/MolRecBench-Wild) | Real journal crops | 5,029 structures, 820 papers | Public |
| [BioVista](https://github.com/jiaxianyan/BioMiner#statistics-and-access-of-biovista) | Whole PDFs | 8,735 structures, 500 papers | Public; PDFs fetched by DOI |
| Internal | Whole PDFs | 222 molecules, 6 papers | Private |

Paper PDFs are copyrighted and are not committed to this repository. Each dataset ships as a
DOI manifest, and PDFs are fetched locally.

## Evaluation

A prediction counts as correct only when its canonical SMILES exactly matches the ground truth
under RDKit 2026.3.2. Whole-PDF results report precision, recall and F1, both pooled over all
molecules and averaged per paper. Baseline setup and results are in
[Benchmarking](Benchmarking.md).

## Results

## Error Analysis

Failures are grouped by cause:

| Failure class | Description | Count |
|:--------------|:------------|------:|
| Markush / R-group | Generic structures with variable substituents | — |
| Stereochemistry | Wedges, hashes and stereo labels read incorrectly | — |
| Multi-panel figures | Several structures in one figure, split or merged wrongly | — |
| Low resolution | Drawings too small or blurred to read | — |

## Limitations

## Roadmap

| Milestone | Date | Status |
|:----------|:-----|:-------|
| Baseline benchmark | — | In progress |
| End-to-end pipeline | — | Planned |
| Error analysis | — | Planned |
| Feature freeze | Nov 29, 2026 | Planned |

## Development Process

MolScout is built with AI-assisted development. Agent sessions, errors and human interventions
are logged, so the build process is part of the evidence alongside the results.

## References

- Gulluoglu, H. S. A., et al. (2026). Practical use of advanced AI frameworks on real-life
  scientific problems: Three case studies. *bioRxiv*.
  [doi:10.64898/2026.06.23.734132](https://doi.org/10.64898/2026.06.23.734132)
- BioMiner and the BioVista benchmark. [arXiv:2604.21508](https://arxiv.org/abs/2604.21508) ·
  [GitHub](https://github.com/jiaxianyan/BioMiner)
- MolRecBench-Wild. [arXiv:2605.05832](https://arxiv.org/abs/2605.05832) ·
  [Hugging Face](https://huggingface.co/datasets/opendatalab/MolRecBench-Wild)

## Acknowledgments

Thanks to Dr. Jianing Li (Purdue University) and Dr. Anton Sinitskiy for sponsoring the
project and providing the internal dataset, and to Navdeep Singh Dhanjal for guidance.
