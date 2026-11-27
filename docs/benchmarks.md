# Benchmarks

The benchmarks cover two types, both run by one harness and scored by `molscout score`: structure
readers on cropped drawings, and complete PDF-to-SMILES systems on whole papers.

| Type | Input | Datasets | Systems | Metric |
|:--|:--|:--|:--|:--|
| 1. Structure readers | one cropped drawing | USPTO, UOB, JPO, CLEF, MolRecBench-Wild | MolScribe, MolNexTR, DECIMER, MolVec, MolGlyph, OCSRGlyph | exact-match accuracy |
| 2. Complete systems | one paper PDF | BioVista, Internal | MolScout, BioMiner, DECIMER.ai, OpenChemIE | micro precision, recall, F1 and accuracy |

## Datasets

| Dataset | Type | Scored | Source |
|:--|:-:|:--|:--|
| [USPTO](https://github.com/Kohulan/OCSR_Review) | 1 | 5,704 of 5,719 patent crops | Rajan et al. (2020) |
| [UOB](https://github.com/Kohulan/OCSR_Review) | 1 | 5,740 catalogue crops | Rajan et al. (2020) |
| [JPO](https://github.com/Kohulan/OCSR_Review) | 1 | 449 of 450 patent crops | Rajan et al. (2020) |
| [CLEF](https://github.com/Kohulan/OCSR_Review) | 1 | 977 of 992 patent crops | Rajan et al. (2020) |
| [MolRecBench-Wild](https://huggingface.co/datasets/opendatalab/MolRecBench-Wild) | 1 | 2,371 of 5,024 journal crops | Yang et al. (2026) |
| [BioVista](https://github.com/jiaxianyan/BioMiner) | 2 | 145 papers, 2,304 molecules | Yan et al. (2026) |
| Internal | 2 | 6 papers, 220 molecules | private; results hold counts only |

- A crop is scored when RDKit can read its reference. MolRecBench-Wild's graph labels are converted to
  SMILES as in its official SMILES track.
- BioVista ships labels only. Each of the 145 papers has a PDF that can be obtained legally without
  an account, and RDKit can read all of its labels. The papers are listed in
  `data/manifests/biovista_papers.csv`.
- In development paper 221_6oac, 7 labels contradict their own backbone and R groups, because pasting
  the fragments made ring-closure digits collide. These 7 labels are rebuilt from the backbone and R
  groups.

### Splits

| Dataset | Development | Held-out |
|:--|:--|:--|
| Internal | papers 1, 16, 19: 125 molecules | papers 2, 4, 6: 95 molecules |
| BioVista | 20 papers of `data/manifests/biovista_dev.csv`: 463 molecules | the other 125 papers: 1,841 molecules |

MolScout's configuration was chosen on the development papers only, so the held-out papers measure it
on papers it was not tuned on. Every system runs every paper and is scored on each split.

## Systems

| Reader | Version | Checkpoint |
|:--|:--|:--|
| [MolScribe](https://github.com/thomas0809/MolScribe) (Qian et al., 2023) | 1.1.1, git 7296a30 | `swin_base_char_aux_1m680k` |
| [MolNexTR](https://github.com/CYF2000127/MolNexTR) (Chen et al., 2024) | 1.0.2, git 6f6502b | `molnextr_best.pth` |
| [DECIMER](https://github.com/Kohulan/DECIMER-Image_Transformer) (Rajan et al., 2023) | 2.8.0, git d927ed1 | Zenodo 8300489 |
| [MolVec](https://github.com/ncats/molvec) | 0.9.8, git b412dd0 | rule-based, no weights |
| [MolGlyph](https://huggingface.co/jiaxianustc/MolGlyph) (Yan et al., 2026) | 1.1.1, BioMiner git 17c6161 | `molglyph_large` |
| [OCSRGlyph](https://github.com/EdisonScientific/glyph) | 0.1.0, git 0bf782f | `model.pth` |

| System | What it is | Version |
|:--|:--|:--|
| MolScout | this repository ([pipeline.md](pipeline.md)) | 0.1.0 |
| [BioMiner](https://github.com/jiaxianyan/BioMiner) (Yan et al., 2026) | MinerU, MolDetV2, MolGlyph and BioMiner-Instruct 32B | 1.0.0, git 17c6161 |
| [DECIMER.ai](https://github.com/OBrink/DECIMER.ai) (Rajan et al., 2023) | DECIMER Segmentation, classifier and reader | app git 1fc858a |
| [OpenChemIE](https://github.com/CrystalEye42/OpenChemIE) (Fan et al., 2024) | MolDet and MolScribe | 0.1.0, git d9b50bb |

MolScout runs its frozen configuration: the `both` joiner (rules plus BioMiner-Instruct 32B, merged),
every fix on except `legends` and `partial_stereo` ([pipeline.md](pipeline.md#fixes)), and only
`accepted` molecules exported. What each fix adds is in [results.md](results.md#the-ladder).

Every tool runs its released code and weights, pinned in its `tool.yaml`. These settings differ from
the defaults, each with its reason in that file:

- MolNexTR pads crops to a square; its released API stretches them.
- OCSRGlyph runs in fp32, as in its published USPTO evaluation.
- BioMiner serves its 32B model on one GPU at memory fraction 0.8; upstream uses four GPUs at 0.6.
- DECIMER.ai reads every page and segment; the public app stops at 10 pages and 20 segments.
- OpenChemIE runs on torch 2.5.1 with CUDA 12.1; its pinned torch has no H200 kernels.

## Scoring

**Both types**
- A prediction is correct when its canonical SMILES equals the reference's under RDKit 2026.3.2,
  rewritten until stable (at most five passes). The scorer refuses any other RDKit version.
- Stereochemistry counts. Stereo-stripped scores are also recorded.
- `*`, `[1*]` and `[*:1]` are different atoms.

**Type 1, structure readers**
- Accuracy is correct crops over scored crops. A missing, empty or unparsable prediction is wrong.
- The scorer gives 95% Wilson intervals.

**Type 2, complete systems**
- Per paper, predictions and references are reduced to distinct canonical SMILES.
- TP: predicted and in the reference. FP: predicted but not in the reference, counting each distinct
  unparsable output. FN: in the reference but not predicted.
- TP, FP and FN are summed over a split's papers: P = TP / (TP + FP), R = TP / (TP + FN),
  F1 = 2PR / (P + R).
- Accuracy is TP / (TP + FP + FN), the usual (TP + TN) / all with TN = 0, since a set of extracted
  molecules has no true negatives.
- The scorer gives 95% intervals from a bootstrap over papers (10,000 resamples, seed 6630).

## Results

### Structure readers on crops

Exact-match accuracy (%), stereochemistry included: `scores.accuracy` in
`benchmarks/results/<reader>__<dataset>/scores.json`. Bold: best on each dataset.

| Reader | USPTO | UOB | JPO | CLEF | MolRecBench-Wild |
|:--|--:|--:|--:|--:|--:|
| MolScribe | 92.1 | **87.3** | 59.5 | 80.8 | 60.1 |
| MolNexTR | 86.0 | 85.9 | 51.2 | 76.9 | 54.5 |
| DECIMER | 56.9 | 86.4 | 39.4 | 72.2 | 37.8 |
| MolVec | 88.3 | 80.2 | **66.6** | 84.7 | 35.6 |
| MolGlyph | 72.5 | 87.0 | 57.5 | 72.1 | **63.2** |
| OCSRGlyph | **94.0** | 86.8 | 51.9 | **90.7** | 42.7 |

### Complete systems on papers

Micro precision (P), recall (R), F1 and accuracy (Acc) in % over distinct molecules, stereochemistry
included. MolScout is its frozen configuration.

**Development papers** (Internal 1, 16, 19; 20 BioVista papers)

| System | Internal P | Internal R | Internal F1 | Internal Acc | BioVista P | BioVista R | BioVista F1 | BioVista Acc |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| MolScout | 98.2 | 44.0 | 60.8 | 43.7 | 74.1 | 53.8 | 62.3 | 45.3 |
| BioMiner | 81.1 | 34.4 | 48.3 | 31.9 | 59.8 | 56.2 | 57.9 | 40.8 |
| DECIMER.ai | 40.5 | 25.6 | 31.4 | 18.6 | 29.4 | 33.9 | 31.5 | 18.7 |
| OpenChemIE | 28.2 | 23.2 | 25.4 | 14.6 | 19.5 | 28.3 | 23.1 | 13.0 |

**Held-out papers** (Internal 2, 4, 6; the other 125 BioVista papers)

| System | Internal P | Internal R | Internal F1 | Internal Acc | BioVista P | BioVista R | BioVista F1 | BioVista Acc |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| MolScout | 65.2 | 31.6 | 42.6 | 27.0 | 67.2 | 52.4 | 58.9 | 41.7 |
| BioMiner | 63.5 | 34.7 | 44.9 | 28.9 | 63.9 | 59.7 | 61.7 | 44.6 |
| DECIMER.ai | 22.2 | 21.1 | 21.6 | 12.1 | 28.7 | 35.3 | 31.7 | 18.8 |
| OpenChemIE | 16.8 | 20.0 | 18.3 | 10.1 | 22.2 | 38.6 | 28.2 | 16.4 |

Figures are in [results.md](results.md); commands that rebuild every number are in
[reproduce.md](reproduce.md).

## References

- Chen, Y., et al. (2024). MolNexTR: a generalized deep learning model for molecular image
  recognition. *Journal of Cheminformatics*, 16. https://doi.org/10.1186/s13321-024-00926-w
- Fan, V., et al. (2024). OpenChemIE: An information extraction toolkit for chemistry literature.
  arXiv:2404.01462.
- Qian, Y., et al. (2023). MolScribe: Robust molecular structure recognition with image-to-graph
  generation. *Journal of Chemical Information and Modeling*, 63(7).
  https://doi.org/10.1021/acs.jcim.2c01480
- Rajan, K., et al. (2020). A review of optical chemical structure recognition tools. *Journal of
  Cheminformatics*, 12. https://doi.org/10.1186/s13321-020-00465-0
- Rajan, K., et al. (2023). DECIMER.ai: an open platform for automated optical chemical structure
  identification, segmentation and recognition in scientific publications. *Nature
  Communications*, 14. https://doi.org/10.1038/s41467-023-40782-0
- Yan, J., et al. (2026). BioMiner: A multi-modal system for automated mining of protein-ligand
  bioactivity data from literature. arXiv:2604.21508.
- Yang, H., et al. (2026). MolRecBench-Wild: A real-world benchmark for optical chemical structure
  recognition. *CVPR 2026*. arXiv:2605.05832.
