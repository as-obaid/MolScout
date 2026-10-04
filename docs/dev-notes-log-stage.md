# Dev Notes

Current stage, open work, fixed settings and reference numbers for MolScout. Specs live in
[dev.md](dev.md), [Benchmarking.md](Benchmarking.md) and [D1.md](D1.md); verified facts about
the papers and tools live in `Course-Project/Docs/log.md`.

---

## Current Stage

| | |
|:--|:--|
| **Phase** | 1, baseline benchmark of existing tools |
| **Session** | D1 Session 1 (Foundations) done; Session 2 next |
| **Repo** | Scoring, loaders, predictions format and `fetch_data.py` on `benchmark` |
| **Internal data** | `data/internal/`, gitignored: 6 PDFs + 222-row ground-truth CSV (220 unique structures) |
| **Internal split** | Frozen in `data/manifests/internal_split.csv`: dev 1, 16, 19 (126) · test 2, 4, 6 (96) |
| **BioVista** | Labels not downloaded; PDFs not fetched |
| **MolGlyph** | Hugging Face access granted (`muhammad-ob`) |
| **Blocker** | 222 internal molecules need hand-marked boxes before any crop- or detection-level run on Internal |

**Dates**

| Date | Item |
|:-----|:-----|
| Oct 4 | Task 1.1, setup |
| Oct 8 | BioMiner go/no-go on Explorer |
| Oct 11 | Tasks 1.2 and 1.3, baseline benchmark |
| Nov 29 | Feature freeze |

---

## Checklist

**Session 1: Foundations**

- [x] `pyproject.toml`, RDKit 2026.3.2
- [x] Scoring: canonical, stereo-stripped, crop accuracy, whole-PDF P/R/F1 (micro, macro)
- [x] `predictions.csv` reader and validator
- [x] Loaders: molfile, SDF, internal CSV
- [x] `fetch_data.py` + manifests: USPTO, UOB, JPO, CLEF, MolRecBench-Wild
- [x] Split loader; Internal scores reported for dev, test and all
- [x] Unit tests; hand-made predictions file matches hand-computed scores

**Session 2: Structure readers (30 runs)**

- [ ] `run.sbatch` + harness writing `meta.json`
- [ ] MolRecBench-Wild loader: its labels are CARBON molecular graphs, not SMILES
- [ ] MolScribe on USPTO within 82.1–93.8%
- [ ] MolNexTR, DECIMER, MolVec, MolGlyph, OCSRGlyph environments
- [ ] 6 readers × 5 datasets scored; Type 1 tables filled

**Session 3: Complete systems and agent**

- [ ] BioVista labels downloaded; DOIs looked up; fetchable PDFs counted
- [ ] BioVista paper list frozen in `data/manifests/` before any scoring
- [ ] DECIMER.ai, OpenChemIE, BioMiner on BioVista and Internal
- [ ] Agent goal prompt + converter to `predictions.csv`; 3 agent runs
- [ ] Type 2 and Type 3 tables filled

**Manual**

- [ ] Explorer: SSH, CUDA environment, storage
- [ ] Hand-mark boxes for the 222 internal molecules
- [x] Freeze the internal split
- [ ] BioMiner go/no-go by Oct 8, or declare the fallback

**MolScout pipeline** (design 2.1 Nov 8; module 2.2 and tests 2.3 Nov 29)

- [ ] Render pages and parse layout (PyMuPDF, Docling)
- [ ] Detect molecule regions
- [ ] Recognize each region; RDKit validity and chemistry checks
- [ ] Enumerate R-groups: parse the table, `molzip` with isotope labels
- [ ] Link each structure to its compound label (vector text or OCR)
- [ ] Write CSV with provenance: SMILES, label, page, bbox, tool, confidence
- [ ] Unit tests per component; keep `AGENTIC.md` current

**Group and final**

- [ ] Sponsor questions from our results; aggregate the cohort's benchmarks (1.4, 1.5; Nov 1)
- [ ] Group pipeline end to end with full error analysis (2.4; Dec 13)
- [ ] Presentation (3.2) and individual reflection (3.3) (Dec 18)
- [ ] Confirm the date for 3.1, which isn't on the schedule

**Open decisions**

| Decision | Options | Note |
|:---------|:--------|:-----|
| BioVista coverage | Score only fetched PDFs | Report as "n of 500 papers"; do not compare directly to BioMiner's published 500-paper score |
| MolScout detector | MolDetv2 · DECIMER-Segmentation | MolDetv2 is CC-BY-NC-SA-4.0; shipping it limits MolScout's license |

---

## Scoring Settings

Fixed before the first run; apply to every tool.

| Setting | Value |
|:--------|:------|
| Canonicalization | RDKit 2026.3.2, default `MolToSmiles`, on prediction and ground truth |
| Match | Exact canonical SMILES, scored stereo-aware and stereo-stripped |
| Scoring unit | One unique canonical SMILES per paper; duplicates collapse to one |
| Invalid output | Unparsable SMILES counts as emitted and wrong; logged for valid-output rate |
| Paper counting | TP: in PDF and output · FP: output only · FN: drawn, not output |
| Crop counting | Accuracy = correct ÷ crops |
| Near-miss audit | Tanimoto, Morgan radius 2, 2,048 bits, wrong answers only; never a headline number |
| Detection match | IoU ≥ 0.5 with the hand-marked box; *cut*: overlap, IoU < 0.5; *merged*: one box over ≥ 2 molecules |
| Reader agreement | Full InChIKey match |
| Page images | Rendered once at 3×, reused by every page-level run |
| Recorded per run | Tool version, checkpoint hash, git commit, hardware, wall-clock time |

## Metrics

| Metric | Definition |
|:-------|:-----------|
| Precision | TP ÷ (TP + FP) |
| Recall | TP ÷ (TP + FN) |
| F1 | 2PR ÷ (P + R) |
| Micro | TP, FP, FN summed over papers, then divided |
| Macro | Unweighted mean of per-paper scores |
| Stereo gap | Stereo-stripped − stereo-aware accuracy, in points |
| Valid-output rate | RDKit-parsable outputs ÷ outputs |
| Label accuracy | Correct molecules with correct `compound_id` ÷ correct molecules |
| Markush | Scaffold match; R-group recall; strict joint match |
| 95% CI | Wilson interval; bootstrap over papers (10,000 resamples) for macro |
| Coverage | Answered ÷ total (for abstaining rules) |
| Selective accuracy | Correct ÷ answered |
| AURC | Area under risk–coverage curve; lower is better |
| Error Jaccard | Crops both readers miss ÷ crops either misses |
| Time | Wall-clock s per paper, page or crop, with hardware |

---

## Tool Settings

| Tool | Role | Settings | Note |
|:-----|:-----|:---------|:-----|
| MolScribe | Reader | Released checkpoint, default inference | Hash recorded |
| MolNexTR | Reader | Released checkpoint, default inference | Hash recorded |
| DECIMER | Reader | Released Transformer, default inference | Package version recorded |
| MolVec | Reader | Default | Java, CPU |
| MolGlyph | Reader | `molglyph_large.pt`, run via BioMiner `interface_molglyph.py` | Gated on Hugging Face; MolScribe encoder/decoder code |
| OCSRGlyph | Reader | `uv sync --extra ocsr`; weights `EdisonScientific/OCSRGlyph` | Apache-2.0; downloads on first use |
| BioMiner | Complete system | Issue #5 environment; `--gres=gpu:h200:4`; 4-way tensor parallel; default config | Only structure output scored |
| DECIMER.ai | Complete system | Default | DECIMER-Segmentation + DECIMER + classifier |
| OpenChemIE | Complete system | Default | MolDet + MolScribe |
| Docling (default) | Parse | Library defaults | MolScout candidate |
| Docling (tuned) | Parse | Heron layout model; picture threshold 0.4; table-cell crop pass; images < 5 KB dropped; pages at 3× | MolScout candidate |
| DECIMER-Segmentation | Detect | Released model; mask expansion on; whole pages at 3× | MolScout candidate |
| MolDetv2 | Detect | Released weights; default threshold | CC-BY-NC-SA-4.0: benchmark only, not shippable |
| DECIMER Image Classifier | Gate | Released model; default threshold | Confirm release before use |

---

## Reference Numbers

Published results, for sanity checks. Not our runs.

| Tool | Data | Score | Source |
|:-----|:-----|:------|:-------|
| MolScribe, MolNexTR | Patent crops (USPTO, JPO) | 82.1–93.8% | Qian et al., 2023; Chen et al., 2024 |
| MolScribe | USPTO, all stereo | 88.4% | Glyph README |
| OCSRGlyph | USPTO 5,719, canonical / chirality-kept / graph | 93.8 / 93.9 / 96.2% | Glyph README |
| MolScribe | MolRecBench-Wild | 41.05% | Yang et al., 2026 |
| MolNexTR | MolRecBench-Wild | 40.90% | Yang et al., 2026 |
| MolGlyph | BioVista OCSR, all / full / chiral / Markush | 0.764 / 0.758 / 0.504 / 0.770 | MolGlyph model card |
| MolScribe | BioVista OCSR, all / full | 0.455 / 0.703 | MolGlyph model card |
| BioMiner | BioVista structures | F1 0.528 | Yan et al., 2026 |
| MolScribe | Internal, whole figures | < 1% | Gulluoglu et al., 2026 |
| DECIMER | Internal, whole figures | 0%; 41% of outputs fail RDKit | Gulluoglu et al., 2026 |
| Docling | Gulluoglu benchmark, figures found | 150 default / 329 tuned | Gulluoglu et al., 2026 |
| Claude Code | Internal, best agentic run | ~21% P / ~11% R | Gulluoglu et al., 2026, Table 3 (corrected) |
| Reader agreement | Correct vs wrong readings | AUROC 0.916 | Guan et al., 2026 |

---

## MolScout Evaluation Plan

For Phase 2. Each stage compares candidates on identical input; the winner goes forward.

| Stage | Candidates | Pick |
|:------|:-----------|:-----|
| Parse | Docling default vs tuned | Most of 222 molecules inside a figure; most labels read |
| Detect | DECIMER-Segmentation vs MolDetv2; Docling figures as baseline | Highest recall; false boxes are cheaper than misses |
| Gate | DECIMER Image Classifier vs none | Keep only if false boxes removed outweigh molecules lost |
| Recognize | Readers from Session 2 | 3 readers with good accuracy and lowest error overlap |
| Verify | Best single reader · 2 of 3 agree · 3 of 3 agree (InChIKey) | By precision vs coverage; precision preferred |

| Analysis | What it shows |
|:---------|:--------------|
| Ladder | Gain from adding each component in build order |
| Leave-one-out | Value of each component once the rest is present |
| Funnel | Share of 222 molecules surviving each stage; recall = product of stage survivals |
| Oracle detection | Hand crops in place of the detector; cost of detection = oracle recall − pipeline recall |
| End-to-end check | Real full run must equal the leave-one-out baseline; a mismatch is a joining bug |

**Failure classes for error tables:** Markush / R-group, stereochemistry, multi-panel,
low resolution (MolScout.md), plus reaction schemes, tautomerism, macrocycles and binding
modes. Classes overlap.

---

## References

- Chen, Y., et al. (2024). MolNexTR. *J. Cheminformatics, 16*, 141. https://doi.org/10.1186/s13321-024-00926-w
- Guan, Y., et al. (2026). *VERDICT: Agreement beats pixel-space verification in real-document OCSR*. arXiv. https://doi.org/10.48550/arXiv.2608.22183
- Gulluoglu, H. S. A., et al. (2026). Practical use of advanced AI frameworks on real-life scientific problems. *bioRxiv*. https://doi.org/10.64898/2026.06.23.734132
- Qian, Y., et al. (2023). MolScribe. *JCIM, 63*(7), 1925–1934. https://doi.org/10.1021/acs.jcim.2c01480
- Yan, J., et al. (2026). *BioMiner*. arXiv. https://doi.org/10.48550/arXiv.2604.21508
- Yang, H., et al. (2026). *MolRecBench-Wild*. arXiv. https://doi.org/10.48550/arXiv.2605.05832
- Glyph (OCSRGlyph, MarkushGlyph): https://github.com/EdisonScientific/glyph
- MolGlyph code: https://github.com/jiaxianyan/BioMiner/blob/main/BioMiner/MolScribe/molscribe/interface_molglyph.py
