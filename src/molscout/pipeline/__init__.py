"""MolScout pipeline: paper PDFs to SMILES.

Stages run in the order render, parse, segment, read (one stage per reader), vote, the joiner (link and build
for rules; the 32B worker for 32b; both and merge for both), finalize and export; see runner.run and
config.PipelineConfig.
"""
