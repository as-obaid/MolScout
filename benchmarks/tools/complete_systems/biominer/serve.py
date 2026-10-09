"""Start one of BioMiner's LitServe servers (MinerU, MolDetV2 or MolGlyph) on this job's GPUs.

Started by run.py in the biominer environment, from the work folder that holds a copy of the pinned
commit, with that folder and its scripts/ on PYTHONPATH (BioMiner's README runs the servers from the
repository root with PYTHONPATH=./). BioMiner's scripts/{mineru,moldet,ocsr}_server.py define each
server's LitAPI class and, under __main__, serve it on GPUs [0, 1, 2, 3] with one worker per GPU on
the port its client calls. This imports those classes unchanged and serves them the same way on every
GPU the job has (one worker each, as upstream), with two differences that do not change any result:
it listens on 127.0.0.1 only (upstream 0.0.0.0) and does not write LitServe's example client.py.
The weights load from the paths the classes hard-code, scripts/moldet_v2_yolo11n_960_doc.pt and
scripts/molglyph_large.pt, which run.py links into the work folder.
"""

import argparse
import importlib

# server -> (module in scripts/, LitAPI class, port its BioMiner client calls, constructor arguments)
SERVERS = {
    "mineru": ("mineru_server", "MinerUAPI", 8002, {"output_dir": "./tmp"}),
    "moldet": ("moldet_server", "MolGlyphAPI", 8001, {}),  # upstream names its MolDetV2 class MolGlyphAPI too
    "ocsr": ("ocsr_server", "MolGlyphAPI", 8003, {}),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server", choices=sorted(SERVERS))
    parser.add_argument("--gpus", type=int, required=True, help="GPUs to serve on: cuda:0 .. cuda:N-1")
    args = parser.parse_args()

    import litserve as ls

    module, name, port, kwargs = SERVERS[args.server]
    # Imported by name from scripts/ (on PYTHONPATH) so that LitServe's spawned workers can unpickle it.
    api = getattr(importlib.import_module(module), name)(**kwargs)
    server = ls.LitServer(api, accelerator="cuda", devices=list(range(args.gpus)), workers_per_device=1, timeout=False)
    print(f"Starting BioMiner {args.server} server ({module}.{name}) on {args.gpus} GPU(s), port {port}", flush=True)
    server.run(host="127.0.0.1", port=port, generate_client_file=False)


if __name__ == "__main__":
    main()
