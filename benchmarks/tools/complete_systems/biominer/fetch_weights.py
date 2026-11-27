"""Download BioMiner's pinned weights from Hugging Face and check every file against its revision.

Run by setup.sh: <python with huggingface_hub> fetch_weights.py <models folder>. Each repository is
fetched at a pinned commit into its folder under the models folder. Every file is then checked against
that revision's metadata, LFS files by sha256 and the others by git blob sha1, and the
.cache/huggingface folder that huggingface_hub leaves in a download folder is removed, because the
benchmark hashes a checkpoint folder whole (hidden files included). Last, it prints the sha256 of each
checkpoint as the harness computes it (molscout.hashing.sha256_tree). A repository that cannot be
fetched (a gated one the token has no access to) is reported and the others still run; the exit status
is then 1.
"""

import fnmatch
import hashlib
import shutil
import sys
from pathlib import Path

# (repository, revision, files to fetch, folder under the models folder, checkpoint path under it)
WEIGHTS = (
    # MolDetV2 (CC-BY-NC-SA-4.0: benchmark use only). Commits after the weights' upload (37e3ec5,
    # 2025-11-16) change only README and LICENSE; this file's LFS object is the same at both.
    ("UniParser/MolDetv2", "88b3836ae4fe692dd2d4c123b4aa0d4a5039eb75",
     ["moldet_v2_yolo11n_960_doc.pt"], ".", "moldet_v2_yolo11n_960_doc.pt"),
    # MinerU 1.3.1's pipeline models as BioMiner's README points to them (the old MinerU models that
    # download_models.py can no longer fetch). magic-pdf.json's models-dir is
    # mineru/mineru_old_version_models.
    ("jiaxianustc/BioMiner-MinerU-Model", "b3f7c96c9e49193ebfb2328cea651129e8f6b9c3",
     ["mineru_old_version_models/*"], "mineru", "mineru"),
    # MinerU's reading-order model (layoutreader-model-dir). download_models.py takes it from
    # ModelScope ppaanngggg/layoutreader; MinerU itself falls back to this Hugging Face repository.
    # transformers loads model.safetensors, so pytorch_model.bin (the same weights) is left out.
    ("hantian/layoutreader", "629be376d86fbab624ddc4020804a4e93b5515bc",
     ["config.json", "model.safetensors"], "layoutreader", "layoutreader"),
    # BioMiner-Instruct, the 32B MLLM (gated, approved on request).
    ("jiaxianustc/BioMiner-Instruct", "c865482f8c8b9515f8c9c04806071e27229be9bc",
     ["*"], "BioMiner-Instruct", "BioMiner-Instruct"),
)
NOT_WEIGHTS = (".gitattributes", "README.md")


def file_hashes(path):
    """(sha256, git blob sha1) of a file, read once."""
    sha256, sha1 = hashlib.sha256(), hashlib.sha1()
    sha1.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            sha256.update(chunk)
            sha1.update(chunk)
    return sha256.hexdigest(), sha1.hexdigest()


def matches(path, entry, hashes):
    """Whether a local file is the revision's file: its LFS sha256, else its git blob sha1."""
    if not path.is_file():
        return False
    sha256, sha1 = hashes(path)
    if entry.lfs is None:
        return sha1 == entry.blob_id
    lfs = entry.lfs
    return sha256 == (lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256)


def tree_sha256(path, hashes):
    """molscout.hashing.sha256_tree: a file's sha256, or of a folder's sorted "<sha256>  <relative path>" lines."""
    if path.is_file():
        return hashes(path)[0]
    files = sorted((f.relative_to(path).as_posix(), f) for f in path.rglob("*") if f.is_file())
    return hashlib.sha256("".join(f"{hashes(f)[0]}  {rel}\n" for rel, f in files).encode()).hexdigest()


def fetch(api, snapshot_download, models, repo, revision, patterns, folder, hashes):
    """Fetch one repository's files at the revision into models/folder and check them all."""
    info = api.model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError(f"{repo}: asked for {revision}, Hugging Face answered {info.sha}")
    wanted = [s for s in info.siblings if s.rfilename not in NOT_WEIGHTS
              and any(fnmatch.fnmatch(s.rfilename, p) for p in patterns)]
    if not wanted:
        raise RuntimeError(f"{repo}@{revision[:7]}: no files match {patterns}")
    target = models / folder
    missing = [s.rfilename for s in wanted if not matches(target / s.rfilename, s, hashes)]
    if missing:
        print(f"{repo}@{revision[:7]}: downloading {len(missing)} of {len(wanted)} files", flush=True)
        snapshot_download(repo, revision=revision, allow_patterns=missing, local_dir=target, max_workers=8)
        hashes.cache_clear()
    shutil.rmtree(target / ".cache", ignore_errors=True)
    bad = [s.rfilename for s in wanted if not matches(target / s.rfilename, s, hashes)]
    if bad:
        raise RuntimeError(f"{repo}@{revision[:7]}: files differ from the revision: {bad}")
    if folder != ".":
        expected = {s.rfilename for s in wanted}
        extra = sorted(f.relative_to(target).as_posix() for f in target.rglob("*")
                       if f.is_file() and f.relative_to(target).as_posix() not in expected)
        if extra:
            raise RuntimeError(f"{target} holds files that are not in {repo}@{revision[:7]}: {extra}")
    print(f"{repo}@{revision[:7]}: {len(wanted)} files match the revision", flush=True)


def main():
    from functools import lru_cache

    from huggingface_hub import HfApi, snapshot_download

    models = Path(sys.argv[1]).resolve()
    models.mkdir(parents=True, exist_ok=True)
    hashes = lru_cache(maxsize=None)(file_hashes)
    api = HfApi()
    failed = []
    for repo, revision, patterns, folder, _ in WEIGHTS:
        try:
            fetch(api, snapshot_download, models, repo, revision, patterns, folder, hashes)
        except Exception as exc:  # one gated or broken repository must not stop the others
            print(f"FAILED {repo}@{revision[:7]}: {type(exc).__name__}: {str(exc)[:400]}", flush=True)
            failed.append(repo)
    shutil.rmtree(models / ".cache", ignore_errors=True)
    for repo, revision, _, _, checkpoint in WEIGHTS:
        path = models / checkpoint
        if repo not in failed:
            print(f"sha256 {tree_sha256(path, hashes)}  {path}  ({repo}@{revision})", flush=True)
    if failed:
        print(f"not fetched: {', '.join(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
