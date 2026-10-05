"""Restore what is not in git on a fresh instance (see CLAUDE.md, "Fresh instance").

    python scripts/000_setup/restore_outputs.py            # outputs/ from the public HF dataset, sha256-checked
    python scripts/000_setup/restore_outputs.py --private  # also the private backup (papers, web sources, review
                                                           # evidence, Claude skills, repo bundle) into /workspace

Public:  senku21x/causal-concordance-outputs  (outputs/ tree at the root + MANIFEST.json with bytes and sha256;
         results_snapshot/<branch>/ is a browsable copy of results/ and is not restored). Update it with
         mirror_outputs.py.
Private: senku21x/causal-concordance-private  (workspace/<path> mirrors /workspace/<path>; repo.bundle; MANIFEST.json)
Secrets (the OpenRouter key in /workspace/.env) are in neither; ask the PI.
"""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT = Path(__file__).resolve().parents[2]
PUBLIC = "senku21x/causal-concordance-outputs"
PRIVATE = "senku21x/causal-concordance-private"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(root, manifest):
    bad = [f["path"] for f in manifest["files"] if not (root / f["path"]).exists() or sha(root / f["path"]) != f["sha256"]]
    print(f"{len(manifest['files']) - len(bad)}/{len(manifest['files'])} files match MANIFEST.json"
          + (f"; mismatched or missing: {bad[:10]}" if bad else ""))
    return not bad


def restore_public():
    dst = PROJECT / "outputs"
    tmp = Path(snapshot_download(PUBLIC, repo_type="dataset"))
    man = json.loads((tmp / "MANIFEST.json").read_text())
    for f in man["files"]:
        (dst / f["path"]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tmp / f["path"], dst / f["path"])
    print(f"outputs/ restored from {PUBLIC} (manifest commit {man['repo_commit'][:7]})")
    return check(dst, man)


def restore_private():
    tmp = Path(snapshot_download(PRIVATE, repo_type="dataset"))
    man = json.loads((tmp / "MANIFEST.json").read_text())
    ok = check(tmp, man)
    nb = man.get("not_backed_up")
    if nb:
        print(f"not in the private backup ({nb['reason']}): {len(nb['paths'])} files, e.g. {nb['paths'][:3]}")
    for f in man["files"]:
        if f["path"].startswith("workspace/"):
            out = Path("/") / f["path"]
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tmp / f["path"], out)
    print(f"private backup restored under /workspace; repo bundle at {tmp / 'repo.bundle'} "
          f"(git clone {tmp / 'repo.bundle'} recovers every branch); paper PDFs and web sources are in the repo "
          f"(docs/papers/, docs/sources/web/)")
    return ok


if __name__ == "__main__":
    ok = restore_public()
    if "--private" in sys.argv:
        ok = restore_private() and ok
    sys.exit(0 if ok else 1)
