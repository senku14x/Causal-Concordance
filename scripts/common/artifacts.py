"""Download and hash-verify every pinned artifact (model, lenses, reader checkpoints, WSB banks).

Usage:
    python scripts/common/artifacts.py download [--what model,lenses,readers]
    python scripts/common/artifacts.py verify   [--out results/000_setup/analysis/artifact_hashes.json]

Verification compares every downloaded file's sha256 against (a) the registry for the three lens
files and the WSB banks, and (b) the LFS sha256 the Hugging Face API reports for that file at the
pinned revision, for everything else (model shards, reader adapters, critic).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REG = json.loads((HERE / "registry.json").read_text())
os.environ.setdefault("HF_HOME", REG["paths"]["hf_home"])
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


def sha256(path, buf=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while (b := f.read(buf)):
            h.update(b)
    return h.hexdigest()


def _hf_lfs_oids(repo, revision, prefix=""):
    """{path: sha256} for LFS files at a pinned revision (non-LFS files map to None)."""
    from huggingface_hub import HfApi
    api = HfApi()
    out = {}
    for e in api.list_repo_tree(repo, revision=revision, recursive=True, path_in_repo=prefix or None):
        if getattr(e, "size", None) is None:
            continue  # folder
        lfs = getattr(e, "lfs", None)
        out[e.path] = (lfs.sha256 if lfs is not None else None)
    return out


def _model_patterns():
    return ["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "merges.txt", "vocab.json"]


def download(what=("model", "lenses", "readers")):
    from huggingface_hub import hf_hub_download, snapshot_download
    t0 = time.time()
    paths = {}
    if "lenses" in what:
        for name, e in REG["lenses"].items():
            p = hf_hub_download(e["hf_repo"], e["file"], revision=e["hf_revision"])
            paths[name] = p
            print(f"[lens] {name} -> {p} ({time.time()-t0:.0f}s)", flush=True)
    if "readers" in what:
        o = REG["readers"]["oracle_lens"]
        paths["oracle_lens"] = snapshot_download(o["hf_repo"], revision=o["hf_revision"],
                                                 allow_patterns=[f"{o['subfolder']}/*", "README.md"])
        print(f"[reader] oracle -> {paths['oracle_lens']} ({time.time()-t0:.0f}s)", flush=True)
        n = REG["readers"]["nla_L42"]
        pats = [n["meta"], "README.md"] + [f"{n[k]}/*" for k in ("av_sft_lora", "av_rl_lora", "ar_critic")]
        paths["nla_L42"] = snapshot_download(n["hf_repo"], revision=n["hf_revision"], allow_patterns=pats,
                                             max_workers=8)
        print(f"[reader] nla_L42 -> {paths['nla_L42']} ({time.time()-t0:.0f}s)", flush=True)
        r = REG["readers"]["nla_reference_activations"]
        paths["nla_reference_activations"] = hf_hub_download(r["hf_repo"], r["file"], revision=r["hf_revision"])
        q = REG["readers"]["nla_rl"]
        # + the final AR (ar_reconstructor/) and the root nla_meta.yaml: 013-015's reconstructor (EasyNLA NLACriticModel)
        # reads both; missing on a fresh instance before 2026-10-11 (015 ledger). Not hash-pinned in the registry.
        paths["nla_rl"] = snapshot_download(q["hf_repo"], revision=q["hf_revision"], max_workers=8,
                                            allow_patterns=[f"{q['av_base']}/*", f"{q['adapter']}/*", "ar_reconstructor/*", "nla_meta.yaml"])
        print(f"[reader] nla_rl -> {paths['nla_rl']} ({time.time()-t0:.0f}s)", flush=True)
    if "model" in what:
        m = REG["model"]
        paths["model"] = snapshot_download(m["repo"], revision=m["revision"], allow_patterns=_model_patterns(),
                                           max_workers=8)
        print(f"[model] -> {paths['model']} ({time.time()-t0:.0f}s)", flush=True)
    return paths


def verify(out=None):
    """Hash every pinned file; return a dict and write it to `out` if given. Raises on any mismatch."""
    from huggingface_hub import hf_hub_download, snapshot_download
    res = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": {}, "mismatches": []}

    def check(label, path, expected):
        got = sha256(path)
        ok = (expected is None) or (got == expected)
        res["files"][label] = {"path": str(path), "sha256": got, "expected": expected, "ok": ok,
                               "bytes": os.path.getsize(path)}
        if not ok:
            res["mismatches"].append(label)
        print(f"{'OK ' if ok else 'BAD'} {label} {got[:16]}", flush=True)

    for name, e in REG["lenses"].items():
        p = hf_hub_download(e["hf_repo"], e["file"], revision=e["hf_revision"], local_files_only=True)
        check(f"lens:{name}", p, e["sha256"])
    m = REG["model"]
    snap = Path(snapshot_download(m["repo"], revision=m["revision"], local_files_only=True,
                                  allow_patterns=_model_patterns()))
    oids = _hf_lfs_oids(m["repo"], m["revision"])
    for f in sorted(snap.rglob("*.safetensors")):
        rel = str(f.relative_to(snap))
        check(f"model:{rel}", f, oids.get(rel))
    o = REG["readers"]["oracle_lens"]
    osnap = Path(snapshot_download(o["hf_repo"], revision=o["hf_revision"], local_files_only=True,
                                   allow_patterns=[f"{o['subfolder']}/*"]))
    ooids = _hf_lfs_oids(o["hf_repo"], o["hf_revision"], o["subfolder"])
    for f in sorted((osnap / o["subfolder"]).glob("*.safetensors")):
        rel = str(f.relative_to(osnap))
        check(f"oracle:{rel}", f, ooids.get(rel))
    n = REG["readers"]["nla_L42"]
    nsnap = Path(snapshot_download(n["hf_repo"], revision=n["hf_revision"], local_files_only=True,
                                   allow_patterns=[f"{n[k]}/*" for k in ("av_sft_lora", "av_rl_lora", "ar_critic")]))
    noids = _hf_lfs_oids(n["hf_repo"], n["hf_revision"])
    for k in ("av_sft_lora", "av_rl_lora", "ar_critic"):
        for f in sorted((nsnap / n[k]).glob("*.safetensors")):
            rel = str(f.relative_to(nsnap))
            check(f"nla:{rel}", f, noids.get(rel))
    q = REG["readers"]["nla_rl"]
    qsnap = Path(snapshot_download(q["hf_repo"], revision=q["hf_revision"], local_files_only=True,
                                   allow_patterns=[f"{q['av_base']}/*", f"{q['adapter']}/*"]))
    for rel, want in sorted(q["sha256"].items()):
        check(f"nla_rl:{rel}", qsnap / rel, want)
    r = REG["readers"]["nla_reference_activations"]
    check("nla:reference_activations",
          hf_hub_download(r["hf_repo"], r["file"], revision=r["hf_revision"], local_files_only=True), r["sha256"])
    wsb = Path(REG["paths"]["wsb_clone"])
    for name, b in REG["wsb"]["banks"].items():
        check(f"wsb:{name}", wsb / b["file"], b["sha256"])
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res["all_ok"] = not res["mismatches"]
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, indent=1))
    if res["mismatches"]:
        raise SystemExit(f"hash mismatches: {res['mismatches']}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "verify"])
    ap.add_argument("--what", default="model,lenses,readers")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "download":
        download(tuple(a.what.split(",")))
    else:
        verify(a.out)
