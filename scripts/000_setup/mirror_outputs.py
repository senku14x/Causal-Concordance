"""Mirror what is not in git to Hugging Face, so a fresh instance can pull it back (restore_outputs.py).

    python scripts/000_setup/mirror_outputs.py            # outputs/ (+ a results/ snapshot) -> public dataset
    python scripts/000_setup/mirror_outputs.py --private  # also repo.bundle (every branch), /workspace/claude_reports
                                                          # and the agent memory -> private dataset

Public:  senku21x/causal-concordance-outputs. The outputs/ tree at the root, MANIFEST.json (bytes + sha256 per file;
         restore_outputs.py restores exactly these into outputs/), and results_snapshot/<branch>/ (results/ of the
         branch this runs on, for browsing without GitHub; listed under MANIFEST["results_snapshot"], not restored).
Private: senku21x/causal-concordance-private. Existing entries are kept; new or changed files are added to its
         MANIFEST.json. Secrets (/workspace/.env) are never uploaded.
Third-party model weights (HF_HOME) are not mirrored: rebuild_env.sh re-downloads them pinned by revision and sha256.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

PROJECT = Path(__file__).resolve().parents[2]
PUBLIC = "senku21x/causal-concordance-outputs"
PRIVATE = "senku21x/causal-concordance-private"
SECRET_HINTS = ("sk-or-", "OPENROUTER_API_KEY")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entries(root, prefix=""):
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and ".ipynb_checkpoints" not in p.parts and p.name != ".env":
            out.append({"path": prefix + str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": sha(p)})
    return out


def no_secrets(paths):
    for p in paths:
        if p.stat().st_size < 50 << 20 and p.suffix in (".json", ".txt", ".md", ".py", ".log", ".yaml", ".sh"):
            t = p.read_text(errors="ignore")
            assert not any(h in t for h in SECRET_HINTS), f"possible secret in {p}"


def git(*a):
    return subprocess.run(["git", *a], cwd=PROJECT, capture_output=True, text=True, check=True).stdout.strip()


def public(api):
    commit, branch = git("rev-parse", "HEAD"), git("branch", "--show-current")
    out = PROJECT / "outputs"
    no_secrets([p for p in out.rglob("*") if p.is_file()])
    files = entries(out)
    snap = entries(PROJECT / "results", f"results_snapshot/{branch}/")
    man = {"repo_commit": commit, "branch": branch, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "registry_sha256": sha(PROJECT / "scripts/common/registry.json"), "files": files,
           "results_snapshot": {"branch": branch, "files": snap,
                                "note": "results/ of this branch for browsing; not restored (results/ is in git)"}}
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "MANIFEST.json").write_text(json.dumps(man, indent=1))
        api.upload_large_folder(repo_id=PUBLIC, repo_type="dataset", folder_path=str(out),
                                ignore_patterns=["**/.ipynb_checkpoints/**", "**/.env"])
        api.upload_folder(repo_id=PUBLIC, repo_type="dataset", folder_path=str(PROJECT / "results"),
                          path_in_repo=f"results_snapshot/{branch}", commit_message=f"results snapshot {branch} @ {commit[:7]}")
        api.upload_file(path_or_fileobj=str(Path(td) / "MANIFEST.json"), path_in_repo="MANIFEST.json", repo_id=PUBLIC,
                        repo_type="dataset", commit_message=f"MANIFEST: {len(files)} outputs files @ {commit[:7]} ({branch})")
    print(f"public: {len(files)} outputs files ({sum(f['bytes'] for f in files) / 1e9:.2f} GB) + {len(snap)} results files")


def private(api):
    man = json.loads(Path(hf_hub_download(PRIVATE, "MANIFEST.json", repo_type="dataset")).read_text())
    have = {f["path"]: f for f in man["files"]}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        subprocess.run(["git", "bundle", "create", str(td / "repo.bundle"), "--all"], cwd=PROJECT, check=True,
                       capture_output=True)
        stage = td / "stage"
        stage.mkdir()
        shutil.copy(td / "repo.bundle", stage / "repo.bundle")
        srcs = {"workspace/claude_reports": Path("/workspace/claude_reports"),
                "workspace/agent_memory": Path("/root/.claude/projects/-workspace/memory")}
        for dst, src in srcs.items():
            if src.exists():
                shutil.copytree(src, stage / dst, ignore=shutil.ignore_patterns(".ipynb_checkpoints", ".env"))
        no_secrets([p for p in stage.rglob("*") if p.is_file()])
        new = entries(stage)
        for f in new:
            have[f["path"]] = f
        man["files"] = sorted(have.values(), key=lambda f: f["path"])
        man["repo_commit"], man["generated_utc"] = git("rev-parse", "HEAD"), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (stage / "MANIFEST.json").write_text(json.dumps(man, indent=1))
        api.upload_folder(repo_id=PRIVATE, repo_type="dataset", folder_path=str(stage),
                          commit_message="repo bundle (all branches), claude_reports, agent memory")
    print(f"private: {len(new)} files updated; manifest {len(man['files'])} files")


if __name__ == "__main__":
    api = HfApi()
    public(api)
    if "--private" in sys.argv:
        private(api)
