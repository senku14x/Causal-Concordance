"""007 per-stage status (design §12.6): commit the stage's status file, results and code to the 007-jsubspace branch,
push (never main, never force), then mirror outputs/ to the public HF dataset (scripts/000_setup/mirror_outputs.py).

  python status.py <stage>          # results/007_jsubspace/status/<stage>.md must already be written
"""
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
ENV = {**os.environ, "GIT_AUTHOR_NAME": "senku14x", "GIT_AUTHOR_EMAIL": "visheshgupta14x@gmail.com",
       "GIT_COMMITTER_NAME": "senku14x", "GIT_COMMITTER_EMAIL": "visheshgupta14x@gmail.com"}
ENV.pop("GITHUB_TOKEN", None)


def git(*a, check=True):
    return subprocess.run(["git", *a], cwd=PROJECT, env=ENV, capture_output=True, text=True, check=check)


def main():
    stage = sys.argv[1]
    st = PROJECT / "results" / "007_jsubspace" / "status" / f"{stage}.md"
    assert st.exists(), st
    assert git("branch", "--show-current").stdout.strip() == "007-jsubspace"
    git("add", "results/007_jsubspace", "scripts/007_jsubspace", "tests/test_007_analysis.py",
        "docs/design_specs/007_jsubspace.md", "scripts/common/readers.py", "CLAUDE.md")
    if git("diff", "--cached", "--quiet", check=False).returncode != 0:
        msg = (f"007 {stage}: status, results and code (unattended run under spec v2 + A1)\n\n"
               "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
               "Claude-Session: https://claude.ai/code/session_014cXLtUXYUQ3xJAqHko9ofL")
        git("commit", "-q", "-m", msg)
    r = git("push", "-q", "origin", "007-jsubspace", check=False)
    print("push:", r.returncode, r.stderr.strip()[-200:])
    if "--no-mirror" not in sys.argv:
        m = subprocess.run([sys.executable, str(PROJECT / "scripts" / "000_setup" / "mirror_outputs.py")], cwd=PROJECT,
                           env=ENV, capture_output=True, text=True)
        print("mirror:", m.returncode, (m.stdout + m.stderr).strip()[-400:])


if __name__ == "__main__":
    main()
