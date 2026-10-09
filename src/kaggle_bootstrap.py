"""Auto-configure paths on Kaggle (safe no-op locally)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/CrimeWizard/f1-podium-prediction.git"


def _find_csv_dir() -> Path | None:
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        return None
    for results in kaggle_input.rglob("results.csv"):
        folder = results.parent
        if (folder / "circuits.csv").exists() and (folder / "races.csv").exists():
            return folder
    return None


def bootstrap_kaggle(*, clone_repo: bool = True) -> dict:
    """
    Ensure F1_DATA_RAW and project code are available on Kaggle.
    Returns diagnostic dict for notebook prints.
    """
    info = {"is_kaggle": Path("/kaggle/working").exists(), "csv_dir": None, "repo_dir": None}

    csv_dir = _find_csv_dir()
    if csv_dir is not None:
        os.environ["F1_DATA_RAW"] = str(csv_dir)
        info["csv_dir"] = str(csv_dir)

    if not info["is_kaggle"]:
        info["repo_dir"] = str(Path(__file__).resolve().parents[1])
        return info

    work = Path("/kaggle/working")
    repo = work / "f1-podium-prediction"
    if clone_repo and not (repo / "src" / "io.py").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(repo)],
            check=True,
        )
    if repo.exists():
        info["repo_dir"] = str(repo)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        os.chdir(repo)

    return info
