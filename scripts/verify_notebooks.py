"""Execute both source notebooks in explicit synthetic mode and retain reviewable output copies."""

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/verification/single-episode"
OUTPUT.mkdir(parents=True, exist_ok=True)
os.environ["RVM_NOTEBOOK_SMOKE"] = "1"
os.environ["RVM_WORK"] = str(OUTPUT / "kaggle-smoke")
os.environ["RVM_RESULTS"] = str(OUTPUT / "evaluation-smoke")
for source in sorted((ROOT / "notebooks").glob("*.ipynb")):
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    NotebookClient(
        notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
    ).execute()
    path = OUTPUT / (source.stem + ".synthetic-executed.ipynb")
    nbformat.write(notebook, path)
    print(path.relative_to(ROOT))
