"""Run fresh synthetic notebook copies without changing the saved presentation outputs."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/verification/notebooks"
OUTPUT.mkdir(parents=True, exist_ok=True)
os.environ["RVM_NOTEBOOK_SMOKE"] = "1"

with TemporaryDirectory(prefix="smoke-", dir=OUTPUT) as work:
    os.environ["RVM_WORK"] = work
    for source in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(source, as_version=4)
        nbformat.validate(notebook)
        if notebook.metadata.get("rvm_recorded_run"):
            continue
        NotebookClient(
            notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
        ).execute()
        path = OUTPUT / (source.stem + ".synthetic-executed.ipynb")
        nbformat.write(notebook, path)
        print(path.relative_to(ROOT))
