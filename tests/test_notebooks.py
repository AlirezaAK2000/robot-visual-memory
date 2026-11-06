import ast
import hashlib
import json
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["01_prepare_openeqa_kaggle.ipynb", "02_graph_retrieval_local.ipynb"])
def test_notebook_schema_and_python_cells(name):
    notebook = nbformat.read(ROOT / "notebooks" / name, as_version=4)
    nbformat.validate(notebook)
    assert len([c for c in notebook.cells if c.cell_type == "markdown"]) >= 5
    for cell in notebook.cells:
        if cell.cell_type == "code" and not notebook.metadata.get("rvm_recorded_run"):
            ast.parse(cell.source)


def test_recorded_kaggle_notebook_keeps_flat_git_bootstrap():
    notebook = nbformat.read(ROOT / "notebooks/01_prepare_openeqa_kaggle.ipynb", as_version=4)
    sources = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "Path('/kaggle/working')" in sources
    assert "checkout_repository(REPOSITORY_URL, REPOSITORY_REF, SOURCE" in sources
    assert "os.chdir(SOURCE)" in sources
    assert "SOURCE_ZIP" not in sources and "extractall" not in sources


@pytest.mark.parametrize("name", ["02_graph_retrieval_local.ipynb"])
def test_offline_notebook_execution(name, tmp_path, monkeypatch):
    from nbclient import NotebookClient

    monkeypatch.setenv("RVM_NOTEBOOK_SMOKE", "1")
    monkeypatch.setenv("RVM_WORK", str(tmp_path / "work"))
    monkeypatch.setenv("RVM_RESULTS", str(tmp_path / "results"))
    notebook = nbformat.read(ROOT / "notebooks" / name, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=90,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT / "notebooks" if name.startswith("02_") else ROOT)}},
    )
    client.execute()
    assert all(not any(o.output_type == "error" for o in cell.get("outputs", [])) for cell in notebook.cells)
    if name.startswith("02_"):
        from rvm.bundle import validate_bundle

        html = "\n".join(
            output.get("data", {}).get("text/html", "")
            for cell in notebook.cells
            for output in cell.get("outputs", [])
        )
        assert html.count("<h2>Example ") == 3
        assert html.count("Inspect generated Cypher and parameters") == 3
        for question in validate_bundle(tmp_path / "work/local-graph")["questions"]:
            assert question.question in html


# Recorded run evidence is deliberately immutable while source and prose are edited.
RECORDED_OUTPUT_HASHES = {
    "01_prepare_openeqa_kaggle.ipynb": "177db4b68f27ae8e403ffd10643ae5f5ca77b6e9101377dafd95a7d75a32c260",
    "02_graph_retrieval_local.ipynb": "304313bf8b5fcb68e97afb5fc71907b51fdc3ad18f2ae51fd5c8c1ddf466af1c",
}


@pytest.mark.parametrize("name", list(RECORDED_OUTPUT_HASHES))
def test_recorded_notebook_outputs_and_execution_counts_are_preserved(name):
    notebook = json.loads((ROOT / "notebooks" / name).read_text())
    payload = [
        {"id": c.get("id"), "outputs": c.get("outputs", []), "execution_count": c.get("execution_count")}
        for c in notebook["cells"]
    ]
    assert (
        hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        == RECORDED_OUTPUT_HASHES[name]
    )
    nbformat.validate(nbformat.read(ROOT / "notebooks" / name, as_version=4))
