import ast
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["01_prepare_openeqa_kaggle.ipynb", "02_evaluate_retrieval_local.ipynb"])
def test_notebook_schema_and_python_cells(name):
    notebook = nbformat.read(ROOT / "notebooks" / name, as_version=4)
    nbformat.validate(notebook)
    assert len([c for c in notebook.cells if c.cell_type == "markdown"]) >= 5
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert cell.execution_count is None  # shipped source notebooks make no execution claim


def test_kaggle_notebook_embeds_current_flat_git_bootstrap():
    notebook = nbformat.read(ROOT / "notebooks/01_prepare_openeqa_kaggle.ipynb", as_version=4)
    sources = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert (ROOT / "scripts/kaggle_bootstrap.py").read_text().strip() in sources
    assert "Path('/kaggle/working')" in sources
    assert "checkout_repository(REPOSITORY_URL, REPOSITORY_REF, SOURCE" in sources
    assert "os.chdir(SOURCE)" in sources
    assert "SOURCE_ZIP" not in sources and "extractall" not in sources


@pytest.mark.parametrize("name", ["01_prepare_openeqa_kaggle.ipynb", "02_evaluate_retrieval_local.ipynb"])
def test_offline_notebook_execution(name, tmp_path, monkeypatch):
    from nbclient import NotebookClient

    monkeypatch.setenv("RVM_NOTEBOOK_SMOKE", "1")
    monkeypatch.setenv("RVM_WORK", str(tmp_path / "work"))
    monkeypatch.setenv("RVM_RESULTS", str(tmp_path / "results"))
    notebook = nbformat.read(ROOT / "notebooks" / name, as_version=4)
    client = NotebookClient(
        notebook, timeout=90, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
    )
    client.execute()
    assert all(not any(o.output_type == "error" for o in cell.get("outputs", [])) for cell in notebook.cells)
    if name.startswith("01_"):
        from rvm.io import read_json

        review = tmp_path / "work/pilot-review"
        assert len(read_json(review / "records.json")) == 3
        html = (review / "extractions.html").read_text()
        assert html.count('<article class="rvm-card">') == 3
        assert "SYNTHETIC EXAMPLE" in html
        assert any(
            "Pilot extraction review" in o.get("data", {}).get("text/html", "")
            for cell in notebook.cells
            for o in cell.get("outputs", [])
        )
