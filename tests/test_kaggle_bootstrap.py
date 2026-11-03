"""Exercise the exact stdlib-only bootstrap embedded in notebook 01, using local Git remotes."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("kaggle_bootstrap", ROOT / "scripts/kaggle_bootstrap.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
checkout_repository = MODULE.checkout_repository


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


@pytest.fixture
def origin(tmp_path):
    path = tmp_path / "upstream"
    path.mkdir()
    git(path, "init", "-b", "main")
    git(path, "config", "user.name", "Synthetic bootstrap test")
    git(path, "config", "user.email", "synthetic@example.invalid")
    (path / "src/rvm").mkdir(parents=True)
    (path / "src/rvm/__init__.py").write_text('"""Synthetic test only."""\n')
    (path / "pyproject.toml").write_text("# synthetic project\n")
    (path / "config.example.yaml").write_text("synthetic: true\n")
    git(path, "add", ".")
    git(path, "commit", "-m", "Synthetic upstream fixture")
    return path


def test_checkout_is_flat_and_preserves_nonempty_working_directory(tmp_path, origin):
    target = tmp_path / "kaggle-working"
    target.mkdir()
    (target / "__notebook__.ipynb").write_text("existing notebook")
    (target / "rvm-single-episode").mkdir()
    (target / "rvm-single-episode/checkpoint.json").write_text("existing checkpoint")
    report = checkout_repository(str(origin), "main", target)
    assert (target / "pyproject.toml").exists()
    assert (target / "src/rvm/__init__.py").exists()
    assert not (target / "robot-visual-memory").exists()
    assert (target / "__notebook__.ipynb").read_text() == "existing notebook"
    assert (target / "rvm-single-episode/checkpoint.json").read_text() == "existing checkpoint"
    assert report["commit"] == git(origin, "rev-parse", "HEAD")
    assert not report["reused_checkout"]


def test_rerun_reuses_commit_and_update_is_explicit(tmp_path, origin):
    target = tmp_path / "working"
    first = checkout_repository(str(origin), "main", target)
    (origin / "new.txt").write_text("new upstream version")
    git(origin, "add", ".")
    git(origin, "commit", "-m", "Advance synthetic upstream")
    second = checkout_repository(str(origin), "main", target)
    assert second["commit"] == first["commit"] and second["reused_checkout"]
    assert not (target / "new.txt").exists()
    updated = checkout_repository(str(origin), "main", target, update=True)
    assert updated["commit"] != first["commit"] and not updated["reused_checkout"]
    assert (target / "new.txt").read_text() == "new upstream version"


def test_untracked_collision_is_never_overwritten(tmp_path, origin):
    target = tmp_path / "working"
    target.mkdir()
    (target / "pyproject.toml").write_text("keep my unrelated file")
    with pytest.raises(RuntimeError, match="not force-overwritten"):
        checkout_repository(str(origin), "main", target)
    assert (target / "pyproject.toml").read_text() == "keep my unrelated file"


def test_update_preserves_tracked_local_edits(tmp_path, origin):
    target = tmp_path / "working"
    checkout_repository(str(origin), "main", target)
    (target / "config.example.yaml").write_text("custom: local\n")
    assert checkout_repository(str(origin), "main", target)["tracked_changes"]
    with pytest.raises(RuntimeError, match="local edits"):
        checkout_repository(str(origin), "main", target, update=True)
    assert (target / "config.example.yaml").read_text() == "custom: local\n"


def test_different_repository_and_ref_change_require_explicit_handling(tmp_path, origin):
    target = tmp_path / "working"
    first = checkout_repository(str(origin), "main", target)
    with pytest.raises(RuntimeError, match="another repository"):
        checkout_repository(str(tmp_path / "different-upstream"), "main", target)
    assert git(target, "remote", "get-url", "origin") == str(origin)
    git(origin, "tag", "fixture-v1")
    with pytest.raises(RuntimeError, match="REPOSITORY_REF changed"):
        checkout_repository(str(origin), "fixture-v1", target)
    assert checkout_repository(str(origin), "fixture-v1", target, update=True)["commit"] == first["commit"]


def test_explicit_commit_sha(tmp_path, origin):
    commit = git(origin, "rev-parse", "HEAD")
    report = checkout_repository(str(origin), commit, tmp_path / "working")
    assert report["commit"] == commit
