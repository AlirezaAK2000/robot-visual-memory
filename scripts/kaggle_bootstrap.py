"""Standard-library-only Git bootstrap embedded into the standalone Kaggle notebook.

The checkout lives directly in the target directory. In-place init/fetch/checkout also works
when Kaggle has already written files there. Git's normal checkout protections preserve
untracked files; no force/reset/clean operations are used.
"""

import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def loaded_dependency_versions():
    """Inspect already imported dependencies without importing any scientific libraries."""
    names = {
        "numpy": "numpy",
        "scipy": "scipy",
        "sklearn": "scikit-learn",
        "pandas": "pandas",
        "PIL": "Pillow",
        "sentence_transformers": "sentence-transformers",
        "transformers": "transformers",
        "tokenizers": "tokenizers",
    }
    return {
        distribution: str(module.__version__)
        for name, distribution in names.items()
        if (module := sys.modules.get(name)) is not None and getattr(module, "__version__", None) is not None
    }


def require_current_dependency_versions(loaded):
    """Stop after pip replaces an imported package; native extensions require a kernel restart."""
    mismatches = []
    for distribution, loaded_version in loaded.items():
        try:
            installed_version = version(distribution)
        except PackageNotFoundError:
            installed_version = "missing"
        if installed_version != loaded_version:
            mismatches.append(f"{distribution}: kernel={loaded_version}, installed={installed_version}")
    if mismatches:
        raise RuntimeError(
            "Dependencies changed while the kernel still has older modules loaded:\n"
            + "\n".join(mismatches)
            + "\nRestart the kernel, keep the same WORK directory, set UPDATE_REPOSITORY=False, "
            "and rerun from configuration. Existing downloads and extraction caches remain on disk. "
            "Do not reload NumPy in place."
        )


def checkout_repository(repository_url, ref, target, *, update=False):
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}

    def git(*args, required=True):
        result = subprocess.run(
            ["git", "-C", str(target), *args],
            env=environment,
            text=True,
            capture_output=True,
        )
        if required and result.returncode:
            raise RuntimeError(
                "Git bootstrap failed. Check Internet access/repository access and the message below. "
                "Existing files were not force-overwritten.\n" + result.stderr.strip()
            )
        return result

    if (target / ".git").exists():
        actual_root = Path(git("rev-parse", "--show-toplevel").stdout.strip()).resolve()
        if actual_root != target:
            raise RuntimeError("PROJECT_ROOT must be the root of its own checkout")
        origin = git("remote", "get-url", "origin", required=False)
        if origin.returncode == 0 and origin.stdout.strip().removesuffix(".git") != str(
            repository_url
        ).removesuffix(".git"):
            raise RuntimeError("PROJECT_ROOT already belongs to another repository; its origin was preserved")
        if origin.returncode:
            git("remote", "add", "origin", repository_url)
    else:
        git("init", "--quiet")
        git("remote", "add", "origin", repository_url)

    head = git("rev-parse", "--verify", "HEAD", required=False)
    previous_ref = git("config", "--local", "--get", "rvm.bootstrapRef", required=False).stdout.strip()
    dirty = bool(git("status", "--porcelain", "--untracked-files=no").stdout.strip())
    if previous_ref and previous_ref != ref and not update:
        raise RuntimeError(
            "REPOSITORY_REF changed; set UPDATE_REPOSITORY=True to check out the new ref explicitly"
        )
    reused = head.returncode == 0 and previous_ref == ref and not update
    if not reused:
        if dirty:
            raise RuntimeError(
                "Tracked files have local edits; commit/stash them before requesting a repository update"
            )
        git("fetch", "--depth", "1", "--no-tags", "origin", ref)
        # This refuses conflicting untracked files and preserves unrelated outputs/caches.
        git("checkout", "--detach", "FETCH_HEAD")
        git("config", "--local", "rvm.bootstrapRef", ref)

    for relative in ["pyproject.toml", "config.example.yaml", "src/rvm/__init__.py"]:
        if not (target / relative).is_file():
            raise RuntimeError(f"Repository is missing expected project file: {relative}")
    return {
        "repository_url": repository_url,
        "requested_ref": ref,
        "commit": git("rev-parse", "HEAD").stdout.strip(),
        "previous_commit": head.stdout.strip() if head.returncode == 0 else None,
        "reused_checkout": reused,
        "tracked_changes": bool(git("status", "--porcelain", "--untracked-files=no").stdout.strip()),
    }
