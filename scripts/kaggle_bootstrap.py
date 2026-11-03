"""Standard-library-only Git bootstrap embedded into the standalone Kaggle notebook.

The checkout lives directly in the target directory. In-place init/fetch/checkout also works
when Kaggle has already written files there. Git's normal checkout protections preserve
untracked files; no force/reset/clean operations are used.
"""

import os
import subprocess
from pathlib import Path


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
