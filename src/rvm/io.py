"""Atomic, portable file operations. No implicit reads of answer metadata."""

import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
import zipfile
from pathlib import Path

import yaml
from tqdm.auto import tqdm


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(value).hexdigest()


def file_hash(path, algorithm="sha256", *, show_progress=False):
    path = Path(path)
    h = hashlib.new(algorithm)
    with (
        path.open("rb") as stream,
        tqdm(
            total=path.stat().st_size,
            desc=f"Hash {path.name}",
            unit="B",
            unit_scale=True,
            disable=not show_progress,
        ) as progress,
    ):
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
            progress.update(len(block))
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path):
    return json.loads(Path(path).read_text())


def load_config(path="config.example.yaml"):
    return yaml.safe_load(Path(path).read_text())


def portable_config(config):
    """Keep experiment settings without an attached dataset's absolute Kaggle location."""
    import copy

    saved = copy.deepcopy(config)
    saved["dataset"]["input_path"] = None
    return saved


def portable(root, relpath):
    root = Path(root).resolve()
    path = (root / relpath).resolve()
    if Path(relpath).is_absolute() or "\\" in str(relpath) or not path.is_relative_to(root):
        raise ValueError(f"unsafe relative path: {relpath}")
    return path


def safe_extract(archive, destination, allowed=None, max_bytes=80 * 1024**3, *, show_progress=False):
    """Reject links, traversal, devices, duplicate names and expansion beyond budget.

    allowed(name) may select RGB episodes while streaming a full .tgz on Kaggle.
    No extractall, no unpickling, no executable archive content.
    """
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    total, seen = 0, set()

    def copy(name, size, stream):
        nonlocal total
        target = portable(dest, name)
        if name in seen:
            raise ValueError(f"duplicate archive entry: {name}")
        seen.add(name)
        total += size
        if total > max_bytes or shutil.disk_usage(dest).free < size + 128 * 1024**2:
            raise OSError("archive expansion exceeds free disk/budget; use a larger Kaggle working volume")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as output:
            shutil.copyfileobj(stream, output)
        if target.stat().st_size != size:
            raise ValueError("truncated archive member")

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for member in tqdm(
                z.infolist(), desc="Extract ZIP archive", unit="member", disable=not show_progress
            ):
                portable(dest, member.filename)
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError("archive links forbidden")
                if not member.is_dir() and (allowed is None or allowed(member.filename)):
                    with z.open(member) as source:
                        copy(member.filename, member.file_size, source)
    else:
        with tarfile.open(archive, "r|*") as tar:
            for member in tqdm(
                tar, desc="Scan/extract RGB archive", unit="member", disable=not show_progress
            ):
                portable(dest, member.name)
                if not (member.isfile() or member.isdir()):
                    raise ValueError("archive links/devices forbidden")
                if member.isfile() and (allowed is None or allowed(member.name)):
                    with tar.extractfile(member) as source:
                        copy(member.name, member.size, source)
    return dest


def write_checksums(root, *, show_progress=False):
    root = Path(root)
    paths = [p for p in sorted(root.rglob("*")) if p.is_file() and p.name != "checksums.json"]
    checks = {
        p.relative_to(root).as_posix(): file_hash(p)
        for p in tqdm(paths, desc="Write bundle checksums", unit="file", disable=not show_progress)
    }
    atomic_json(root / "checksums.json", checks)


def verify_checksums(root, *, show_progress=False):
    root = Path(root)
    checks = read_json(root / "checksums.json")
    actual = {
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != "checksums.json"
    }
    if set(checks) != actual:
        raise ValueError("bundle file list differs from checksums")
    for name, expected in tqdm(
        checks.items(), desc="Verify bundle checksums", unit="file", disable=not show_progress
    ):
        if file_hash(portable(root, name)) != expected:
            raise ValueError(f"checksum mismatch: {name}")


def export_zip(root, target, *, show_progress=True):
    """Only allowlisted export directories belong in root. Never export a work/cache directory."""
    from .bundle import validate_bundle

    root, target = Path(root), Path(target)
    if target.resolve().is_relative_to(root.resolve()):
        raise ValueError("ZIP must be outside bundle directory")
    allowed = {
        "images",
        "annotation",
        "provenance",
        "model_snapshot",
        "frames.json",
        "selection.json",
        "records.json",
        "questions.json",
        "caption_vectors.npy",
        "vector_frame_ids.json",
        "embedding.json",
        "development-freeze.json",
        "bundle.json",
        "checksums.json",
        "sampling_windows.json",
        "review",
    }
    if any(p.name not in allowed for p in root.iterdir()):
        raise ValueError("Unexpected file in export root; keep work, secrets and caches outside the bundle")
    if any(
        p.is_symlink() or p.name in {".env", "credentials", "token", "budget.json"} for p in root.rglob("*")
    ):
        raise ValueError("Symlink or private work/credential file in export root")
    write_checksums(root, show_progress=show_progress)
    validate_bundle(root, show_progress=show_progress)
    paths = [p for p in sorted(root.rglob("*")) if p.is_file()]
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in tqdm(paths, desc="Write compact ZIP", unit="file", disable=not show_progress):
            archive.write(path, path.relative_to(root))
    with tempfile.TemporaryDirectory() as fresh:
        safe_extract(target, fresh, show_progress=show_progress)
        validate_bundle(fresh, show_progress=show_progress)
    return target
