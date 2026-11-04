"""Validate graph exports and saved legacy extraction results before database writes."""

from pathlib import Path

from tqdm.auto import tqdm

from .io import (
    atomic_json,
    digest,
    file_hash,
    portable,
    read_json,
    safe_extract,
    verify_checksums,
    write_checksums,
)
from .schemas import SCHEMA_VERSION, Frame, Question, Record


def open_bundle(path, *, destination=None, show_progress=False):
    path = Path(path)
    if path.is_dir():
        return path
    destination = Path(destination) if destination else path.with_suffix("")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("unpack destination must be empty to avoid mixing exports")
    safe_extract(path, destination, show_progress=show_progress)
    return destination


def content_hash(frames, records, questions):
    return digest(
        {
            "frames": [f.model_dump() for f in frames],
            "records": [r.model_dump(mode="json") for r in records],
            "questions": [q.model_dump() for q in questions],
        }
    )


def validate_bundle(root, *, checksums=True, show_progress=False):
    root = Path(root)
    if checksums:
        verify_checksums(root, show_progress=show_progress)
    manifest = read_json(root / "bundle.json")
    if manifest["schema_version"] not in {"1.0", SCHEMA_VERSION}:
        raise ValueError("unsupported bundle schema")
    frames = [Frame.model_validate(f) for f in read_json(root / "frames.json")]
    records = [Record.model_validate(r) for r in read_json(root / "records.json")]
    questions = (
        [Question.model_validate(q) for q in read_json(root / "questions.json")]
        if (root / "questions.json").exists()
        else []
    )
    fids, rids = [f.frame_id for f in frames], [r.frame_id for r in records]
    if not fids or len(set(fids)) != len(fids) or len(set(rids)) != len(rids) or set(fids) != set(rids):
        raise ValueError("duplicate/missing manifest or extraction frame IDs")
    if len({q.query_id for q in questions}) != len(questions):
        raise ValueError("duplicate question IDs")
    environments = {f.environment_id for f in frames}
    positions = set()
    for f in tqdm(frames, desc="Validate gallery images", unit="frame", disable=not show_progress):
        if f.environment_id != f.episode_history:
            raise ValueError("environment/history linkage differs")
        position = (f.environment_id, f.sequence_index)
        if position in positions:
            raise ValueError("duplicate sequence position")
        positions.add(position)
        image = portable(root, f.image_relpath)
        if not image.is_file() or file_hash(image) != f.image_sha256:
            raise ValueError(f"missing/corrupt image: {f.image_relpath}")
        from PIL import Image

        with Image.open(image) as im:
            if im.size != (f.width, f.height):
                raise ValueError("manifest image dimensions differ")
            im.verify()
    for q in questions:
        if q.environment_id != q.episode_history or q.environment_id not in environments:
            raise ValueError("question environment not in gallery")
    selection = read_json(root / "selection.json")
    if selection.get("config", {}).get("dataset", {}).get("environment_selection") in {"smallest", "largest"}:
        if len(environments) != 1 or len(frames) > 200:
            raise ValueError("single-episode gallery requires one environment and at most 200 frames")
        from .episode_selection import validate_single_episode_sampling

        validate_single_episode_sampling(root, selection, frames)
    if manifest["schema_version"] == SCHEMA_VERSION:
        if manifest.get("content_hash") != content_hash(frames, records, questions):
            raise ValueError("graph bundle content contract mismatch")
    return {"root": root, "manifest": manifest, "frames": frames, "records": records, "questions": questions}


def finalize_bundle(root, *, synthetic=False, show_progress=False):
    import importlib.metadata
    import platform

    root = Path(root)
    frames = [Frame.model_validate(f) for f in read_json(root / "frames.json")]
    records = [Record.model_validate(r) for r in read_json(root / "records.json")]
    questions = (
        [Question.model_validate(q) for q in read_json(root / "questions.json")]
        if (root / "questions.json").exists()
        else []
    )
    versions = {name: importlib.metadata.version(name) for name in ["openai", "neo4j", "pydantic"]}
    atomic_json(
        root / "bundle.json",
        {
            "schema_version": SCHEMA_VERSION,
            "project": "Robot Visual Memory: Graph Retrieval",
            "synthetic": synthetic,
            "content_hash": content_hash(frames, records, questions),
            "versions": versions,
            "python": platform.python_version(),
            "selected_images_only": True,
        },
    )
    write_checksums(root, show_progress=show_progress)
    return validate_bundle(root, show_progress=show_progress)
