"""Validate a portable export before any database writes or evaluation."""

from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .annotation import load_qrels, validate_qrels
from .embeddings import validate_vectors
from .io import digest, file_hash, portable, read_json, safe_extract, verify_checksums
from .schemas import SCHEMA_VERSION, Frame, Question, Record


def open_bundle(path, *, destination=None, show_progress=False):
    path = Path(path)
    if path.is_dir():
        return path
    if destination is None:
        destination = path.with_suffix("")
    if Path(destination).exists() and any(Path(destination).iterdir()):
        raise ValueError("unpack destination must be empty to avoid mixing exports")
    safe_extract(path, destination, show_progress=show_progress)
    return Path(destination)


def validate_bundle(root, *, checksums=True, show_progress=False):
    root = Path(root)
    if checksums:
        verify_checksums(root, show_progress=show_progress)
    manifest = read_json(root / "bundle.json")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported bundle schema")
    frames = [Frame.model_validate(f) for f in read_json(root / "frames.json")]
    records = [Record.model_validate(r) for r in read_json(root / "records.json")]
    questions = [Question.model_validate(q) for q in read_json(root / "questions.json")]
    fids = [f.frame_id for f in frames]
    rids = [r.frame_id for r in records]
    if not fids or len(set(fids)) != len(fids) or len(set(rids)) != len(rids) or set(fids) != set(rids):
        raise ValueError("duplicate/missing manifest or extraction frame IDs")
    if len({q.query_id for q in questions}) != len(questions):
        raise ValueError("duplicate question IDs")
    split_map = {}
    for f in tqdm(frames, desc="Validate gallery images", unit="frame", disable=not show_progress):
        if f.environment_id != f.episode_history:
            raise ValueError("environment/history linkage differs")
        if f.environment_id in split_map and split_map[f.environment_id] != f.split:
            raise ValueError("environment leaks between splits")
        split_map[f.environment_id] = f.split
        image = portable(root, f.image_relpath)
        if not image.is_file() or file_hash(image) != f.image_sha256:
            raise ValueError(f"missing/corrupt image: {f.image_relpath}")
        from PIL import Image

        with Image.open(image) as im:
            if im.size != (f.width, f.height):
                raise ValueError("manifest image dimensions differ")
            im.verify()
    selection = read_json(root / "selection.json")
    scope = selection.get("evaluation_scope", "held_out_environments")
    if scope == "single_episode_exploratory":
        if (
            len(split_map) != 1
            or set(split_map.values()) != {"evaluation"}
            or selection.get("development_environment") is not None
        ):
            raise ValueError("single-episode exploratory mode requires exactly one evaluation environment")
        if len(frames) > 200:
            raise ValueError("single-episode gallery exceeds 200 frames")
        from .episode_selection import validate_single_episode_sampling

        validate_single_episode_sampling(root, selection, frames)
    elif scope == "held_out_environments":
        if sum(v == "development" for v in split_map.values()) != 1:
            raise ValueError("exactly one development environment is required")
    else:
        raise ValueError("unknown evaluation scope")
    for q in questions:
        if q.environment_id != q.episode_history or split_map.get(q.environment_id) != q.split:
            raise ValueError("question environment/split not in gallery")
    mapping = read_json(root / "vector_frame_ids.json")
    if mapping != fids:
        raise ValueError("vector row IDs differ from ordered frame manifest")
    vectors = np.load(root / "caption_vectors.npy", allow_pickle=False)
    embedding = read_json(root / "embedding.json")
    validate_vectors(vectors, len(frames), embedding["dimension"])
    caption_by_id = {r.frame_id: r.extraction.caption for r in records}
    if embedding.get("caption_hashes") != [digest(caption_by_id[fid]) for fid in fids]:
        raise ValueError("caption source hashes differ from the embedding rows")
    contract_keys = [
        "model",
        "revision",
        "dimension",
        "max_length",
        "query_instruction",
        "pooling",
        "normalize",
        "dtype",
    ]
    if digest({k: embedding[k] for k in contract_keys}) != manifest["embedding_contract_hash"]:
        raise ValueError("embedding provenance contract mismatch")
    judgments = load_qrels(root / "annotation/qrels.csv")
    validate_qrels(judgments, frames, questions)
    ids = read_json(root / "annotation/query_ids.json")
    qmap = {q.query_id: q for q in questions}
    if (
        not ids
        or len(set(ids)) != len(ids)
        or any(qid not in qmap or qmap[qid].split != "evaluation" for qid in ids)
    ):
        raise ValueError("annotation query selection must contain unique evaluation queries")
    return {
        "root": root,
        "manifest": manifest,
        "frames": frames,
        "records": records,
        "questions": questions,
        "vectors": vectors,
        "embedding": embedding,
        "judgments": judgments,
    }


def finalize_bundle(root, *, synthetic=False, show_progress=False):
    import importlib.metadata
    import platform
    from .annotation import annotation_status
    from .io import atomic_json, write_checksums
    from .normalization import rules_hash

    root = Path(root)
    frames = [Frame.model_validate(f) for f in read_json(root / "frames.json")]
    ids = read_json(root / "annotation/query_ids.json")
    questions = [
        Question.model_validate(q) for q in read_json(root / "questions.json") if q["query_id"] in ids
    ]
    status = annotation_status(
        load_qrels(root / "annotation/qrels.csv"), frames, questions, synthetic=synthetic
    )
    atomic_json(root / "annotation/status.json", status)
    embedding = read_json(root / "embedding.json")
    keys = [
        "model",
        "revision",
        "dimension",
        "max_length",
        "query_instruction",
        "pooling",
        "normalize",
        "dtype",
    ]
    versions = {}
    for name in [
        "robot-visual-memory",
        "openai",
        "sentence-transformers",
        "torch",
        "neo4j",
        "qdrant-client",
        "pydantic",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    atomic_json(
        root / "provenance/runtime-packages.json",
        {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata.get("Name")},
    )
    atomic_json(
        root / "bundle.json",
        {
            "schema_version": SCHEMA_VERSION,
            "project": "Robot Visual Memory: Graph Retrieval vs Caption Embedding Retrieval",
            "synthetic": synthetic,
            "benchmark_status": "synthetic-test-only"
            if synthetic
            else "annotation-complete"
            if all(s["complete"] for s in status)
            else "annotation-pending",
            "embedding_contract_hash": digest({k: embedding[k] for k in keys}),
            "rules_hash": rules_hash(),
            "versions": versions,
            "python": platform.python_version(),
            "selected_images_only": True,
            "evaluation_scope": read_json(root / "selection.json").get(
                "evaluation_scope", "held_out_environments"
            ),
        },
    )
    write_checksums(root, show_progress=show_progress)
    return validate_bundle(root, show_progress=show_progress)
