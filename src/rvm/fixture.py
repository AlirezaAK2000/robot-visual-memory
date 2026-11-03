"""Tiny synthetic fixture: colored drawings and hand-authored records, NEVER benchmark data."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .annotation import save_qrels
from .bundle import finalize_bundle
from .io import atomic_json, digest, file_hash, read_json
from .schemas import Attribute, Extraction, Frame, Judgment, Object, Question, Record, Relation


def create_fixture(root):
    root = Path(root)
    frames, records = [], []
    for i, (env, split) in enumerate(
        [
            ("synthetic/dev", "development"),
            ("synthetic/test", "evaluation"),
            ("synthetic/test", "evaluation"),
            ("synthetic/other", "evaluation"),
        ]
    ):
        fid = f"synthetic-frame-{i}"
        relpath = f"images/{fid}.jpg"
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (320, 180), ["white", "yellow", "blue", "red"][i])
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), f"SYNTHETIC TEST ONLY {i}", fill="black")
        image.save(path)
        frames.append(
            Frame(
                environment_id=env,
                episode_history=env,
                frame_id=fid,
                sequence_index=i,
                image_relpath=relpath,
                width=320,
                height=180,
                source_sha256=file_hash(path),
                image_sha256=file_hash(path),
                dhash="0000000000000000",
                split=split,
            )
        )
        objects = [
            Object(
                object_id="a",
                label="kettle" if i != 2 else "chair",
                attributes=[Attribute(name="color", value="red" if i == 3 else "white")],
                states=["closed"],
            ),
            Object(object_id="b", label="table", attributes=[], states=[]),
        ]
        extraction = Extraction(
            caption=f"SYNTHETIC drawing {i}. A {objects[0].label} is on a table.",
            objects=objects,
            relationships=[Relation(source="a", predicate="on", target="b")],
        )
        records.append(Record(frame_id=fid, extraction=extraction, cache_key="synthetic-no-api"))
    questions = [
        Question(
            query_id="synthetic-q1",
            question="What is on the table beside the kettle?",
            environment_id="synthetic/test",
            episode_history="synthetic/test",
            category="spatial",
            split="evaluation",
        ),
        Question(
            query_id="synthetic-q0",
            question="Where is the mirror?",
            environment_id="synthetic/test",
            episode_history="synthetic/test",
            category="coverage",
            split="evaluation",
        ),
    ]
    atomic_json(root / "frames.json", [f.model_dump() for f in frames])
    atomic_json(root / "records.json", [r.model_dump(mode="json") for r in records])
    atomic_json(root / "questions.json", [q.model_dump() for q in questions])
    atomic_json(
        root / "annotation/reference_answers.json",
        [{"query_id": "synthetic-q1", "reference_answer": "SYNTHETIC"}],
    )
    atomic_json(root / "annotation/query_ids.json", [q.query_id for q in questions])
    save_qrels(
        root / "annotation/qrels.csv",
        [
            Judgment(
                query_id=q.query_id,
                frame_id=f.frame_id,
                relevance=int(q.query_id == "synthetic-q1" and f.frame_id == "synthetic-frame-1"),
                source="synthetic",
                status="confirmed",
            )
            for q in questions
            for f in frames
            if f.environment_id == q.environment_id
        ],
    )
    # Tiny, explicit synthetic vectors; no embedding model or API is called for this fixture.
    vectors = np.array([[1, 0, 0], [0.8, 0.6, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    np.save(root / "caption_vectors.npy", vectors, allow_pickle=False)
    atomic_json(root / "vector_frame_ids.json", [f.frame_id for f in frames])
    atomic_json(
        root / "embedding.json",
        {
            "model": "SYNTHETIC-NO-MODEL",
            "revision": "fixture-v1",
            "dimension": 3,
            "max_length": 512,
            "query_instruction": "",
            "pooling": "synthetic",
            "normalize": True,
            "dtype": "float32",
            "caption_hashes": [digest(r.extraction.caption) for r in records],
        },
    )
    atomic_json(root / "selection.json", {"development_environment": "synthetic/dev", "synthetic": True})
    finalize_bundle(root, synthetic=True)
    return root


def create_episode_collection(root):
    """Raw synthetic histories to exercise inventory/minimum selection/window sampling/video."""
    root = Path(root)
    qa = []
    for episode, count in [("000-synthetic-long", 9), ("001-synthetic-short", 5), ("002-synthetic-mid", 7)]:
        history = f"hm3d-v0/{episode}"
        directory = root / history
        directory.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            image = Image.new("RGB", (320, 180), (30 + i * 15, 70, 110))
            ImageDraw.Draw(image).text(
                (12, 12), f"SYNTHETIC TEST ONLY\n{episode}\nsequence {i}", fill="white"
            )
            image.save(directory / f"{i:05d}-rgb.png")
        for q in range(2):
            qa.append(
                {
                    "question_id": f"synthetic-{episode}-q{q}",
                    "episode_history": history,
                    "question": "What is visible?" if q == 0 else "Where is the kettle?",
                    "answer": "SYNTHETIC TEST ONLY",
                    "category": "synthetic",
                }
            )
    return root, qa


def complete_prepared_fixture(root):
    """Finish a prepared single-episode smoke gallery with explicit authored synthetic outputs."""
    root = Path(root)
    frames = [Frame.model_validate(f) for f in read_json(root / "frames.json")]
    questions = [Question.model_validate(q) for q in read_json(root / "questions.json")]
    records = [
        Record(
            frame_id=f.frame_id,
            extraction=Extraction(
                caption="SYNTHETIC drawing for offline tests.", objects=[], relationships=[]
            ),
            cache_key="synthetic-no-api",
        )
        for f in frames
    ]
    atomic_json(root / "records.json", [r.model_dump(mode="json") for r in records])
    vectors = np.zeros((len(frames), 3), dtype=np.float32)
    vectors[:, 0] = 1
    np.save(root / "caption_vectors.npy", vectors, allow_pickle=False)
    atomic_json(root / "vector_frame_ids.json", [f.frame_id for f in frames])
    atomic_json(
        root / "embedding.json",
        {
            "model": "SYNTHETIC-NO-MODEL",
            "revision": "fixture-v1",
            "dimension": 3,
            "max_length": 512,
            "query_instruction": "",
            "pooling": "synthetic",
            "normalize": True,
            "dtype": "float32",
            "caption_hashes": [digest(r.extraction.caption) for r in records],
        },
    )
    # These authored labels belong only to synthetic smoke data, never real human metrics.
    ids = read_json(root / "annotation/query_ids.json")
    save_qrels(
        root / "annotation/qrels.csv",
        [
            Judgment(
                query_id=q.query_id,
                frame_id=f.frame_id,
                relevance=int(q == questions[0] and f == frames[0]),
                source="synthetic",
                status="confirmed",
            )
            for q in questions
            if q.query_id in ids
            for f in frames
        ],
    )
    return root


def create_single_episode_fixture(root, *, config_path):
    """End-to-end synthetic one-episode bundle for clean-kernel local notebook verification."""
    from .annotation import create_materials
    from .episode_selection import inventory_episodes, save_inventory, make_preview_video
    from .io import load_config
    from .preparation import discover_rgb, prepare_gallery, select_environments

    root = Path(root)
    source, qa = create_episode_collection(root.parent / (root.name + "-raw-collection"))
    config = load_config(config_path)
    inventory, summary = inventory_episodes(source, qa, config)
    environments = select_environments(qa, config, inventory=inventory)
    save_inventory(inventory, summary, root / "review", synthetic=True)
    frames, questions = prepare_gallery(discover_rgb(source, environments), qa, environments, config, root)
    make_preview_video(
        [root / f.image_relpath for f in frames],
        root / "review/sampled_frames.mp4",
        sequence_indexes=[f.sequence_index for f in frames],
        fps=config["dataset"]["preview_fps"],
        synthetic=True,
    )
    create_materials(root, frames, questions, config)
    complete_prepared_fixture(root)
    finalize_bundle(root, synthetic=True)
    return root
