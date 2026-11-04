"""Synthetic drawings and authored graphs for offline software checks."""

from pathlib import Path

from PIL import Image, ImageDraw

from .bundle import finalize_bundle
from .io import atomic_json, file_hash, read_json
from .schemas import Attribute, Extraction, Frame, Object, Question, Record, Relation


def create_fixture(root):
    root = Path(root)
    frames, records = [], []
    for i, env in enumerate(["synthetic/dev", "synthetic/test", "synthetic/test", "synthetic/other"]):
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
        ),
        Question(
            query_id="synthetic-q0",
            question="Where is the mirror?",
            environment_id="synthetic/test",
            episode_history="synthetic/test",
            category="coverage",
        ),
    ]
    atomic_json(root / "frames.json", [f.model_dump() for f in frames])
    atomic_json(root / "records.json", [r.model_dump(mode="json") for r in records])
    atomic_json(root / "questions.json", [q.model_dump() for q in questions])
    atomic_json(root / "selection.json", {"synthetic": True})
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
        for q, question in enumerate(["What is visible?", "Where is the kettle?", "Is the kettle closed?"]):
            qa.append(
                {
                    "question_id": f"synthetic-{episode}-q{q}",
                    "episode_history": history,
                    "question": question,
                    "answer": "SYNTHETIC TEST ONLY",
                    "category": "synthetic",
                }
            )
    return root, qa


def complete_prepared_fixture(root):
    """Finish a prepared single-episode smoke gallery with explicit authored synthetic outputs."""
    root = Path(root)
    frames = [Frame.model_validate(f) for f in read_json(root / "frames.json")]
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
    return root


def create_single_episode_fixture(root, *, config_path):
    """End-to-end synthetic one-episode bundle for clean-kernel local notebook verification."""
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
    complete_prepared_fixture(root)
    finalize_bundle(root, synthetic=True)
    return root
