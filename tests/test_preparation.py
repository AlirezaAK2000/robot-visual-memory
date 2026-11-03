from pathlib import Path

from PIL import Image

from rvm.io import load_config, read_json
from rvm.preparation import discover_rgb, inspect_images, prepare_gallery, select_environments

ROOT = Path(__file__).resolve().parents[1]


def test_selection_reproducible_and_answer_independent(tmp_path):
    config = load_config(ROOT / "config.example.yaml")
    config["dataset"].update(
        environments=3,
        frames_per_environment=2,
        max_image_side=100,
        environment_selection="seeded",
        frame_sampling="linspace",
    )
    rows = [
        {
            "episode_history": f"hm3d-v0/{i:03}-test",
            "question_id": str(i),
            "question": "Where?",
            "answer": "SECRET",
            "category": "location",
        }
        for i in range(6)
    ]
    envs = select_environments(rows, config)
    changed = [{**r, "answer": "DIFFERENT", "question": "Different question"} for r in reversed(rows)]
    assert select_environments(changed, config) == envs
    for env in envs:
        folder = tmp_path / "rgb" / env
        folder.mkdir(parents=True)
        for i in range(4):
            Image.new("RGB", (200, 100), (i * 40, 20, 30)).save(folder / f"{i:05}-rgb.png")
            Image.new("L", (200, 100)).save(folder / f"{i:05}-depth.png")
    paths = discover_rgb(tmp_path / "rgb", envs)
    assert all(len(v) == 4 for v in paths.values())
    frames, _ = prepare_gallery(paths, rows, envs, config, tmp_path / "one")
    frames2, _ = prepare_gallery(paths, changed, envs, config, tmp_path / "two")
    assert [f.model_dump() for f in frames] == [f.model_dump() for f in frames2]
    assert all(f.width == 100 and f.height == 50 and f.timestamp is None for f in frames)
    assert [f.sequence_index for f in frames[:2]] == [0, 3]
    assert read_json(tmp_path / "one/selection.json")["pilot_frame_ids"] == [f.frame_id for f in frames[:2]]
    assert all(r["valid"] == 4 and not r["invalid"] for r in inspect_images(paths))
