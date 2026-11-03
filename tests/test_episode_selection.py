import io
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from rvm.bundle import validate_bundle
from rvm.episode_selection import (
    disjoint_window_sample,
    inventory_episodes,
    make_preview_video,
    save_inventory,
    select_shortest_episode,
)
from rvm.fixture import create_episode_collection, create_single_episode_fixture
from rvm.io import atomic_json, load_config, read_json, write_checksums
from rvm.preparation import acquire_rgb, discover_rgb, prepare_gallery, select_environments

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "n,cap", [(1, 200), (5, 200), (199, 200), (200, 200), (201, 200), (401, 200), (10003, 200), (551, 37)]
)
def test_windows_partition_and_hard_cap(n, cap):
    windows = disjoint_window_sample(n, cap)
    assert len(windows) == min(n, cap) <= 200
    assert windows[0]["start_index"] == 0 and windows[-1]["end_index_exclusive"] == n
    assert all(a["end_index_exclusive"] == b["start_index"] for a, b in zip(windows, windows[1:]))
    assert all(w["start_index"] <= w["selected_index"] < w["end_index_exclusive"] for w in windows)
    chosen = [w["selected_index"] for w in windows]
    assert chosen == sorted(set(chosen))
    widths = [w["end_index_exclusive"] - w["start_index"] for w in windows]
    assert max(widths) - min(widths) <= 1
    if n <= cap:
        assert chosen == list(range(n))
    assert windows == disjoint_window_sample(n, cap)


@pytest.mark.parametrize("n,cap", [(0, 200), (10, 0), (10, 201)])
def test_invalid_window_cap(n, cap):
    with pytest.raises(ValueError):
        disjoint_window_sample(n, cap)


@pytest.mark.parametrize("kind", ["directory", "zip", "tgz"])
def test_inventory_minimum_counts_and_only_selected_extraction(tmp_path, kind):
    config = load_config(ROOT / "config.example.yaml")
    directory, qa = create_episode_collection(tmp_path / "raw")
    source = directory
    if kind == "zip":
        source = tmp_path / "raw.zip"
        with zipfile.ZipFile(source, "w") as archive:
            for path in directory.rglob("*.png"):
                archive.write(path, path.relative_to(directory))
    elif kind == "tgz":
        source = tmp_path / "raw.tgz"
        with tarfile.open(source, "w:gz") as archive:
            for path in directory.rglob("*.png"):
                archive.add(path, arcname=path.relative_to(directory), recursive=False)
    inventory, summary = inventory_episodes(source, qa, config)
    assert sorted(r["frame_count"] for r in inventory) == [5, 7, 9]
    assert summary["all_metadata_episodes_present"]
    selected = select_environments(qa, config, inventory=inventory)
    assert selected == ["hm3d-v0/001-synthetic-short"]
    altered = [{**row, "question": "Changed", "answer": "Different"} for row in reversed(qa)]
    assert inventory_episodes(source, altered, config)[0] == inventory
    dest = acquire_rgb(config, tmp_path / "work", {}, selected, source=source)
    assert len(discover_rgb(dest, selected)[selected[0]]) == 5
    if kind != "directory":
        assert len(list(dest.rglob("*-rgb.png"))) == 5
        assert not (dest / "hm3d-v0/000-synthetic-long").exists()
    histogram = save_inventory(inventory, summary, tmp_path / "review", synthetic=True)
    assert histogram.stat().st_size > 1000
    hist = read_json(tmp_path / "review/episode_histogram.json")
    assert sum(hist["counts"]) == 3


def test_ties_missing_scope_and_duplicate_archive(tmp_path):
    assert (
        select_shortest_episode(
            [{"environment_id": "z", "frame_count": 2}, {"environment_id": "a", "frame_count": 2}]
        )
        == "a"
    )
    config = load_config(ROOT / "config.example.yaml")
    directory, qa = create_episode_collection(tmp_path / "raw")
    qa.append({"episode_history": "hm3d-v0/missing"})
    _, summary = inventory_episodes(directory, qa, config)
    assert not summary["all_metadata_episodes_present"]
    assert summary["missing_episodes"] == ["hm3d-v0/missing"]
    archive = tmp_path / "duplicate.tar"
    name = "hm3d-v0/001-synthetic-short/00000-rgb.png"
    with tarfile.open(archive, "w") as tar:
        for _ in range(2):
            member = tarfile.TarInfo(name)
            member.size = 1
            tar.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Duplicate"):
        inventory_episodes(archive, qa, config)


def test_video_has_actual_selected_frames_in_order(tmp_path):
    import imageio_ffmpeg
    from PIL import Image

    paths = []
    for i, red in enumerate([30, 100, 200]):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (100, 50), (red, 0, 0)).save(path)
        paths.append(path)
    path = make_preview_video(
        paths, tmp_path / "selected.mp4", sequence_indexes=[2, 5, 9], fps=3, synthetic=True, size=(320, 240)
    )
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    meta = next(reader)
    decoded = [np.frombuffer(frame, dtype=np.uint8).reshape((240, 320, 3)) for frame in reader]
    assert len(decoded) == 3 and meta["fps"] == 3
    assert np.allclose([im[96, 160, 0] for im in decoded], [30, 100, 200], atol=8)
    assert read_json(path.with_suffix(".json"))["sequence_indexes"] == [2, 5, 9]


def test_real_gallery_path_caps_a_long_synthetic_sequence(tmp_path):
    from PIL import Image

    config = load_config(ROOT / "config.example.yaml")
    env = "hm3d-v0/synthetic-long-sequence"
    directory = tmp_path / "raw" / env
    directory.mkdir(parents=True)
    for i in range(417):
        Image.new("RGB", (40, 20), (i % 256, 30, 40)).save(directory / f"{i:05}-rgb.png")
    qa = [
        {
            "episode_history": env,
            "question_id": "synthetic-q",
            "question": "What is visible?",
            "answer": "SYNTHETIC",
            "category": "synthetic",
        }
    ]
    frames, questions = prepare_gallery(
        discover_rgb(tmp_path / "raw", [env]), qa, [env], config, tmp_path / "bundle"
    )
    assert len(frames) == len({f.frame_id for f in frames}) == 200
    assert read_json(tmp_path / "bundle/selection.json")["pilot_frame_ids"] == [
        f.frame_id for f in frames[:3]
    ]
    assert all(f.split == "evaluation" and f.environment_id == env for f in frames)
    assert questions[0].split == "evaluation"
    windows = read_json(tmp_path / "bundle/sampling_windows.json")
    assert windows[0]["start_index"] == 0 and windows[-1]["end_index_exclusive"] == 417
    assert [f.sequence_index for f in frames] == [w["selected_index"] for w in windows]


@pytest.mark.parametrize("fault", ["windows", "inventory", "video"])
def test_single_episode_bundle_and_corruption(tmp_path, fault):
    root = create_single_episode_fixture(tmp_path / "bundle", config_path=ROOT / "config.example.yaml")
    bundle = validate_bundle(root)
    assert bundle["manifest"]["evaluation_scope"] == "single_episode_exploratory"
    assert len(bundle["frames"]) == 5 and len({f.environment_id for f in bundle["frames"]}) == 1
    assert all(f.split == "evaluation" for f in bundle["frames"])
    if fault == "windows":
        path = root / "sampling_windows.json"
        rows = read_json(path)
        rows[1]["start_index"] = 0
        atomic_json(path, rows)
    elif fault == "inventory":
        path = root / "review/episode_inventory.json"
        data = read_json(path)
        data["episodes"][0]["frame_count"] = 1
        atomic_json(path, data)
    else:
        path = root / "review/sampled_frames.json"
        data = read_json(path)
        data["sequence_indexes"].reverse()
        atomic_json(path, data)
    write_checksums(root)
    with pytest.raises(ValueError):
        validate_bundle(root)
