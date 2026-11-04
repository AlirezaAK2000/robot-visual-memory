from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from rvm.bundle import finalize_bundle, validate_bundle
from rvm.episode_selection import (
    inventory_episodes,
    make_preview_video,
    save_inventory,
    select_episode_by_length,
)
from rvm.fixture import complete_prepared_fixture, create_episode_collection, create_single_episode_fixture
from rvm.handoff import recover_longest_backup
from rvm.io import (
    atomic_json,
    export_zip,
    file_hash,
    load_config,
    read_json,
    safe_extract,
    write_checksums,
)
from rvm.preparation import discover_rgb, prepare_gallery, select_environments

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def backup(tmp_path):
    config = load_config(ROOT / "config.example.yaml")
    raw, qa = create_episode_collection(tmp_path / "raw")
    inventory, summary = inventory_episodes(raw, qa, config)
    backup = tmp_path / "backup"
    save_inventory(inventory, summary, backup / "compact-bundle/review", synthetic=True)
    env = select_episode_by_length(inventory, "largest")
    bundle = backup / "compact-bundle-longest"
    # Reproduce the manual override: largest episode, legacy smallest config, new bundle root.
    frames, questions = prepare_gallery(discover_rgb(raw, [env]), qa, [env], config, bundle)
    selection = read_json(bundle / "selection.json")
    selection["policy"] = "highest collected RGB frame count"
    atomic_json(bundle / "selection.json", selection)
    complete_prepared_fixture(bundle)
    make_preview_video(
        [bundle / f.image_relpath for f in frames],
        bundle / "review/sampled_frames.mp4",
        sequence_indexes=[f.sequence_index for f in frames],
        synthetic=True,
    )
    with pytest.raises(ValueError, match="review inventory is missing"):
        finalize_bundle(bundle, synthetic=True)
    return backup


@pytest.mark.parametrize("format", ["directory", "zip"])
def test_recovery_preserves_source_and_payloads_and_passes_normal_export_cli(backup, tmp_path, format):
    original = backup / "compact-bundle-longest"
    before = {p.relative_to(original).as_posix(): file_hash(p) for p in original.rglob("*") if p.is_file()}
    source = backup
    if format == "zip":
        source = tmp_path / "backup.zip"
        with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in backup.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(backup))
        zip_hash = file_hash(source)
    target = tmp_path / "importable"
    report = recover_longest_backup(source, target, show_progress=False)
    assert report["frames"] == 9
    assert report["selected_episode"] == "hm3d-v0/000-synthetic-long"
    assert report["hosted_model_calls"] == 0
    assert all(file_hash(original / name) == value for name, value in before.items())
    if format == "zip":
        assert file_hash(source) == zip_hash
    journal = read_json(target / "provenance/graph-handoff.json")
    assert file_hash(target / "records.json") == journal["paid_records_sha256"]
    assert not (target / "caption_vectors.npy").exists()
    assert not (target / "annotation").exists()
    assert read_json(target / "selection.json")["config"]["dataset"]["environment_selection"] == "largest"
    saved = read_json(target / "review/episode_inventory.json")
    assert saved["selection_mode"] == "largest" and saved["selected_frame_count"] == 9
    assert sum(read_json(target / "review/episode_histogram.json")["counts"]) == len(saved["episodes"]) == 3
    archive = export_zip(target, tmp_path / "compact.zip", show_progress=False)
    fresh = safe_extract(archive, tmp_path / "fresh")
    assert len(validate_bundle(fresh)["frames"]) == 9
    result = subprocess.run(
        [sys.executable, "-m", "rvm.cli", "validate", str(target)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    with pytest.raises(ValueError, match="fresh destination"):
        recover_longest_backup(source, target, show_progress=False)


@pytest.mark.parametrize("fault", ["inventory", "payload"])
def test_recovery_rejects_inconsistent_backup_before_creating_destination(backup, tmp_path, fault):
    bundle = backup / "compact-bundle-longest"
    if fault == "inventory":
        path = backup / "compact-bundle/review/episode_inventory.json"
        inventory = read_json(path)
        inventory["episodes"][1]["frame_count"] = 100
        atomic_json(path, inventory)
    elif fault == "payload":
        next((bundle / "images").glob("*.jpg")).write_bytes(b"corrupt")
    target = tmp_path / "importable"
    with pytest.raises(ValueError):
        recover_longest_backup(backup, target, show_progress=False)


def test_largest_selection_is_consistent_and_legacy_shortest_still_works(tmp_path):
    config = load_config(ROOT / "config.example.yaml")
    raw, qa = create_episode_collection(tmp_path / "raw")
    inventory, _ = inventory_episodes(raw, qa, config)
    config["dataset"]["environment_selection"] = "largest"
    envs = select_environments(qa, config, inventory=inventory)
    assert envs == ["hm3d-v0/000-synthetic-long"]
    frames, questions = prepare_gallery(discover_rgb(raw, envs), qa, envs, config, tmp_path / "largest")
    assert (
        select_episode_by_length(
            [{"environment_id": "b", "frame_count": 9}, {"environment_id": "a", "frame_count": 9}], "largest"
        )
        == "a"
    )
    legacy = create_single_episode_fixture(tmp_path / "legacy", config_path=ROOT / "config.example.yaml")
    data = read_json(legacy / "review/episode_inventory.json")
    data.pop("selection_mode")
    atomic_json(legacy / "review/episode_inventory.json", data)
    write_checksums(legacy)
    assert len(validate_bundle(legacy)["frames"]) == 5
