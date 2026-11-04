"""Create graph-only working bundles without modifying saved extraction results."""

import shutil
import tempfile
from copy import deepcopy
from pathlib import Path

from .bundle import finalize_bundle
from .episode_selection import save_inventory, select_episode_by_length
from .io import atomic_json, file_hash, read_json, safe_extract, verify_checksums
from .schemas import Frame, Question, Record


def prepare_graph_bundle(source, destination, *, show_progress=True):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError("Use a fresh destination outside the original results")
    if source.is_file():
        with tempfile.TemporaryDirectory(prefix="rvm-graph-") as temporary:
            safe_extract(source, temporary, show_progress=show_progress)
            result = _convert(Path(temporary), destination, show_progress)
            atomic_json(destination / "provenance/source-archive.json", {"sha256": file_hash(source)})
            finalize_bundle(destination, synthetic=result["synthetic"], show_progress=show_progress)
            return result
    return _convert(source, destination, show_progress)


def _convert(source, destination, show_progress):
    bundle = source / "compact-bundle-longest" if (source / "compact-bundle-longest").is_dir() else source
    verify_checksums(bundle, show_progress=show_progress)
    hashes = {
        p.relative_to(bundle).as_posix(): file_hash(p) for p in sorted(bundle.rglob("*")) if p.is_file()
    }
    frames = [Frame.model_validate(f) for f in read_json(bundle / "frames.json")]
    records = [Record.model_validate(r) for r in read_json(bundle / "records.json")]
    if {r.frame_id for r in records} != {f.frame_id for f in frames} or len(records) != len(frames):
        raise ValueError("Incomplete saved graph records")
    selection = deepcopy(read_json(bundle / "selection.json"))
    for directory in ["images", "review", "provenance"]:
        if (bundle / directory).exists():
            shutil.copytree(bundle / directory, destination / directory)
    (destination / "provenance").mkdir(parents=True, exist_ok=True)
    for name in ["selection.json", "bundle.json", "checksums.json", "frames.json"]:
        shutil.copyfile(bundle / name, destination / "provenance" / f"source-{name}")
    # Preserve paid graph output byte for byte. Frame metadata only loses obsolete split labels.
    shutil.copyfile(bundle / "records.json", destination / "records.json")
    atomic_json(destination / "frames.json", [f.model_dump() for f in frames])
    if (bundle / "questions.json").exists():
        atomic_json(
            destination / "questions.json",
            [Question.model_validate(q).model_dump() for q in read_json(bundle / "questions.json")],
        )
    for name in ["sampling_windows.json"]:
        if (bundle / name).exists():
            shutil.copyfile(bundle / name, destination / name)
    selection.pop("development_environment", None)
    selection.pop("evaluation_scope", None)
    config = selection.get("config", {})
    config.pop("embedding", None)
    config.get("dataset", {}).pop("questions_per_eval_environment", None)
    for key in ["k_values", "warm_repeats"]:
        config.get("retrieval", {}).pop(key, None)
    config.setdefault("retrieval", {})["top_k"] = 5
    inventory_path = destination / "review/episode_inventory.json"
    if not inventory_path.exists() and config.get("dataset", {}).get("environment_selection") in {
        "smallest",
        "largest",
    }:
        original = read_json(source / "compact-bundle/review/episode_inventory.json")
        selected = select_episode_by_length(original["episodes"], "largest")
        if {f.environment_id for f in frames} != {selected}:
            raise ValueError("Saved gallery is not the longest collected episode")
        config["dataset"]["environment_selection"] = "largest"
        save_inventory(original["episodes"], original, destination / "review", selection_mode="largest")
    atomic_json(destination / "selection.json", selection)
    atomic_json(
        destination / "provenance/graph-handoff.json",
        {
            "source_file_hashes": hashes,
            "normalization": "Applied at Neo4j import; raw records remain byte-identical",
            "paid_records_sha256": hashes["records.json"],
            "hosted_model_calls": 0,
        },
    )
    synthetic = read_json(bundle / "bundle.json")["synthetic"]
    validated = finalize_bundle(destination, synthetic=synthetic, show_progress=show_progress)
    if any(file_hash(bundle / name) != sha for name, sha in hashes.items()):
        raise RuntimeError("Original results changed")
    if file_hash(destination / "records.json") != hashes["records.json"]:
        raise RuntimeError("Saved extraction payload changed")
    return {
        "bundle": str(destination),
        "frames": len(validated["frames"]),
        "synthetic": synthetic,
        "selected_episode": frames[0].environment_id,
        "hosted_model_calls": 0,
    }


def recover_longest_backup(backup, destination, *, show_progress=True):
    return prepare_graph_bundle(backup, destination, show_progress=show_progress)
