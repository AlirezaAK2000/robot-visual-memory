"""Inventory episode lengths, select the shortest, and review disjoint-window sampling.

Inventory reads archive headers without extracting other episodes' image files. Windows use
sequence positions because the prepared RGB release does not provide verified timestamps.
"""

import csv
import re
import stat
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tqdm.auto import tqdm

from .io import atomic_json, file_hash, portable, read_json


def inventory_scope(qa_rows, config):
    available = sorted({r["episode_history"] for r in qa_rows if r["episode_history"].startswith("hm3d-v0/")})
    requested = config["dataset"].get("environment_ids", [])
    if requested:
        if len(requested) != len(set(requested)) or not set(requested) <= set(available):
            raise ValueError("environment_ids must be unique HM3D episode histories present in metadata")
        return sorted(requested)
    return available


def inventory_episodes(source, qa_rows, config):
    """Count documented RGB members for every present episode in scope, before any sampling.

    Missing episodes are reported separately, never interpreted as zero-frame candidates.
    Only episode membership/filenames/counts affect selection, never answers/question text.
    """
    source = Path(source)
    scope = inventory_scope(qa_rows, config)
    scope_set = set(scope)
    names = defaultdict(set)
    ignored = set()
    seen_paths = set()

    def add(name):
        name = str(name).replace("\\", "/")
        if not re.search(r"-rgb\.(png|jpg|jpeg)$", name, re.I):
            return
        match = re.search(r"(?:^|/)(hm3d-v0/[^/]+)/", name)
        if not match:
            return
        episode = match.group(1)
        if episode not in scope_set:
            ignored.add(episode)
            return
        basename = name.rsplit("/", 1)[-1]
        if name in seen_paths or basename in names[episode]:
            raise ValueError(f"Duplicate/ambiguous RGB frame filename for {episode}: {basename}")
        seen_paths.add(name)
        names[episode].add(basename)

    if source.is_dir():
        for path in tqdm(source.rglob("*"), desc="Inventory RGB directory", unit="entry"):
            if path.is_file():
                if path.is_symlink():
                    raise ValueError("Attached extracted RGB inputs must not contain symlinks")
                add(path.relative_to(source).as_posix())
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for member in tqdm(archive.infolist(), desc="Inventory ZIP members", unit="member"):
                portable(Path("/inventory-only"), member.filename)
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError("Archive links forbidden")
                if not member.is_dir():
                    add(member.filename)
    else:
        # Streaming decompresses the archive once, without writing unselected image files.
        with tarfile.open(source, "r|*") as archive:
            for member in tqdm(archive, desc="Inventory RGB archive members", unit="members"):
                portable(Path("/inventory-only"), member.name)
                if not (member.isfile() or member.isdir()):
                    raise ValueError("Archive links/devices forbidden")
                if member.isfile():
                    add(member.name)
    rows = [
        {"environment_id": episode, "episode_history": episode, "frame_count": len(files)}
        for episode, files in sorted(names.items())
    ]
    if not rows:
        raise ValueError("No HM3D *-rgb images with QA metadata found in the attached/downloaded collection")
    present = set(names)
    summary = {
        "scope": "available HM3D episodes with QA metadata",
        "metadata_episodes": len(scope),
        "collected_episodes": len(rows),
        "all_metadata_episodes_present": present == scope_set,
        "missing_episodes": sorted(scope_set - present),
        "ignored_episodes": sorted(ignored),
        "total_rgb_frames": sum(r["frame_count"] for r in rows),
        "source_type": "directory" if source.is_dir() else "archive",
        "selection_basis": "RGB filename counts; not image content, question text, or answers",
    }
    return rows, summary


def select_episode_by_length(inventory, selection_mode="smallest"):
    if not inventory or any(r["frame_count"] <= 0 for r in inventory):
        raise ValueError("Inventory must contain positive frame counts")
    if len({r["environment_id"] for r in inventory}) != len(inventory):
        raise ValueError("Inventory episode IDs must be unique")
    if selection_mode not in {"smallest", "largest"}:
        raise ValueError("Episode length selection must be smallest or largest")
    sign = 1 if selection_mode == "smallest" else -1
    return min(inventory, key=lambda r: (sign * r["frame_count"], r["environment_id"]))["environment_id"]


def select_shortest_episode(inventory):
    return select_episode_by_length(inventory, "smallest")


def save_inventory(inventory, summary, directory, *, synthetic=False, selection_mode="smallest"):
    """Standalone CSV + JSON + PNG histogram, with exact counts available for review."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.ticker import MaxNLocator

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    selected = select_episode_by_length(inventory, selection_mode)
    selected_count = next(r["frame_count"] for r in inventory if r["environment_id"] == selected)
    counts = np.array([r["frame_count"] for r in inventory], dtype=int)
    atomic_json(
        directory / "episode_inventory.json",
        {
            **summary,
            "episodes": inventory,
            "selected_episode": selected,
            "selected_frame_count": selected_count,
            "selection_mode": selection_mode,
            "synthetic": synthetic,
        },
    )
    with (directory / "episode_frame_counts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["environment_id", "episode_history", "frame_count"])
        writer.writeheader()
        writer.writerows(inventory)
    # Integer-centred bins for small ranges, capped to readable bins for longer collections.
    span = int(counts.max() - counts.min() + 1)
    bins = (
        np.arange(counts.min() - 0.5, counts.max() + 1.5)
        if span <= 30
        else np.linspace(counts.min() - 0.5, counts.max() + 0.5, 21)
    )
    fig = Figure(figsize=(9, 4.8), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    frequencies, edges, _ = ax.hist(counts, bins=bins, color="#3476b4", edgecolor="white")
    ax.axvline(
        selected_count,
        color="#c13831",
        linestyle="--",
        label=f"Selected {'shortest' if selection_mode == 'smallest' else 'longest'}: {selected_count} frames",
    )
    ax.set(
        xlabel="Original RGB frames per episode (before sampling)",
        ylabel="Number of episodes",
        title=("SYNTHETIC TEST ONLY — " if synthetic else "")
        + f"Collected HM3D episode lengths (n={len(counts)})",
    )
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(directory / "episode_frame_count_histogram.png", dpi=160)
    atomic_json(
        directory / "episode_histogram.json",
        {"counts": frequencies.astype(int).tolist(), "bin_edges": edges.tolist(), "synthetic": synthetic},
    )
    return directory / "episode_frame_count_histogram.png"


def disjoint_window_sample(frame_count, max_frames=200):
    """Partition [0, N) into min(N, cap) positive disjoint windows; take each midpoint.

    Windows cover the complete sequence and differ in width by at most one. If N<=cap,
    every frame is retained. End indexes are exclusive. No FPS/timestamps are fabricated.
    """
    if frame_count < 1 or not 1 <= max_frames <= 200:
        raise ValueError("positive frame count and max_frames in 1..200 required")
    # Presentation: evenly cover the recording with one midpoint per disjoint window.
    boundaries = np.linspace(0, frame_count, min(frame_count, max_frames) + 1, dtype=int)
    return [
        {
            "window_index": i,
            "start_index": int(start),
            "end_index_exclusive": int(end),
            "selected_index": int(start + (end - start - 1) // 2),
            "axis": "sequence_index",
        }
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
    ]


def make_preview_video(paths, target, *, sequence_indexes=None, fps=6, synthetic=False, size=(768, 480)):
    """Encode actual ordered images as an H.264 MP4, with aspect-preserving letterboxing.

    FPS is review speed, not capture time. Each input contributes exactly one video frame.
    imageio-ffmpeg wheels provide ffmpeg on Kaggle/CPU machines; no simulator or hosted API.
    """
    import imageio_ffmpeg

    paths = list(map(Path, paths))
    indexes = list(range(len(paths))) if sequence_indexes is None else list(sequence_indexes)
    if (
        not paths
        or len(paths) != len(indexes)
        or fps <= 0
        or any(a >= b for a, b in zip(indexes, indexes[1:]))
    ):
        raise ValueError("Video requires ordered unique image indexes and positive playback FPS")
    if size[0] % 2 or size[1] % 2:
        raise ValueError("H.264 preview dimensions must be even")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + ".partial.mp4")
    writer = imageio_ffmpeg.write_frames(
        str(temporary),
        size,
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="error",
        output_params=["-movflags", "+faststart"],
    )
    try:
        writer.send(None)
        for i, (path, original_index) in enumerate(
            tqdm(zip(paths, indexes), total=len(paths), desc=f"Encode {target.name}", unit="frame")
        ):
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image = ImageOps.contain(image, (size[0], size[1] - 48))
                canvas = Image.new("RGB", size, "black")
                canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - 48 - image.height) // 2))
            ImageDraw.Draw(canvas).text(
                (8, size[1] - 40),
                ("SYNTHETIC TEST ONLY | " if synthetic else "")
                + f"Review frame {i + 1}/{len(paths)} | original index {original_index}\nPlayback {fps} fps; original timestamps unavailable",
                fill="white",
            )
            writer.send(canvas.tobytes())
        writer.close()
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("ffmpeg did not produce a preview")
        temporary.replace(target)
    except BaseException:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    provenance = {
        "frame_count": len(paths),
        "sequence_indexes": indexes,
        "playback_fps": fps,
        "duration_seconds": len(paths) / fps,
        "source_timing": "unknown; playback rate only",
        "width": size[0],
        "height": size[1],
        "sha256": file_hash(target),
        "synthetic": synthetic,
    }
    atomic_json(target.with_suffix(".json"), provenance)
    return target


def validate_single_episode_sampling(root, selection, frames):
    """Keep selection, histogram, window provenance, gallery and preview aligned on import."""
    root = Path(root)
    if not (root / "review/episode_inventory.json").is_file():
        raise ValueError(
            "Bundle review inventory is missing. Restore inventory/histogram metadata before import; "
            "a manual longest-episode backup can be recovered with rvm recover-longest."
        )
    inventory = read_json(root / "review/episode_inventory.json")
    rows = inventory["episodes"]
    env = frames[0].environment_id
    mode = selection["config"]["dataset"].get("environment_selection", "smallest")
    if inventory.get("selection_mode", "smallest") != mode:
        raise ValueError("inventory selection mode differs from saved configuration")
    if (
        not rows
        or len({r["environment_id"] for r in rows}) != len(rows)
        or select_episode_by_length(rows, mode) != env
    ):
        raise ValueError(f"selected episode does not match {mode} selection over unique inventoried episodes")
    if inventory["selected_episode"] != env:
        raise ValueError("inventory-selected episode differs from gallery")
    original_count = next(r["frame_count"] for r in rows if r["environment_id"] == env)
    if selection["original_frame_counts"] != {env: original_count}:
        raise ValueError("original frame count differs between inventory and selection")
    cap = selection["config"]["dataset"]["frames_per_environment"]
    expected = disjoint_window_sample(original_count, cap)
    windows = read_json(root / "sampling_windows.json")
    if len(windows) != len(frames) or len(windows) != len(expected):
        raise ValueError("sampling-window count differs from gallery")
    for window, required, frame in zip(windows, expected, frames):
        if any(window[key] != value for key, value in required.items()):
            raise ValueError("sampling windows overlap, leave gaps, or use unexpected representatives")
        if (
            window["environment_id"] != env
            or window["frame_id"] != frame.frame_id
            or window["selected_index"] != frame.sequence_index
        ):
            raise ValueError("sampling windows refer to different frames/environment")
    histogram = read_json(root / "review/episode_histogram.json")
    if sum(histogram["counts"]) != len(rows):
        raise ValueError("histogram does not cover the inventoried episodes")
    video = root / "review/sampled_frames.mp4"
    provenance = read_json(video.with_suffix(".json"))
    if (
        provenance["frame_count"] != len(frames)
        or provenance["sequence_indexes"] != [f.sequence_index for f in frames]
        or provenance["sha256"] != file_hash(video)
    ):
        raise ValueError("sampled video provenance differs from selected gallery")
