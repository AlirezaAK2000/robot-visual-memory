"""Pinned third-party episode archives, materialized incrementally without a second full copy."""

import re
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import requests
from tqdm.auto import tqdm

from .io import atomic_json, file_hash, portable, read_json, safe_extract


def collect_hm3d_mirror(config, work, qa_rows):
    """Collect PNG episodes for inventory; original Dropbox archive identity is unverified.

    Verify each TAR against its pinned HF LFS SHA-256 before extraction. Keep a per-episode
    completion checkpoint; remove only that episode's downloaded TAR after extraction succeeds.
    This keeps temporary storage near one complete image collection plus one episode TAR.
    """
    from .preparation import download

    settings = config["dataset"]
    repository, revision = settings["mirror_repository"], settings["mirror_revision"]
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository) or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Mirror requires an owner/repository and a pinned 40-character commit SHA")
    response = requests.get(
        f"https://huggingface.co/api/datasets/{repository}/revision/{revision}",
        params={"blobs": "true"},
        timeout=(20, 60),
    )
    response.raise_for_status()
    metadata = response.json()
    if metadata.get("sha") != revision:
        raise ValueError("Mirror API returned a different revision")
    scope = {row["episode_history"] for row in qa_rows if row["episode_history"].startswith("hm3d-v0/")}
    requested = settings.get("environment_ids", [])
    if requested:
        if len(set(requested)) != len(requested) or not set(requested) <= scope:
            raise ValueError("environment_ids must be unique HM3D episode histories present in metadata")
        scope = set(requested)
    files = []
    for item in metadata["siblings"]:
        name = item["rfilename"]
        if name.endswith(".tar") and name[:-4] in scope:
            sha256 = item.get("lfs", {}).get("sha256", "")
            if not re.fullmatch(r"[a-f0-9]{64}", sha256) or item.get("size", 0) <= 0:
                raise ValueError(f"Mirror TAR has no usable size/LFS SHA-256: {name}")
            files.append({"path": name, "size": item["size"], "sha256": sha256})
    files.sort(key=lambda item: item["path"])
    if not files or len({item["path"] for item in files}) != len(files):
        raise ValueError("Mirror has no matching HM3D TARs or contains duplicate paths")

    root = Path(work) / "source" / f"hf-openeqa-{revision}"
    frames = root / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    pending = []
    completed_bytes = 0
    for item in files:
        episode = item["path"][:-4]
        marker = root / "completed" / f"{Path(episode).name}.json"
        old = read_json(marker) if marker.exists() else {}
        valid = (
            old.get("archive") == item
            and old.get("revision") == revision
            and bool(old.get("images"))
            and all(
                (p := portable(frames, image["path"])).is_file() and p.stat().st_size == image["size"]
                for image in old.get("images", [])
            )
        )
        if valid:
            completed_bytes += item["size"]
        else:
            pending.append((item, marker))
    remaining = sum(item["size"] for item, _ in pending)
    if pending and shutil.disk_usage(root).free < max(
        remaining + max(item["size"] for item, _ in pending) + 128 * 1024**2,
        settings["min_free_gb"] * 1024**3 - completed_bytes,
    ):
        raise OSError(
            "Insufficient disk for remaining mirror PNGs plus an episode TAR; use a larger Kaggle volume"
        )
    with tqdm(
        total=len(files),
        initial=len(files) - len(pending),
        desc="Collect HM3D mirror episodes",
        unit="episode",
    ) as progress:
        for item, marker in pending:
            episode = item["path"][:-4]
            archive = download(
                f"https://huggingface.co/datasets/{repository}/resolve/{revision}/{quote(item['path'])}",
                root / "archives" / item["path"],
            )
            if (
                archive.stat().st_size != item["size"]
                or file_hash(archive, show_progress=True) != item["sha256"]
            ):
                archive.unlink()  # discard only our failed downloaded artifact, allowing a fresh retry
                raise ValueError(f"Mirror TAR size/SHA-256 mismatch: {item['path']}")
            images, prefixes = [], set()
            # These are uncompressed TARs: header iteration seeks over PNG payloads.
            with tarfile.open(archive, "r:") as tar:
                for member in tar:
                    portable(frames, member.name)
                    if not (member.isfile() or member.isdir()):
                        raise ValueError("Mirror archive links/devices forbidden")
                    if not member.isfile() or not re.search(r"-rgb\.(png|jpg|jpeg)$", member.name, re.I):
                        continue
                    parts = PurePosixPath(member.name).parts
                    if parts[:-1] == (Path(episode).name,):
                        prefixes.add("short")
                    elif parts[:-1] == tuple(episode.split("/")):
                        prefixes.add("full")
                    else:
                        raise ValueError(f"Unexpected RGB episode path in mirror TAR: {member.name}")
                    images.append({"path": f"{episode}/{parts[-1]}", "size": member.size})
            if not images or len(prefixes) != 1 or len({i["path"] for i in images}) != len(images):
                raise ValueError("Mirror TAR has missing, duplicate or mixed-layout RGB images")
            destination = frames / "hm3d-v0" if prefixes == {"short"} else frames
            safe_extract(
                archive,
                destination,
                allowed=lambda name: bool(re.search(r"-rgb\.(png|jpg|jpeg)$", name, re.I)),
                show_progress=True,
            )
            atomic_json(marker, {"archive": item, "revision": revision, "images": images})
            archive.unlink()  # verified PNGs and completion record now replace this temporary TAR
            progress.update(1)
    provenance = {
        "provider": "huggingface_episode_mirror",
        "repository": repository,
        "revision": revision,
        "archives": files,
        "verification": "per-TAR SHA-256 from pinned HF LFS metadata before extraction",
        "original_dropbox_archive_byte_identity": "unverified; different archive packaging",
        "missing_metadata_episodes": sorted(scope - {item["path"][:-4] for item in files}),
    }
    atomic_json(root / "source.json", provenance)
    return frames, provenance
