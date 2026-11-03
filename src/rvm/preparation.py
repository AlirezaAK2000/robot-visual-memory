"""Kaggle-only acquisition and deterministic gallery preparation."""

import re
import shutil
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

import numpy as np
import requests
from PIL import Image, ImageOps
from tqdm.auto import tqdm

from .io import atomic_json, digest, file_hash, portable_config, read_json, safe_extract
from .schemas import Frame, Question

MANUAL_ROUTE = (
    "Open https://github.com/facebookresearch/open-eqa/blob/main/data/README.md, "
    "HM3D / Option 1 / third party location. Obtain the documented RGB .tgz, upload it as a "
    "private Kaggle Dataset and attach it via Add Input. Set dataset.input_path to its "
    "/kaggle/input/<dataset>/open-eqa-hm3d-frames-v0.tgz (or extracted directory), then rerun. "
    "No simulator/depth/mesh is needed. Do not download this archive on the evaluation laptop."
)


def download(url, target, *, expected_md5=None, min_free_gb=0):
    """Stream and resume only with a matching ETag/Last-Modified and valid byte range."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if expected_md5 and file_hash(target, "md5") != expected_md5:
            raise ValueError(f"bad cached archive checksum: {target}")
        return target
    part = target.with_suffix(target.suffix + ".part")
    metadata = part.with_suffix(part.suffix + ".json")
    old = read_json(metadata) if metadata.exists() else {}
    offset = part.stat().st_size if part.exists() and old.get("url") == url else 0
    validator = old.get("validator")
    headers = {"Range": f"bytes={offset}-", "If-Range": validator} if offset and validator else {}
    if not headers:
        offset = 0
    with requests.get(url, headers=headers, stream=True, timeout=(20, 180)) as response:
        response.raise_for_status()
        if "text/html" in response.headers.get("Content-Type", ""):
            raise RuntimeError("Host returned an access/login page. " + MANUAL_ROUTE)
        if response.status_code == 206:
            if not offset or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise ValueError("server returned an invalid resume range")
        else:
            offset = 0
        remaining = int(response.headers.get("Content-Length", 0))
        if shutil.disk_usage(target.parent).free < max(remaining, min_free_gb * 1024**3):
            raise OSError("Insufficient disk for download; archive plus extraction need temporary space")
        atomic_json(
            metadata,
            {"url": url, "validator": response.headers.get("ETag") or response.headers.get("Last-Modified")},
        )
        with (
            part.open("ab" if offset else "wb") as output,
            tqdm(
                total=remaining + offset or None,
                initial=offset,
                unit="B",
                unit_scale=True,
                desc=target.name,
            ) as progress,
        ):
            written = 0
            for block in response.iter_content(1024**2):
                if shutil.disk_usage(target.parent).free < len(block) + 128 * 1024**2:
                    raise OSError("Download stopped before exhausting disk; partial file retained")
                output.write(block)
                written += len(block)
                progress.update(len(block))
            if remaining and written != remaining:
                raise ValueError("truncated download; rerun to resume")
    if expected_md5 and file_hash(part, "md5") != expected_md5:
        raise ValueError("archive MD5 differs from official instructions; delete .part and retry")
    part.replace(target)
    metadata.unlink(missing_ok=True)
    return target


def metadata_first(config, work):
    work = Path(work)
    revision = config["dataset"]["repository_revision"]
    base = f"https://raw.githubusercontent.com/facebookresearch/open-eqa/{revision}/data/"
    qa = download(base + "open-eqa-v0.json", work / "source/open-eqa-v0.json")
    rows = read_json(qa)
    if (
        not isinstance(rows, list)
        or not rows
        or not all(
            {"question_id", "episode_history", "question", "answer", "category"} <= set(r) for r in rows
        )
    ):
        raise ValueError("Unexpected official QA metadata schema")
    readme = download(base + "README.md", work / "source/data-README.md").read_text()
    section = readme.split("#### Option 1:", 1)[1].split("#### Option 2:", 1)[0]
    match = re.search(r"\[third party location\]\((https://[^)]+)\)", section)
    checksum = re.search(r"#\s*([a-f0-9]{32})", section)
    if not match or not checksum:
        raise RuntimeError("Could not resolve documented prepared RGB archive. " + MANUAL_ROUTE)
    parsed = urlsplit(match.group(1))
    query = parse_qs(parsed.query)
    query["dl"] = ["1"]
    url = urlunsplit(parsed._replace(query=urlencode(query, doseq=True)))
    return rows, {
        "url": url,
        "md5": checksum.group(1),
        "repository_revision": revision,
        "qa_sha256": file_hash(qa),
        "readme_sha256": digest(readme.encode()),
    }


def select_environments(rows, config, *, inventory=None):
    ds = config["dataset"]
    if ds.get("environment_selection") == "smallest":
        from .episode_selection import select_shortest_episode

        if ds["environments"] != 1:
            raise ValueError("smallest selection operates on exactly one episode")
        if inventory is None:
            raise ValueError("inventory episode frame counts before selecting the shortest episode")
        return [select_shortest_episode(inventory)]
    available = sorted({r["episode_history"] for r in rows if r["episode_history"].startswith("hm3d-v0/")})
    requested = ds.get("environment_ids", [])
    if requested:
        if len(set(requested)) != len(requested) or not set(requested) <= set(available):
            raise ValueError("environment_ids must be unique HM3D episode_history values from metadata")
        return list(requested)
    n = ds["environments"]
    if not 2 <= n <= len(available):
        raise ValueError("select at least one development and one evaluation environment")
    return list(np.random.default_rng(config["seed"]).choice(available, n, replace=False))


def locate_rgb_source(config, work, archive_info):
    """Resolve attached/downloaded data before inventory; no image extraction here."""
    ds, work = config["dataset"], Path(work)
    attached = ds.get("input_path")
    if attached:
        source = Path(attached)
        if not source.exists():
            raise FileNotFoundError(f"Attached input missing: {source}. " + MANUAL_ROUTE)
        if source.is_dir():
            return source
    elif ds["allow_download"]:
        try:
            source = download(
                archive_info["url"],
                work / "source/open-eqa-hm3d-frames-v0.tgz",
                expected_md5=archive_info["md5"],
                min_free_gb=ds["min_free_gb"],
            )
        except (requests.RequestException, RuntimeError) as exc:
            raise RuntimeError("Prepared RGB download unavailable. " + MANUAL_ROUTE) from exc
    else:
        raise RuntimeError("Set input_path or explicitly enable Kaggle download. " + MANUAL_ROUTE)
    if source.suffix in {".tgz", ".gz"} and file_hash(source, "md5") != archive_info["md5"]:
        raise ValueError("attached RGB archive does not match documented MD5")
    return source


def acquire_rgb(config, work, archive_info, environments, *, source=None):
    """Extract only chosen episodes' RGB images after the collection has been inventoried."""
    work = Path(work)
    source = locate_rgb_source(config, work, archive_info) if source is None else Path(source)
    if source.is_dir():
        return source
    dest = work / "rgb"
    marker = dest / "extraction-complete.json"
    key = digest({"archive": file_hash(source), "environments": environments})
    if marker.exists() and read_json(marker).get("key") == key:
        return dest

    def allowed(name):
        return any(env + "/" in "/" + name for env in environments) and bool(
            re.search(r"-rgb\.(png|jpg|jpeg)$", name, re.I)
        )

    safe_extract(source, dest, allowed=allowed)
    atomic_json(marker, {"key": key})
    return dest


def natural_key(path):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", str(path))]


def discover_rgb(root, environments):
    result = {e: [] for e in environments}
    for p in Path(root).rglob("*"):
        if p.is_file() and re.search(r"-rgb\.(png|jpg|jpeg)$", p.name, re.I):
            name = "/" + p.as_posix()
            for env in environments:
                if env + "/" in name:
                    result[env].append(p)
    for env, paths in result.items():
        paths.sort(key=natural_key)
        if not paths:
            raise ValueError(f"No documented *-rgb images for {env}; check attached dataset root")
    return result


def dhash(image):
    pixels = np.asarray(image.convert("L").resize((9, 8)))
    bits = (pixels[:, 1:] > pixels[:, :-1]).reshape(-1)
    return f"{int(''.join(str(int(b)) for b in bits), 2):016x}"


def inspect_images(paths_by_env):
    rows = []
    for env, paths in paths_by_env.items():
        hashes, perceptual, resolutions, invalid = [], [], Counter(), []
        for path in tqdm(paths, desc=f"Inspect {env}"):
            try:
                with Image.open(path) as image:
                    image.load()
                    resolutions[str(image.size)] += 1
                    perceptual.append(int(dhash(image), 16))
                hashes.append(file_hash(path))
            except (OSError, ValueError) as exc:
                invalid.append({"file": path.name, "error": type(exc).__name__})
        # Exact pairwise 64-bit dHash distance <=4; manageable episode histories only.
        near_pairs = sum(
            (a ^ b).bit_count() <= 4 for i, a in enumerate(perceptual) for b in perceptual[i + 1 :]
        )
        pairs = len(perceptual) * (len(perceptual) - 1) // 2
        rows.append(
            {
                "environment_id": env,
                "frames": len(paths),
                "valid": len(hashes),
                "invalid": invalid,
                "resolutions": dict(resolutions),
                "exact_duplicate_fraction": (1 - len(set(hashes)) / len(hashes)) if hashes else None,
                "near_duplicate_pair_fraction": near_pairs / pairs if pairs else 0,
            }
        )
    return rows


def prepare_gallery(paths_by_env, qa_rows, environments, config, bundle):
    from .episode_selection import disjoint_window_sample

    bundle = Path(bundle)
    ds = config["dataset"]
    single_episode = ds.get("environment_selection") == "smallest"
    if single_episode and len(environments) != 1:
        raise ValueError("single-episode mode requires exactly one selected environment")
    if single_episode and ds.get("frame_sampling") != "disjoint_windows":
        raise ValueError("single-episode mode requires disjoint_windows sampling")
    plans = []
    sampling = []
    for env in environments:
        paths = paths_by_env[env]
        if ds.get("frame_sampling", "linspace") == "disjoint_windows":
            windows = disjoint_window_sample(len(paths), ds["frames_per_environment"])
            indexes = [w["selected_index"] for w in windows]
            sampling.extend(
                {
                    "environment_id": env,
                    **w,
                    "frame_id": digest({"episode": env, "source_name": paths[w["selected_index"]].name})[:32],
                    "source_filename": paths[w["selected_index"]].name,
                }
                for w in windows
            )
        else:
            n = min(ds["frames_per_environment"], len(paths))
            if n < 1:
                raise ValueError("frames_per_environment must be positive")
            indexes = np.linspace(0, len(paths) - 1, n, dtype=int).tolist()
        plans.append((env, indexes))
    expected_ids = [
        digest({"episode": env, "source_name": paths_by_env[env][idx].name})[:32]
        for env, indexes in plans
        for idx in indexes
    ]
    if (bundle / "frames.json").exists() and [
        f["frame_id"] for f in read_json(bundle / "frames.json")
    ] != expected_ids:
        raise ValueError("Existing bundle has a different episode/gallery; use a fresh WORK directory")
    frames = []
    for env, indexes in plans:
        paths = paths_by_env[env]
        for idx in indexes:
            path = paths[int(idx)]
            frame_id = digest({"episode": env, "source_name": path.name})[:32]
            relpath = f"images/{frame_id}.jpg"
            output = bundle / relpath
            output.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((ds["max_image_side"], ds["max_image_side"]), Image.Resampling.LANCZOS)
                image.save(output, quality=ds["jpeg_quality"], optimize=True)
            frames.append(
                Frame(
                    environment_id=env,
                    episode_history=env,
                    frame_id=frame_id,
                    sequence_index=int(idx),
                    image_relpath=relpath,
                    width=image.width,
                    height=image.height,
                    source_sha256=file_hash(path),
                    image_sha256=file_hash(output),
                    dhash=dhash(image),
                    split="evaluation"
                    if single_episode
                    else "development"
                    if env == environments[0]
                    else "evaluation",
                )
            )
    questions, answers = [], []
    for row in qa_rows:
        env = row["episode_history"]
        if env not in environments:
            continue
        questions.append(
            Question(
                query_id=row["question_id"],
                question=row["question"],
                environment_id=env,
                episode_history=env,
                category=row["category"],
                split="evaluation"
                if single_episode
                else "development"
                if env == environments[0]
                else "evaluation",
            )
        )
        answers.append(
            {
                "query_id": row["question_id"],
                "reference_answer": row["answer"],
                "extra_answers": row.get("extra_answers", []),
            }
        )
    atomic_json(bundle / "frames.json", [f.model_dump() for f in frames])
    atomic_json(bundle / "questions.json", [q.model_dump() for q in questions])
    atomic_json(bundle / "annotation/reference_answers.json", answers)
    atomic_json(
        bundle / "selection.json",
        {
            "config": portable_config(config),
            "environment_order": environments,
            "development_environment": None if single_episode else environments[0],
            "evaluation_scope": "single_episode_exploratory" if single_episode else "held_out_environments",
            "policy": "lowest positive collected RGB frame count; ties by episode_history"
            if single_episode
            else "seeded metadata environment sample",
            "frame_sampling": ds.get("frame_sampling", "linspace"),
            "time_axis": "sequence_index (RGB archive has no verified timestamps)",
            "original_frame_counts": {env: len(paths_by_env[env]) for env in environments},
            "pilot_frame_ids": [f.frame_id for f in frames[:5]]
            if single_episode
            else [f.frame_id for f in frames if f.split == "development"][:5],
            "preprocessing": {
                "exif_transpose": True,
                "resize": "aspect-preserving LANCZOS thumbnail",
                "max_side": ds["max_image_side"],
                "jpeg_quality": ds["jpeg_quality"],
            },
        },
    )
    if sampling:
        atomic_json(bundle / "sampling_windows.json", sampling)
    return frames, questions


def contact_sheet(frames, root, *, columns=4, thumb=(240, 160), title=""):
    from PIL import ImageDraw

    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb[0], max(1, rows) * (thumb[1] + 42) + 30), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title[:150], fill="black")
    for i, frame in enumerate(frames):
        x, y = i % columns * thumb[0], i // columns * (thumb[1] + 42) + 30
        with Image.open(Path(root) / frame.image_relpath) as image:
            image.thumbnail(thumb)
            sheet.paste(image, (x, y))
        draw.text((x + 3, y + thumb[1]), f"{frame.frame_id[:16]}\nseq {frame.sequence_index}", fill="black")
    return sheet
