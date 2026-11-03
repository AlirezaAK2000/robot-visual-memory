import hashlib
import io
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from rvm.episode_selection import inventory_episodes
from rvm.fixture import create_episode_collection
from rvm.io import load_config, read_json
from rvm.mirror import collect_hm3d_mirror
from rvm.preparation import locate_rgb_source, select_environments

ROOT = Path(__file__).resolve().parents[1]


def mock_metadata(monkeypatch, config, items):
    payload = {"sha": config["dataset"]["mirror_revision"], "siblings": items}
    monkeypatch.setattr(
        "rvm.mirror.requests.get",
        lambda *a, **kw: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: payload,
        ),
    )


def archive_item(path, filename):
    return {
        "rfilename": filename,
        "size": path.stat().st_size,
        "lfs": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
    }


@pytest.mark.parametrize("layout", ["short", "full"])
def test_collect_only_hm3d_inventory_and_per_episode_resume(tmp_path, monkeypatch, layout):
    config = load_config(ROOT / "config.example.yaml")
    raw, qa = create_episode_collection(tmp_path / "raw")
    sources, items = {}, []
    for directory in sorted((raw / "hm3d-v0").iterdir()):
        filename = f"hm3d-v0/{directory.name}.tar"
        archive = tmp_path / (directory.name + ".tar")
        with tarfile.open(archive, "w") as tar:
            for image in sorted(directory.glob("*.png")):
                tar.add(
                    image,
                    arcname=(
                        f"{directory.name}/{image.name}"
                        if layout == "short"
                        else f"hm3d-v0/{directory.name}/{image.name}"
                    ),
                )
        sources[filename] = archive
        items.append(archive_item(archive, filename))
    # A different subset must never be downloaded.
    items.append({"rfilename": "scannet-v0/unused.tar", "size": 10**9})
    mock_metadata(monkeypatch, config, items)
    downloads = []

    def download(url, target):
        relative = url.split(config["dataset"]["mirror_revision"] + "/", 1)[1]
        downloads.append(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sources[relative], target)
        return target

    monkeypatch.setattr("rvm.preparation.download", download)
    work = tmp_path / "work"
    source, provenance = collect_hm3d_mirror(config, work, qa)
    inventory, summary = inventory_episodes(source, qa, config)
    assert sorted(row["frame_count"] for row in inventory) == [5, 7, 9]
    assert summary["all_metadata_episodes_present"]
    selected = select_environments(qa, config, inventory=inventory)
    assert selected == ["hm3d-v0/001-synthetic-short"]
    assert len(downloads) == 3 and not list((source.parent / "archives").rglob("*.tar"))
    assert provenance["original_dropbox_archive_byte_identity"].startswith("unverified")
    assert provenance["missing_metadata_episodes"] == []
    assert collect_hm3d_mirror(config, work, qa)[0] == source and len(downloads) == 3
    # A missing output invalidates just that episode's completion checkpoint.
    next((source / selected[0]).glob("*-rgb.png")).unlink()
    collect_hm3d_mirror(config, work, qa)
    assert len(downloads) == 4 and downloads[-1] == selected[0] + ".tar"
    assert len(list((source / selected[0]).glob("*-rgb.png"))) == 5
    assert len(read_json(source.parent / "source.json")["archives"]) == 3


@pytest.mark.parametrize("fault", ["sha256", "traversal", "link"])
def test_reject_bad_mirror_archive_before_completion(tmp_path, monkeypatch, fault):
    config = load_config(ROOT / "config.example.yaml")
    episode = "hm3d-v0/synthetic-episode"
    archive = tmp_path / "upstream.tar"
    with tarfile.open(archive, "w") as tar:
        member = tarfile.TarInfo(
            "../escape-rgb.png" if fault == "traversal" else "synthetic-episode/00000-rgb.png"
        )
        if fault == "link":
            member.type = tarfile.SYMTYPE
            member.linkname = "/tmp/target"
            tar.addfile(member)
        else:
            member.size = 1
            tar.addfile(member, io.BytesIO(b"x"))
    item = archive_item(archive, episode + ".tar")
    if fault == "sha256":
        item["lfs"]["sha256"] = "0" * 64
    mock_metadata(monkeypatch, config, [item])

    def download(url, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive, target)
        return target

    monkeypatch.setattr("rvm.preparation.download", download)
    with pytest.raises(ValueError):
        collect_hm3d_mirror(config, tmp_path / "work", [{"episode_history": episode}])
    assert not list((tmp_path / "work").rglob("completed/*.json"))
    assert not (tmp_path / "work/source/escape-rgb.png").exists()


def test_mirror_dispatch_keeps_actual_source_provenance(tmp_path, monkeypatch):
    config = load_config(ROOT / "config.example.yaml")
    config["dataset"]["allow_download"] = True
    source = tmp_path / "frames"
    source.mkdir()
    qa = [{"episode_history": "hm3d-v0/synthetic"}]
    provenance = {"provider": "synthetic-mirror-test"}

    def collect(settings, work, rows):
        assert rows == qa
        return source, provenance

    monkeypatch.setattr("rvm.mirror.collect_hm3d_mirror", collect)
    info = {"url": "https://expired.test/archive", "md5": "original-reference-only"}
    assert locate_rgb_source(config, tmp_path, info, qa_rows=qa) == source
    assert info["rgb_acquisition"] == provenance
