import io
import tarfile
import zipfile

import pytest
from pydantic import ValidationError

from rvm.bundle import validate_bundle
from rvm.io import atomic_json, export_zip, portable, read_json, safe_extract, write_checksums
from rvm.schemas import Extraction


def graph():
    return {
        "caption": "A white kettle on a table.",
        "objects": [
            {"object_id": "1", "label": "kettle", "attributes": [], "states": []},
            {"object_id": "2", "label": "table", "attributes": [], "states": []},
        ],
        "relationships": [{"source": "1", "predicate": "on", "target": "2"}],
    }


@pytest.mark.parametrize("fault", ["dangling", "duplicate", "predicate", "missing", "self", "cross-frame"])
def test_graph_rejects_bad_extraction(fault):
    g = graph()
    if fault in {"dangling", "cross-frame"}:
        g["relationships"][0]["target"] = "other-frame:2"
    elif fault == "duplicate":
        g["objects"][1]["object_id"] = "1"
    elif fault == "predicate":
        g["relationships"][0]["predicate"] = "teleports_to"
    elif fault == "missing":
        del g["caption"]
    elif fault == "self":
        g["relationships"][0]["target"] = "1"
    with pytest.raises(ValidationError):
        Extraction.model_validate(g)


def test_export_roundtrip_images_and_relative_paths(fixture_bundle, tmp_path):
    archive = export_zip(fixture_bundle, tmp_path / "compact.zip")
    fresh = safe_extract(archive, tmp_path / "fresh")
    bundle = validate_bundle(fresh)
    assert len(bundle["frames"]) == 4
    assert bundle["manifest"]["synthetic"] is True
    for frame in bundle["frames"]:
        assert portable(fresh, frame.image_relpath).exists()
        assert not frame.image_relpath.startswith("/")


@pytest.mark.parametrize("fault", ["missing-image", "duplicate-record", "caption-source"])
def test_bad_bundle_rejected(fixture_bundle, fault):
    if fault == "missing-image":
        next((fixture_bundle / "images").glob("*.jpg")).unlink()
    else:
        records = read_json(fixture_bundle / "records.json")
        if fault == "caption-source":
            records[0]["extraction"]["caption"] = "Changed without sealing a new content contract"
        else:
            records.append(records[0])
        atomic_json(fixture_bundle / "records.json", records)
    write_checksums(fixture_bundle)
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_bundle(fixture_bundle)


@pytest.mark.parametrize("name", ["../escaped", "/tmp/escaped", "a/../../escaped", "a\\evil"])
def test_archive_traversal(tmp_path, name):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(name, b"x")
    with pytest.raises(ValueError):
        safe_extract(archive, tmp_path / "out")


def test_archive_symlink_and_expansion(tmp_path):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("symlink")
        info.type = tarfile.SYMTYPE
        info.linkname = "/tmp/escaped"
        tar.addfile(info)
    with pytest.raises(ValueError):
        safe_extract(archive, tmp_path / "out")
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("file")
        info.size = 10
        tar.addfile(info, io.BytesIO(b"x" * 10))
    with pytest.raises(OSError):
        safe_extract(archive, tmp_path / "out", max_bytes=5)
