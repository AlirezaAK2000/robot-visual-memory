import io
import tarfile
import zipfile

import numpy as np
import pytest
from pydantic import ValidationError

from rvm.annotation import annotation_status, load_qrels, save_qrels, validate_qrels
from rvm.bundle import validate_bundle
from rvm.io import atomic_json, export_zip, portable, read_json, safe_extract, write_checksums
from rvm.schemas import Extraction, Judgment


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


@pytest.mark.parametrize(
    "fault", ["missing-image", "vector-id", "nan", "dimension", "duplicate-record", "caption-source"]
)
def test_bad_bundle_rejected(fixture_bundle, fault):
    if fault == "missing-image":
        next((fixture_bundle / "images").glob("*.jpg")).unlink()
    elif fault == "vector-id":
        mapping = read_json(fixture_bundle / "vector_frame_ids.json")
        mapping.reverse()
        atomic_json(fixture_bundle / "vector_frame_ids.json", mapping)
    elif fault in {"nan", "dimension"}:
        vectors = np.load(fixture_bundle / "caption_vectors.npy")
        if fault == "nan":
            vectors[0, 0] = np.nan
        else:
            vectors = vectors[:, :2]
        np.save(fixture_bundle / "caption_vectors.npy", vectors)
    elif fault == "caption-source":
        records = read_json(fixture_bundle / "records.json")
        records[0]["extraction"]["caption"] = "Changed after vectors were computed"
        atomic_json(fixture_bundle / "records.json", records)
    else:
        records = read_json(fixture_bundle / "records.json")
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


def test_unjudged_silver_and_all_zero_coverage(fixture_bundle):
    bundle = validate_bundle(fixture_bundle)
    frames, questions = bundle["frames"], bundle["questions"]
    status = annotation_status(bundle["judgments"], frames, questions, synthetic=True)
    assert [s["coverage"] for s in status] == ["has-evidence", "no-relevant"]
    # Synthetic judgments must never satisfy a real human evaluation.
    assert not any(s["complete"] for s in annotation_status(bundle["judgments"], frames, questions))
    judgments = [
        Judgment(
            query_id=j.query_id,
            frame_id=j.frame_id,
            relevance=j.relevance,
            source="human",
            status="confirmed",
        )
        for j in bundle["judgments"]
    ]
    judgments[0] = judgments[0].model_copy(
        update={"relevance": None, "source": "unjudged", "status": "unjudged"}
    )
    status = annotation_status(judgments, frames, questions)
    assert not status[0]["complete"] and status[1]["complete"]
    judgments[0] = judgments[0].model_copy(update={"relevance": 1, "source": "silver", "status": "proposed"})
    assert not annotation_status(judgments, frames, questions)[0]["complete"]


def test_wrong_environment_qrel_and_resumable_csv(fixture_bundle):
    bundle = validate_bundle(fixture_bundle)
    path = fixture_bundle / "annotation/qrels.csv"
    judgments = load_qrels(path)
    judgments[0].notes = "checked, useful partial evidence\nsecond line"
    save_qrels(path, judgments)
    assert load_qrels(path)[0].notes == judgments[0].notes
    judgments[0] = judgments[0].model_copy(update={"frame_id": "synthetic-frame-3"})
    with pytest.raises(ValueError, match="environment"):
        validate_qrels(judgments, bundle["frames"], bundle["questions"])
