import base64
from pathlib import Path

import pytest

from rvm.fixture import create_single_episode_fixture
from rvm.io import read_json
from rvm.review import extraction_review_html
from rvm.schemas import Extraction, Frame, Record

ROOT = Path(__file__).resolve().parents[1]


def test_review_pairs_exact_images_with_outputs_and_escapes_model_text(tmp_path):
    root = create_single_episode_fixture(tmp_path / "bundle", config_path=ROOT / "config.example.yaml")
    frames = [Frame.model_validate(f) for f in read_json(root / "frames.json")][:3]
    extraction = Extraction.model_validate(
        {
            "caption": "A kettle <script>alert('test')</script> beside a sink.",
            "objects": [
                {
                    "object_id": "k1",
                    "label": "kettle",
                    "attributes": [{"name": "color", "value": None}],
                    "states": ["open"],
                },
                {"object_id": "s1", "label": "sink", "attributes": [], "states": []},
            ],
            "relationships": [{"source": "k1", "predicate": "left_of", "target": "s1"}],
        }
    )
    records = [Record(frame_id=f.frame_id, extraction=extraction, cache_key="test-only") for f in frames]
    html = extraction_review_html(frames, reversed(records), root, model="test-model")
    assert html.count('<article class="rvm-card">') == 3
    for f in frames:
        encoded = base64.b64encode((root / f.image_relpath).read_bytes()).decode("ascii")
        assert f"data:image/jpeg;base64,{encoded}" in html
    assert html.index(frames[0].frame_id) < html.index(frames[1].frame_id) < html.index(frames[2].frame_id)
    assert "&lt;script&gt;" in html and "<script>" not in html
    assert "kettle (k1)" in html and "left of" in html and "sink (s1)" in html
    assert "Unknown" in html and "open" in html and "None reported" in html
    assert "View structured JSON" in html and "Model: test-model" in html


def test_review_rejects_missing_outputs_and_changed_images_and_marks_synthetic(tmp_path):
    root = create_single_episode_fixture(tmp_path / "bundle", config_path=ROOT / "config.example.yaml")
    frames = [Frame.model_validate(f) for f in read_json(root / "frames.json")][:3]
    records = [Record.model_validate(r) for r in read_json(root / "records.json")][:3]
    with pytest.raises(ValueError, match="exactly one extraction"):
        extraction_review_html(frames, records[:2], root, model="test-model")
    html = extraction_review_html(frames, records, root, model="test-model", synthetic=True)
    assert "SYNTHETIC EXAMPLE" in html and "No model requests" in html
    assert "No objects reported." in html and "No relationships reported." in html
    (root / frames[0].image_relpath).write_bytes(b"changed")
    with pytest.raises(ValueError, match="image hash mismatch"):
        extraction_review_html(frames, records, root, model="test-model")


def test_custom_search_reuses_saved_evidence_and_closes_each_connection(fixture_bundle, monkeypatch):
    from unittest.mock import Mock
    from rvm.bundle import validate_bundle
    from rvm.notebook import GallerySearch
    from rvm.planning import RulesPlanner
    from rvm.schemas import Hit, RetrievalResult

    bundle = validate_bundle(fixture_bundle)
    drivers = [Mock(), Mock()]
    outputs = []
    monkeypatch.setattr("rvm.notebook.clients", Mock(side_effect=drivers))
    monkeypatch.setattr("rvm.notebook.display", outputs.append)
    retriever = Mock()
    retriever.retrieve.return_value = RetrievalResult(
        hits=[Hit(frame_id=bundle["frames"][0].frame_id, score=2, diagnostics={"matched": ["anchor:0"]})],
        diagnostics={"error": None},
    )
    monkeypatch.setattr("rvm.notebook.GraphRetriever", Mock(return_value=retriever))
    gallery = GallerySearch(
        bundle, RulesPlanner(), "test", "synthetic/dev", {"extraction": {"model": "saved"}, "retrieval": {}}
    )
    gallery.ask(["Where is the kettle?"])
    gallery.ask(["Where is the kettle?"])
    for driver in drivers:
        driver.close.assert_called_once()
    assert sum("synthetic-frame-0" in str(getattr(output, "data", "")) for output in outputs) == 2
