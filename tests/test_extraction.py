import json
from types import SimpleNamespace

import pytest

from rvm.extraction import RequestBudget, StructuredRunner, extract_frames, usage_delta, usage_summary
from rvm.schemas import Extraction, Frame


SETTINGS = {
    "model": "fake-synthetic-only",
    "concurrency": 2,
    "max_attempts": 2,
    "max_requests": 4,
    "max_output_tokens": 100,
    "max_reserved_output_tokens": 400,
}


class FakeResponse:
    status = "completed"
    output_text = json.dumps({"caption": "A test chair.", "objects": [], "relationships": []})
    usage = SimpleNamespace(model_dump=lambda: {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30})

    def model_dump(self, **kwargs):
        return {"status": self.status, "output": self.output_text}


class FakeClient:
    def __init__(self):
        self.calls = []
        self.responses = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse()


def test_cache_reuse_and_invalidation(tmp_path):
    client = FakeClient()
    runner = StructuredRunner(tmp_path, SETTINGS, client=client)
    kwargs = {
        "prompt": "p1",
        "content": [{"type": "input_text", "text": "synthetic"}],
        "schema": Extraction,
        "identity": {"image_hash": "abc", "preprocessing": {"side": 1024}},
    }
    one = runner.run(**kwargs)
    before = usage_summary(tmp_path)
    two = runner.run(**kwargs)
    assert usage_delta(before, usage_summary(tmp_path))["actual_attempts"] == 0
    assert len(client.calls) == 1 and two[2]["cache_hit"]
    assert one[1] == two[1]
    runner.run(**{**kwargs, "prompt": "p2"})
    assert len(client.calls) == 2
    runner.run(**{**kwargs, "identity": {"image_hash": "changed"}})
    assert len(client.calls) == 3
    assert client.calls[0]["text"]["format"]["strict"] is True
    assert "answer" not in json.dumps(client.calls[0])
    assert usage_summary(tmp_path)["tokens"]["total_tokens"] == 90
    assert usage_summary(tmp_path)["currency_cost"] is None


def test_persistent_budget_and_errors(tmp_path):
    budget = RequestBudget(tmp_path, {**SETTINGS, "max_requests": 1})
    budget.reserve()
    with pytest.raises(RuntimeError, match="budget"):
        RequestBudget(tmp_path, {**SETTINGS, "max_requests": 1}).reserve()


def test_failed_validation_raw_response_preserved(tmp_path):
    client = FakeClient()

    class Invalid(FakeResponse):
        output_text = '{"caption":"no graph fields"}'

    client.create = lambda **kw: Invalid()
    runner = StructuredRunner(tmp_path, {**SETTINGS, "max_attempts": 1}, client=client)
    with pytest.raises(RuntimeError, match="capped"):
        runner.run(prompt="test", content=[], schema=Extraction, identity={})
    from rvm.io import read_json

    event = read_json(next((tmp_path / "attempts").glob("*.json")))
    assert event["error"] == "ValidationError" and event["raw_response"] is not None


def test_frame_progress_tracks_completions_without_reordering_results(fixture_bundle, monkeypatch):
    import threading
    from rvm.io import read_json

    frames = [Frame.model_validate(f) for f in read_json(fixture_bundle / "frames.json")][:3]
    first_released = threading.Event()
    updates, counters = [], []

    class Progress:
        def __init__(self, **kwargs):
            assert kwargs["total"] == 3 and not kwargs["disable"]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, count):
            updates.append(count)
            # A later frame must advance the UI while the first is still waiting.
            first_released.set()

        def set_postfix(self, **kwargs):
            counters.append(kwargs)

    class Runner:
        settings = {"concurrency": 2}

        def run(self, *, identity, **kwargs):
            index = next(i for i, f in enumerate(frames) if f.image_sha256 == identity["image_sha256"])
            if index == 0:
                assert first_released.wait(timeout=5), (
                    "Progress waited for manifest order instead of completion"
                )
            result = Extraction(caption=f"Synthetic frame {index}", objects=[], relationships=[])
            return (
                result,
                f"key-{index}",
                {"cache_hit": index == 1, "attempts": [{}] * (2 if index == 0 else 1)},
            )

    monkeypatch.setattr("rvm.extraction.tqdm", Progress)
    records = extract_frames(frames, fixture_bundle, Runner(), preprocessing={}, show_progress=True)
    assert [r.frame_id for r in records] == [f.frame_id for f in frames]
    assert [r.extraction.caption for r in records] == [f"Synthetic frame {i}" for i in range(3)]
    assert updates == [1, 1, 1]
    assert counters[-1]["cached"] == 1 and counters[-1]["api_attempts"] == 3


def test_failed_frame_does_not_count_as_completed_progress(fixture_bundle, monkeypatch):
    from rvm.io import read_json

    frames = [Frame.model_validate(f) for f in read_json(fixture_bundle / "frames.json")][:1]
    updates = []

    class Runner:
        settings = {"concurrency": 1}

        def run(self, **kwargs):
            raise RuntimeError("synthetic failure")

    class Progress:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, count):
            updates.append(count)

    monkeypatch.setattr("rvm.extraction.tqdm", Progress)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        extract_frames(frames, fixture_bundle, Runner(), preprocessing={})
    assert not updates
