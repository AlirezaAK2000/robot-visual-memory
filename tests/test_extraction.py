import json
from types import SimpleNamespace

import pytest

from rvm.extraction import RequestBudget, StructuredRunner, usage_delta, usage_summary
from rvm.schemas import Extraction


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
