import math

import pytest

from rvm.bundle import validate_bundle
from rvm.evaluation import evaluate, ranking_metrics, report_results
from rvm.schemas import Hit, Judgment, RetrievalResult


def test_known_rankings():
    m = ranking_metrics(["b", "a", "x", "c"], {"a", "c"}, 3)
    assert m["precision"] == pytest.approx(1 / 3)
    assert m["recall"] == 0.5
    assert m["mrr"] == 0.5
    assert m["ndcg"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))
    assert ranking_metrics(["a"], {"a"}, 5)["precision"] == 0.2
    assert ranking_metrics([], {"a"}, 3) == {"precision": 0, "recall": 0, "mrr": 0, "ndcg": 0}
    assert ranking_metrics([], set(), 3) == {"precision": 0, "recall": None, "mrr": None, "ndcg": None}
    assert ranking_metrics(["x", "a"], {"a"}, 1)["mrr"] == 0.5  # full-ranking MRR, not truncated at k
    with pytest.raises(ValueError):
        ranking_metrics(["a", "a"], {"a"}, 3)


class Replay:
    def __init__(self, ids, error=None):
        self.ids, self.error = ids, error
        self.calls = []

    def retrieve(self, question, environment_id, k):
        self.calls.append((question, environment_id, k))
        return RetrievalResult(
            hits=[Hit(frame_id=fid, score=1) for fid in self.ids[:k]],
            diagnostics={"error": self.error, "total_seconds": 0.01, "database_seconds": 0.002},
        )


def test_evaluation_incomplete_errors_and_reports(fixture_bundle, tmp_path):
    bundle = validate_bundle(fixture_bundle)
    human = [
        Judgment(
            query_id=j.query_id,
            frame_id=j.frame_id,
            relevance=j.relevance,
            source="human",
            status="confirmed",
        )
        for j in bundle["judgments"]
    ]
    human[0] = Judgment(
        query_id=human[0].query_id,
        frame_id=human[0].frame_id,
        relevance=None,
        source="unjudged",
        status="unjudged",
    )
    systems = {"graph": Replay([], "ServiceUnavailable"), "caption_embedding": Replay(["synthetic-frame-1"])}
    table, runs, status = evaluate(systems, bundle["questions"], bundle["frames"], human)
    pending = table[table.query_id == "synthetic-q1"]
    assert pending[["precision", "recall", "mrr", "ndcg"]].isna().all().all()
    assert table[table.query_id == "synthetic-q0"].recall.isna().all()
    assert len(systems["graph"].calls) == 6
    assert all(len(call) == 3 for call in systems["graph"].calls)
    agg, *_ = report_results(table, runs, status, tmp_path / "reports", provenance={"synthetic_test": True})
    assert agg.empty
    # Complete qrels: failed graph counts as zero, not a dropped query.
    human[0] = Judgment(
        query_id=human[0].query_id,
        frame_id=human[0].frame_id,
        relevance=1,
        source="human",
        status="confirmed",
    )
    complete, _, _ = evaluate(systems, bundle["questions"], bundle["frames"], human)
    row = complete[(complete.query_id == "synthetic-q1") & (complete.method == "graph")].iloc[0]
    assert row.precision == 0 and row.recall == 0 and row.mrr == 0


@pytest.mark.parametrize("ids", [["synthetic-frame-3"], ["missing"], ["synthetic-frame-1"] * 2])
def test_wrong_environment_unknown_duplicate_retrieval_fatal(fixture_bundle, ids):
    b = validate_bundle(fixture_bundle)
    with pytest.raises(ValueError):
        evaluate(
            {"graph": Replay(ids), "caption_embedding": Replay([])},
            b["questions"],
            b["frames"],
            b["judgments"],
            synthetic=True,
        )


def test_exactly_two_methods(fixture_bundle):
    b = validate_bundle(fixture_bundle)
    with pytest.raises(ValueError, match="exactly"):
        evaluate({"uniform": Replay([])}, b["questions"], b["frames"], b["judgments"])
