"""Offline tests use Qdrant's explicit in-process mode; production always uses HTTP Qdrant."""

import os

import numpy as np
import pytest

from rvm.bundle import validate_bundle
from rvm.ingestion import point_id
from rvm.retrieval import CaptionRetriever, GraphRetriever
from rvm.planning import RulesPlanner


class SyntheticEmbedder:
    def encode(self, texts, *, query=False):
        assert query is True
        return np.array([[0.8, 0.6, 0]], dtype=np.float32)


def test_qdrant_real_client_alignment_filter_and_bruteforce(fixture_bundle):
    from qdrant_client import QdrantClient, models as qm

    client = QdrantClient(":memory:")
    b = validate_bundle(fixture_bundle)
    client.create_collection("synthetic", vectors_config=qm.VectorParams(size=3, distance=qm.Distance.COSINE))
    points = [
        qm.PointStruct(
            id=point_id(f.frame_id),
            vector=b["vectors"][i].tolist(),
            payload={"frame_id": f.frame_id, "environment_id": f.environment_id},
        )
        for i, f in enumerate(b["frames"])
    ]
    client.upsert("synthetic", points=points, wait=True)
    client.upsert("synthetic", points=points, wait=True)
    assert client.count("synthetic", exact=True).count == 4
    retriever = CaptionRetriever(client, SyntheticEmbedder(), "synthetic")
    result = retriever.retrieve("synthetic question", "synthetic/test", 5)
    assert result.diagnostics["error"] is None
    assert [h.frame_id for h in result.hits] == ["synthetic-frame-1", "synthetic-frame-2"]
    candidates = [i for i, f in enumerate(b["frames"]) if f.environment_id == "synthetic/test"]
    brute = sorted(
        candidates, key=lambda i: (-float(np.dot(b["vectors"][i], [0.8, 0.6, 0])), b["frames"][i].frame_id)
    )
    assert [h.frame_id for h in result.hits] == [b["frames"][i].frame_id for i in brute]
    assert retriever.retrieve("synthetic", "environment-does-not-exist", 5).hits == []
    client.close()


@pytest.mark.integration
def test_repeat_ingestion_graph_constraints_and_real_qdrant(fixture_bundle):
    if os.getenv("RVM_RUN_DB_TESTS") != "1":
        pytest.skip("Set RVM_RUN_DB_TESTS=1 after docker compose up -d --wait")
    import uuid
    from rvm.ingestion import clients, ingest, verify_import

    collection = "synthetic_test_" + uuid.uuid4().hex
    driver, qdrant = clients()
    ids = [f.frame_id for f in validate_bundle(fixture_bundle)["frames"]]
    with driver.session() as session:
        existing = session.run(
            "MATCH (f:Frame) WHERE f.frame_id IN $ids RETURN count(f) AS n", ids=ids
        ).single()["n"]
    if existing:
        driver.close()
        qdrant.close()
        pytest.skip("Preexisting synthetic fixture frames found; use an isolated test database")
    try:
        one = ingest(fixture_bundle, driver, qdrant, collection)
        two = ingest(fixture_bundle, driver, qdrant, collection)
        assert {k: v for k, v in one.items() if k != "indexing_seconds"} == {
            k: v for k, v in two.items() if k != "indexing_seconds"
        }
        assert one["frames"] == one["vectors"] == 4 and one["objects"] == 8 and one["relationships"] == 4
        graph = GraphRetriever(driver, RulesPlanner(), collection)
        result = graph.retrieve("What is on the table beside the kettle?", "synthetic/test", 5)
        assert result.diagnostics["error"] is None, result.diagnostics
        assert {h.frame_id for h in result.hits} <= {"synthetic-frame-1", "synthetic-frame-2"}
        assert (
            graph.retrieve("white kettle on table", "synthetic/other", 5).hits[0].frame_id
            == "synthetic-frame-3"
        )
        assert graph.retrieve("white kettle on table", "missing", 5).hits == []
        # The same label across frames is not a relation. Corrupt one edge and require detection.
        with driver.session() as session:
            session.run("""MATCH (a:ObjectObservation {frame_id:'synthetic-frame-1',local_id:'a'}),
                                 (b:ObjectObservation {frame_id:'synthetic-frame-2',local_id:'b'})
                           CREATE (a)-[:RELATES {predicate:'near'}]->(b)""").consume()
        with pytest.raises(ValueError, match="cross-frame"):
            verify_import(fixture_bundle, driver, qdrant, collection)
    finally:
        # Only this explicitly synthetic fixture's nodes are removed, never real benchmark data.
        with driver.session() as session:
            session.run(
                "MATCH (o:ObjectObservation) WHERE o.frame_id IN $ids DETACH DELETE o", ids=ids
            ).consume()
            session.run("MATCH (f:Frame) WHERE f.frame_id IN $ids DETACH DELETE f", ids=ids).consume()
            session.run(
                "MATCH (e:Environment) WHERE e.environment_id STARTS WITH 'synthetic/' AND NOT (e)--() DELETE e"
            ).consume()
            session.run("MATCH (g:Gallery {collection:$c}) DETACH DELETE g", c=collection).consume()
        if qdrant.collection_exists(collection):
            qdrant.delete_collection(collection)
        driver.close()
        qdrant.close()


@pytest.mark.real
def test_real_compact_bundle_smoke():
    path = os.getenv("RVM_REAL_BUNDLE")
    if not path:
        pytest.skip("Set RVM_REAL_BUNDLE to a real Kaggle compact export; no paid API requests are made")
    b = validate_bundle(path)
    assert b["manifest"]["synthetic"] is False
    assert len(b["frames"]) >= 5
    assert all(f.episode_history.startswith("hm3d-v0/") for f in b["frames"])
