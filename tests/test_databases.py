"""Neo4j import integrity and graph-query checks; no hosted model calls."""

import os
import pytest
from rvm.bundle import validate_bundle
from rvm.retrieval import GraphRetriever
from rvm.planning import RulesPlanner


@pytest.mark.integration
def test_repeat_ingestion_and_graph_constraints(fixture_bundle):
    if os.getenv("RVM_RUN_DB_TESTS") != "1":
        pytest.skip("Set RVM_RUN_DB_TESTS=1 after docker compose up -d --wait")
    import uuid
    from rvm.ingestion import clients, ingest, verify_import

    collection = "synthetic_test_" + uuid.uuid4().hex
    driver = clients()
    ids = [f.frame_id for f in validate_bundle(fixture_bundle)["frames"]]
    with driver.session() as session:
        existing = session.run(
            "MATCH (f:Frame) WHERE f.frame_id IN $ids RETURN count(f) AS n", ids=ids
        ).single()["n"]
    if existing:
        driver.close()
        pytest.skip("Preexisting synthetic fixture frames found; use an isolated test database")
    try:
        one = ingest(fixture_bundle, driver, collection)
        two = ingest(fixture_bundle, driver, collection)
        assert one == two
        assert one["frames"] == 4 and one["objects"] == 8 and one["relationships"] == 4
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
            verify_import(fixture_bundle, driver, collection)
    finally:
        # Only this explicitly synthetic fixture's nodes are removed, never saved user data.
        with driver.session() as session:
            session.run(
                "MATCH (o:ObjectObservation) WHERE o.frame_id IN $ids DETACH DELETE o", ids=ids
            ).consume()
            session.run("MATCH (f:Frame) WHERE f.frame_id IN $ids DETACH DELETE f", ids=ids).consume()
            session.run(
                "MATCH (e:Environment) WHERE e.environment_id STARTS WITH 'synthetic/' AND NOT (e)--() DELETE e"
            ).consume()
            session.run("MATCH (g:Gallery {collection:$c}) DETACH DELETE g", c=collection).consume()
        driver.close()


@pytest.mark.real
def test_real_compact_bundle_smoke():
    path = os.getenv("RVM_REAL_BUNDLE")
    if not path:
        pytest.skip("Set RVM_REAL_BUNDLE to a real Kaggle compact export; no paid API requests are made")
    b = validate_bundle(path)
    assert b["manifest"]["synthetic"] is False
    assert len(b["frames"]) >= 5
    assert all(f.episode_history.startswith("hm3d-v0/") for f in b["frames"])
