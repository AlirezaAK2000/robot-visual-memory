from rvm.normalization import canonical
from rvm.planning import RulesPlanner, validate_grounding
from rvm.retrieval import compile_plan
from rvm.schemas import Anchor, PlanRelation, QueryPlan


def test_question_only_wildcard_plan():
    plan, diag = RulesPlanner().plan("What is beside the kettle?")
    assert [a.label for a in plan.anchors] == ["kettle"]
    assert plan.relationships[0].target is None
    assert plan.relationships[0].predicate.value == "near"
    assert not diag["unsupported"]


def test_plan_cannot_insert_expected_answer_constraints():
    import pytest

    invented = QueryPlan(anchors=[Anchor(label="kettle", attributes=["red"], states=[])], relationships=[])
    with pytest.raises(ValueError, match="unmentioned"):
        validate_grounding(invented, "What color is the kettle?")
    invented = QueryPlan(anchors=[Anchor(label="spoon", attributes=[], states=[])], relationships=[])
    with pytest.raises(ValueError, match="unmentioned"):
        validate_grounding(invented, "What is beside the kettle?")


def test_normalization_attributes_and_direction():
    plan, _ = RulesPlanner().plan("What is on the white wooden table?")
    assert plan.anchors[0].attributes == ["white", "wood"]
    assert plan.relationships[0].direction == "in"
    plan, _ = RulesPlanner().plan("Is the red kettle left of the closed cabinet?")
    assert len(plan.anchors) == 2
    assert plan.anchors[0].attributes == ["red"]
    assert plan.anchors[1].states == ["closed"]
    assert plan.relationships[0].target == 1
    assert canonical("chairs") == "chair" and canonical("television") == "tv"
    assert not RulesPlanner().plan("Which object is not visible?")[0].anchors


def test_cypher_parameterization_environment_frame_and_connected_constraints():
    malicious = "kettle'}) DETACH DELETE f //"
    plan = QueryPlan(
        anchors=[
            Anchor(label=malicious, attributes=["red"], states=["open"]),
            Anchor(label="cabinet", attributes=[], states=[]),
        ],
        relationships=[PlanRelation(source=0, target=1, predicate="inside", direction="out")],
    )
    query, parameters = compile_plan(plan, "env-A", "gallery", 5)
    assert malicious not in query
    assert parameters["anchors"][0]["label"] == malicious
    assert "environment_id:$env" in query
    assert "a.frame_id=f.frame_id" in query and "b.frame_id=f.frame_id" in query
    assert "(a)-[r:RELATES]->(b)" in query
    assert "a.attributes" in query and "a.states" in query
    assert "ORDER BY score DESC, frame_id ASC" in query
    assert all(word not in query.upper() for word in ["DELETE", "MERGE", "CREATE", "SET "])


def test_plan_graph_types_and_no_shared_id_merge():
    from rvm.ingestion import observation_id, point_id
    import uuid

    assert observation_id("e", "frame-1", "chair") != observation_id("e", "frame-2", "chair")
    assert observation_id("e1", "frame", "chair") != observation_id("e2", "frame", "chair")
    assert str(uuid.UUID(point_id("frame"))) == point_id("frame")
