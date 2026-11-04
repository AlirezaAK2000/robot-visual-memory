from rvm.audit import audit_dataset
from rvm.normalization import graph_vocabulary, object_properties
from rvm.planning import RulesPlanner
from rvm.schemas import Object


def test_normalization_preserves_source_and_does_not_assert_uncertain_values():
    obj = Object.model_validate(
        {
            "object_id": "a",
            "label": "Armchair",
            "attributes": [
                {"name": "color", "value": "dark brown"},
                {"name": "material", "value": "unknown"},
                {"name": "shape", "value": None},
            ],
            "states": ["partially_visible"],
        }
    )
    props = object_properties(obj)
    assert props["raw_label"] == "Armchair" and props["label"] == "armchair"
    assert set(props["labels"]) == {"chair", "armchair"}
    assert set(props["attributes"]) == {"dark brown", "brown"}
    assert props["attribute_pairs"] == ["color=dark brown"]
    assert props["states"] == ["partially visible"]
    assert obj.attributes[1].value == "unknown"
    obj.attributes[0].value = "brown or black"
    assert object_properties(obj)["attributes"] == ["brown or black"]


def test_stored_vocabulary_multiword_attributes_and_postnominal_state():
    vocabulary = {"labels": ["support post", "railing"], "attributes": ["dark brown"], "states": ["hanging"]}
    plan, diag = RulesPlanner(vocabulary).plan("dark brown support post near railing")
    assert [a.label for a in plan.anchors] == ["support post", "railing"]
    assert plan.anchors[0].attributes == ["dark brown"]
    assert plan.relationships[0].predicate.value == "near"
    assert not diag["unsupported"]
    plan, _ = RulesPlanner().plan("Is the TV on?")
    assert plan.anchors[0].states == ["on"]
    assert not plan.relationships
    plan, _ = RulesPlanner().plan("kettle on table")
    assert not plan.anchors[0].states and not plan.anchors[1].states
    assert plan.relationships[0].predicate.value == "on"


def test_audit_reports_structure_and_vocabulary_without_labels(fixture_bundle):
    from rvm.bundle import validate_bundle

    report = audit_dataset(fixture_bundle)
    assert report["frames"] == 4 and report["objects"] == 8 and report["relationships"] == 4
    assert report["invalid_or_dangling_edges"] == 0
    assert report["normalized_empty_labels"] == 0
    assert "kettle" in report["vocabulary"]["labels"]
    b = validate_bundle(fixture_bundle)
    assert graph_vocabulary(b["records"]) == report["vocabulary"]


def test_audit_flags_conflicting_spatial_directions(fixture_bundle):
    from rvm.bundle import finalize_bundle
    from rvm.io import atomic_json, read_json

    records = read_json(fixture_bundle / "records.json")
    records[0]["extraction"]["relationships"] += [{"source": "b", "predicate": "on", "target": "a"}]
    atomic_json(fixture_bundle / "records.json", records)
    finalize_bundle(fixture_bundle, synthetic=True)
    report = audit_dataset(fixture_bundle)
    assert report["reverse_direction_conflicts"]["on"] == 1
    assert report["frames_with_predicate_cycles"]["on"] == 1
