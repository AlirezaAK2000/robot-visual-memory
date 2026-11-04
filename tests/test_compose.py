from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_compose_has_only_manually_started_neo4j():
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text())
    assert set(compose["services"]) == {"neo4j"}
    neo4j = compose["services"]["neo4j"]
    assert neo4j["image"].startswith("neo4j:5.26.12-community@sha256:")
    assert neo4j["restart"] == "no"
    assert all(port.startswith("127.0.0.1:") for port in neo4j["ports"])
    assert "healthcheck" in neo4j
