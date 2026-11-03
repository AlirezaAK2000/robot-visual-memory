from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_compose_scopes_credentials_and_official_mcp_lifecycle():
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text())
    services = compose["services"]
    assert set(services) == {"neo4j", "qdrant", "neo4j-mcp"}
    assert services["neo4j"]["image"].startswith("neo4j:5.26.12-community@sha256:")
    assert services["neo4j-mcp"]["image"].startswith("neo4j/mcp:v2.0.0@sha256:")
    for name in ["neo4j", "qdrant"]:
        assert all(port.startswith("127.0.0.1:") for port in services[name]["ports"])
        assert "healthcheck" in services[name]
    mcp = services["neo4j-mcp"]
    assert mcp["profiles"] == ["mcp"] and mcp["stdin_open"] and not mcp["tty"]
    assert mcp["environment"]["NEO4J_MCP_READ_ONLY"] == "true"
    assert mcp["environment"]["NEO4J_MCP_TRANSPORT_MODE"] == "stdio"
    assert mcp["environment"]["NEO4J_MCP_URI"] == "bolt://neo4j:7687"
    assert "ports" not in mcp
