"""Graph dataset preparation, import, inspection and retrieval commands."""

import argparse
import json
from pathlib import Path

from .bundle import open_bundle, validate_bundle
from .io import atomic_json, export_zip, read_json


def main():
    parser = argparse.ArgumentParser(description="Robot Visual Memory — Neo4j graph retrieval")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["import", "verify", "validate", "audit", "query"]:
        p = commands.add_parser(name)
        p.add_argument("bundle", type=Path)
        p.add_argument("--destination", type=Path)
        if name in {"import", "verify", "query"}:
            p.add_argument("--collection", default=None)
        if name == "query":
            p.add_argument("question")
            p.add_argument("--environment")
            p.add_argument("--top-k", type=int, default=5)
            p.add_argument(
                "--plan", type=Path, help="Optional explicit QueryPlan JSON for precise constraints"
            )
    p = commands.add_parser("fixture")
    p.add_argument("output", type=Path)
    p = commands.add_parser("export")
    p.add_argument("bundle", type=Path)
    p.add_argument("output", type=Path)
    for name in ["prepare-graph", "recover-longest"]:
        p = commands.add_parser(name)
        p.add_argument("backup", type=Path)
        p.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command in {"prepare-graph", "recover-longest"}:
        from .handoff import prepare_graph_bundle

        result = prepare_graph_bundle(args.backup, args.destination)
    elif args.command == "fixture":
        from .fixture import create_fixture

        result = {"bundle": str(create_fixture(args.output))}
    elif args.command == "export":
        result = {"archive": str(export_zip(args.bundle, args.output))}
    else:
        root = open_bundle(args.bundle, destination=args.destination)
        bundle = validate_bundle(root)
        if args.command == "validate":
            result = {"frames": len(bundle["frames"]), "synthetic": bundle["manifest"]["synthetic"]}
        elif args.command == "audit":
            from .audit import audit_dataset

            result = audit_dataset(root)
            atomic_json(root.parent / f"{root.name}-audit.json", result)
        else:
            from .ingestion import clients, ingest, verify_import, stored_vocabulary

            driver = clients()
            try:
                collection = args.collection or read_json(root / "selection.json").get("config", {}).get(
                    "retrieval", {}
                ).get("collection", "robot_visual_memory")
                if args.command == "query":
                    from .planning import RulesPlanner
                    from .retrieval import GraphRetriever
                    from .schemas import QueryPlan

                    environments = {f.environment_id for f in bundle["frames"]}
                    if not args.environment and len(environments) != 1:
                        parser.error("Choose --environment for a gallery with multiple environments")
                    environment = args.environment or next(iter(environments))
                    if environment not in environments:
                        parser.error("Environment is not in this gallery")
                    verify_import(root, driver, collection)
                    planner = RulesPlanner(stored_vocabulary(driver, collection))
                    if args.plan:
                        plan = QueryPlan.model_validate(read_json(args.plan))

                        class ExplicitPlanner:
                            def plan(self, question):
                                return plan, {"parser": "explicit"}

                        planner = ExplicitPlanner()
                    retrieved = GraphRetriever(driver, planner, collection).retrieve(
                        args.question, environment, args.top_k
                    )
                    result = retrieved.model_dump()
                    if result["diagnostics"].get("error"):
                        raise RuntimeError(f"Graph query failed: {result['diagnostics']['error']}")
                    f_by_id = {f.frame_id: f for f in bundle["frames"]}
                    r_by_id = {r.frame_id: r for r in bundle["records"]}
                    for hit in result["hits"]:
                        fid = hit["frame_id"]
                        hit["image"] = str((root / f_by_id[fid].image_relpath).resolve())
                        hit["caption"] = r_by_id[fid].extraction.caption
                else:
                    result = (ingest if args.command == "import" else verify_import)(root, driver, collection)
                    atomic_json(root.parent / f"{root.name}-{args.command}-report.json", result)
            finally:
                driver.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
