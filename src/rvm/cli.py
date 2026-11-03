import argparse
import json
from pathlib import Path

from .bundle import open_bundle, validate_bundle
from .io import atomic_json, export_zip, read_json


def main():
    parser = argparse.ArgumentParser(description="Robot Visual Memory compact bundle utilities")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["import", "verify", "validate"]:
        p = commands.add_parser(name)
        p.add_argument("bundle", type=Path)
        p.add_argument("--destination", type=Path)
        p.add_argument("--collection", default=None, help="defaults to the collection recorded in the export")
    p = commands.add_parser("fixture")
    p.add_argument("output", type=Path)
    p = commands.add_parser("source-zip")
    p.add_argument("output", type=Path)
    p = commands.add_parser("export")
    p.add_argument("bundle", type=Path)
    p.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "fixture":
        from .fixture import create_fixture

        print(create_fixture(args.output))
    elif args.command == "source-zip":
        import zipfile

        root = Path(__file__).resolve().parents[2]
        with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
            for pattern in [
                "src/**/*.py",
                "notebooks/*.ipynb",
                "requirements-*.txt",
                "requirements-*.lock",
                "pyproject.toml",
                "config.example.yaml",
                "README.md",
                "docs/*.md",
                "scripts/*.py",
                "tests/**/*.py",
                "compose.yaml",
                "Dockerfile.qdrant",
                ".env.example",
                ".gitignore",
            ]:
                for path in sorted(root.glob(pattern)):
                    archive.write(path, "robot-visual-memory/" + path.relative_to(root).as_posix())
        print(args.output)
    elif args.command == "export":
        print(export_zip(args.bundle, args.output))
    else:
        root = open_bundle(args.bundle, destination=args.destination)
        bundle = validate_bundle(root)
        if args.command == "validate":
            print(
                json.dumps(
                    {"frames": len(bundle["frames"]), "status": bundle["manifest"]["benchmark_status"]}
                )
            )
        else:
            from .ingestion import clients, ingest, verify_import

            driver, qdrant = clients()
            try:
                collection = args.collection or read_json(root / "selection.json").get("config", {}).get(
                    "retrieval", {}
                ).get("collection", "robot_visual_memory")
                result = (ingest if args.command == "import" else verify_import)(
                    root, driver, qdrant, collection
                )
                result["collection"] = collection
                # Keep local reports outside immutable portable exports.
                atomic_json(root.parent / f"{root.name}-{args.command}-report.json", result)
                print(json.dumps(result, indent=2))
            finally:
                driver.close()
                qdrant.close()


if __name__ == "__main__":
    main()
