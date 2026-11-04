"""Idempotent, frame-scoped ingestion through the official Neo4j Python driver."""

import json
import os
from tqdm.auto import tqdm

from .bundle import validate_bundle
from .io import digest
from .normalization import object_properties, rules_hash


def observation_id(environment_id, frame_id, local_id):
    # Presentation: repeated labels across views remain separate object observations.
    return digest([environment_id, frame_id, local_id])


def gallery_contract(bundle):
    return digest(
        {
            "normalization": rules_hash(),
            "frames": [f.model_dump() for f in bundle["frames"]],
            "images": [f.image_sha256 for f in bundle["frames"]],
            "records": [r.model_dump(mode="json") for r in bundle["records"]],
        }
    )


def clients():
    from dotenv import load_dotenv
    from neo4j import GraphDatabase

    load_dotenv()
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.environ["NEO4J_PASSWORD"]),
        max_transaction_retry_time=10,
    )
    return driver


def database():
    return os.getenv("NEO4J_DATABASE", "neo4j")


def ingest(root, driver, collection="robot_visual_memory", *, show_progress=True):
    bundle = validate_bundle(root)
    frames = bundle["frames"]
    records = {r.frame_id: r.extraction for r in bundle["records"]}
    # Collection contract binds the complete gallery. Prevents stale candidates and incompatible models.
    contract = gallery_contract(bundle)
    driver.verify_connectivity()
    with driver.session(database=database()) as session:
        # Frame IDs are stable across selections. Refuse to mutate observations owned by an
        # existing gallery if a new export changed preprocessing or extraction for that frame.
        for frame in frames:
            extraction_hash = digest(records[frame.frame_id].model_dump(mode="json"))
            old = session.run(
                "MATCH (f:Frame {frame_id:$fid}) RETURN f.image_sha256 AS image, f.extraction_hash AS extraction",
                fid=frame.frame_id,
            ).single()
            if old and (old["image"] != frame.image_sha256 or old["extraction"] != extraction_hash):
                raise ValueError(
                    "an existing graph frame has different image/extraction content; use a separate Neo4j database/project"
                )
        for label, prop in [
            ("Environment", "environment_id"),
            ("Frame", "frame_id"),
            ("ObjectObservation", "observation_id"),
            ("Gallery", "collection"),
        ]:
            session.run(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
            ).consume()
        session.run("CREATE INDEX IF NOT EXISTS FOR (f:Frame) ON (f.environment_id)").consume()
        session.run("CREATE INDEX IF NOT EXISTS FOR (o:ObjectObservation) ON (o.frame_id)").consume()
        session.run(
            "CREATE INDEX IF NOT EXISTS FOR (o:ObjectObservation) ON (o.environment_id, o.label)"
        ).consume()
        existing = session.run(
            "MATCH (g:Gallery {collection:$c}) RETURN g.contract AS contract", c=collection
        ).single()
        if existing and existing["contract"] != contract:
            raise ValueError("collection already owns a different export; choose a new collection name")
        session.run(
            "MERGE (g:Gallery {collection:$c}) SET g.contract=$contract", c=collection, contract=contract
        ).consume()

    def write_frame(tx, frame, extraction):
        f = frame.model_dump()
        f["extraction_hash"] = digest(extraction.model_dump(mode="json"))
        f["caption"] = extraction.caption
        objects = []
        for obj in extraction.objects:
            objects.append(
                {
                    "observation_id": observation_id(frame.environment_id, frame.frame_id, obj.object_id),
                    "local_id": obj.object_id,
                    **object_properties(obj),
                    "attributes_json": json.dumps([a.model_dump() for a in obj.attributes], sort_keys=True),
                }
            )
        edges = [
            {
                "source": observation_id(frame.environment_id, frame.frame_id, r.source),
                "target": observation_id(frame.environment_id, frame.frame_id, r.target),
                "predicate": r.predicate.value,
            }
            for r in extraction.relationships
        ]
        tx.run(
            """MERGE (e:Environment {environment_id:$f.environment_id})
                  MERGE (f:Frame {frame_id:$f.frame_id}) SET f += $f
                  MERGE (e)-[:HAS_FRAME]->(f)
                  WITH f MATCH (g:Gallery {collection:$collection}) MERGE (g)-[:CANDIDATE]->(f)""",
            f=f,
            collection=collection,
        ).consume()
        # Atomic per-frame replacement removes stale observations on retry without duplicates.
        tx.run(
            "MATCH (f:Frame {frame_id:$fid})-[:OBSERVES]->(o:ObjectObservation) DETACH DELETE o",
            fid=frame.frame_id,
        ).consume()
        tx.run(
            """MATCH (f:Frame {frame_id:$fid}) UNWIND $objects AS obj
                  CREATE (o:ObjectObservation) SET o = obj, o.frame_id=$fid, o.environment_id=$env
                  CREATE (f)-[:OBSERVES]->(o)""",
            fid=frame.frame_id,
            env=frame.environment_id,
            objects=objects,
        ).consume()
        tx.run(
            """UNWIND $edges AS edge
                  MATCH (a:ObjectObservation {observation_id:edge.source, frame_id:$fid, environment_id:$env})
                  MATCH (b:ObjectObservation {observation_id:edge.target, frame_id:$fid, environment_id:$env})
                  MERGE (a)-[r:RELATES {predicate:edge.predicate}]->(b)""",
            edges=edges,
            fid=frame.frame_id,
            env=frame.environment_id,
        ).consume()

    with driver.session(database=database()) as session:
        for frame in tqdm(frames, desc="Import Neo4j frames", unit="frame", disable=not show_progress):
            session.execute_write(write_frame, frame, records[frame.frame_id])
    return verify_import(root, driver, collection)


def verify_import(root, driver, collection="robot_visual_memory"):
    """Check actual frame properties, normalized objects and directed edges against the export."""
    bundle = validate_bundle(root)
    frames = {f.frame_id: f for f in bundle["frames"]}
    records = {r.frame_id: r.extraction for r in bundle["records"]}
    with driver.session(database=database(), default_access_mode="READ") as session:
        stored = session.run(
            "MATCH (g:Gallery {collection:$c}) RETURN g.contract AS contract", c=collection
        ).single()
        if not stored or stored["contract"] != gallery_contract(bundle):
            raise ValueError("Neo4j gallery content contract differs")
        rows = list(
            session.run(
                "MATCH (:Gallery {collection:$c})-[:CANDIDATE]->(f:Frame) RETURN properties(f) AS f",
                c=collection,
            )
        )
        if len(rows) != len(frames) or {r["f"]["frame_id"] for r in rows} != set(frames):
            raise ValueError("Neo4j gallery frame count/IDs differ")
        for row in rows:
            stored_frame = row["f"]
            f = frames[stored_frame["frame_id"]]
            expected = {k: v for k, v in f.model_dump().items() if v is not None}
            expected.update(
                caption=records[f.frame_id].caption,
                extraction_hash=digest(records[f.frame_id].model_dump(mode="json")),
            )
            if any(stored_frame.get(k) != v for k, v in expected.items()):
                raise ValueError("Neo4j frame content differs")
        objects = list(
            session.run(
                """MATCH (:Gallery {collection:$c})-[:CANDIDATE]->(f:Frame)-[:OBSERVES]->(o:ObjectObservation)
                                 RETURN properties(o) AS o""",
                c=collection,
            )
        )
        expected_objects = {}
        for fid, extraction in records.items():
            for obj in extraction.objects:
                oid = observation_id(frames[fid].environment_id, fid, obj.object_id)
                expected_objects[oid] = {
                    "observation_id": oid,
                    "local_id": obj.object_id,
                    "frame_id": fid,
                    "environment_id": frames[fid].environment_id,
                    **object_properties(obj),
                    "attributes_json": json.dumps([a.model_dump() for a in obj.attributes], sort_keys=True),
                }
        if len(objects) != len(expected_objects) or {r["o"]["observation_id"] for r in objects} != set(
            expected_objects
        ):
            raise ValueError("Neo4j object count/IDs differ")
        if any(dict(r["o"]) != expected_objects[r["o"]["observation_id"]] for r in objects):
            raise ValueError("Neo4j object properties differ")
        edges = list(
            session.run(
                """MATCH (a:ObjectObservation)-[r:RELATES]->(b:ObjectObservation)
                                WHERE a.frame_id IN $ids OR b.frame_id IN $ids
                                RETURN a.observation_id AS a, b.observation_id AS b, r.predicate AS p,
                                       a.frame_id <> b.frame_id OR a.environment_id <> b.environment_id AS invalid""",
                ids=list(frames),
            )
        )
        if any(e["invalid"] for e in edges):
            raise ValueError("cross-frame/environment relation found")
        expected_edges = {
            (
                observation_id(frames[fid].environment_id, fid, e.source),
                e.predicate.value,
                observation_id(frames[fid].environment_id, fid, e.target),
            )
            for fid, extraction in records.items()
            for e in extraction.relationships
        }
        if len(edges) != len(expected_edges) or {(e["a"], e["p"], e["b"]) for e in edges} != expected_edges:
            raise ValueError("Neo4j relationships differ")
    return {
        "frames": len(frames),
        "objects": len(expected_objects),
        "relationships": len(expected_edges),
        "environments": len({f.environment_id for f in frames.values()}),
        "collection": collection,
    }


def stored_vocabulary(driver, collection):
    with driver.session(database=database(), default_access_mode="READ") as session:
        rows = list(
            session.run(
                """MATCH (:Gallery {collection:$c})-[:CANDIDATE]->(:Frame)-[:OBSERVES]->(o:ObjectObservation)
                                RETURN o.labels AS labels, o.attributes AS attributes, o.states AS states""",
                c=collection,
            )
        )
    return {key: sorted({v for row in rows for v in row[key]}) for key in ("labels", "attributes", "states")}
