"""Idempotent, frame-scoped ingestion through official Python clients, never MCP."""

import json
import os
import time
import uuid

from .bundle import validate_bundle
from .io import digest
from .normalization import object_properties

NAMESPACE = uuid.UUID("75cd126b-de13-4843-b32e-b1d240c8f710")


def point_id(frame_id):
    return str(uuid.uuid5(NAMESPACE, frame_id))


def observation_id(environment_id, frame_id, local_id):
    return digest([environment_id, frame_id, local_id])


def gallery_contract(bundle):
    return digest(
        {
            "embedding": bundle["manifest"]["embedding_contract_hash"],
            "frame_ids": [f.frame_id for f in bundle["frames"]],
            "images": [f.image_sha256 for f in bundle["frames"]],
            "records": [r.model_dump(mode="json") for r in bundle["records"]],
        }
    )


def clients():
    from dotenv import load_dotenv
    from neo4j import GraphDatabase
    from qdrant_client import QdrantClient

    load_dotenv()
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.environ["NEO4J_PASSWORD"]),
        max_transaction_retry_time=10,
    )
    return driver, QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"), timeout=15)


def database():
    return os.getenv("NEO4J_DATABASE", "neo4j")


def ingest(root, driver, qdrant, collection):
    from qdrant_client import models as qm

    start = time.perf_counter()
    bundle = validate_bundle(root)
    frames, vectors = bundle["frames"], bundle["vectors"]
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
        existing = session.run(
            "MATCH (g:Gallery {collection:$c}) RETURN g.contract AS contract", c=collection
        ).single()
        if existing and existing["contract"] != contract:
            raise ValueError("collection already owns a different export; choose a new collection name")
        session.run(
            "MERGE (g:Gallery {collection:$c}) SET g.contract=$contract", c=collection, contract=contract
        ).consume()
    dimension = bundle["embedding"]["dimension"]
    if not qdrant.collection_exists(collection):
        qdrant.create_collection(
            collection, vectors_config=qm.VectorParams(size=dimension, distance=qm.Distance.COSINE)
        )
    vector_config = qdrant.get_collection(collection).config.params.vectors
    if vector_config.size != dimension or vector_config.distance != qm.Distance.COSINE:
        raise ValueError("Qdrant vector contract mismatch")
    qdrant.create_payload_index(collection, "environment_id", qm.PayloadSchemaType.KEYWORD, wait=True)
    # If retrying after interruption, existing points must belong to this exact bundle contract.
    offset = None
    while True:
        points, offset = qdrant.scroll(
            collection, limit=256, offset=offset, with_payload=True, with_vectors=False
        )
        if any(p.payload.get("bundle_contract") != contract for p in points):
            raise ValueError("Qdrant collection contains a different export; use a fresh collection name")
        if offset is None:
            break

    def write_frame(tx, frame, extraction):
        f = frame.model_dump()
        f["extraction_hash"] = digest(extraction.model_dump(mode="json"))
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
        for frame in frames:
            session.execute_write(write_frame, frame, records[frame.frame_id])
    for start_idx in range(0, len(frames), 64):
        batch = []
        for i in range(start_idx, min(start_idx + 64, len(frames))):
            frame = frames[i]
            batch.append(
                qm.PointStruct(
                    id=point_id(frame.frame_id),
                    vector=vectors[i].tolist(),
                    payload={
                        "frame_id": frame.frame_id,
                        "environment_id": frame.environment_id,
                        "caption": records[frame.frame_id].caption,
                        "image_relpath": frame.image_relpath,
                        "bundle_contract": contract,
                    },
                )
            )
        qdrant.upsert(collection, batch, wait=True)
    report = verify_import(root, driver, qdrant, collection)
    report["indexing_seconds"] = time.perf_counter() - start
    return report


def verify_import(root, driver, qdrant, collection):
    import numpy as np

    bundle = validate_bundle(root)
    frames, records = bundle["frames"], bundle["records"]
    expected = {f.frame_id for f in frames}
    contract = gallery_contract(bundle)
    with driver.session(database=database(), default_access_mode="READ") as session:
        stored = session.run(
            "MATCH (g:Gallery {collection:$c}) RETURN g.contract AS contract", c=collection
        ).single()
        if not stored or stored["contract"] != contract:
            raise ValueError("Neo4j gallery content contract differs")
        rows = list(
            session.run(
                """MATCH (:Gallery {collection:$c})-[:CANDIDATE]->(f:Frame)
                                  OPTIONAL MATCH (f)-[:OBSERVES]->(o:ObjectObservation)
                                  RETURN f.frame_id AS fid, f.environment_id AS env,
                                  f.image_relpath AS image_relpath, count(o) AS objects""",
                c=collection,
            )
        )
        if {r["fid"] for r in rows} != expected or len(rows) != len(expected):
            raise ValueError("Neo4j gallery frame count/IDs differ")
        by_frame = {r.frame_id: len(r.extraction.objects) for r in records}
        environments = {f.frame_id: f.environment_id for f in frames}
        paths = {f.frame_id: f.image_relpath for f in frames}
        if any(
            r["objects"] != by_frame[r["fid"]]
            or r["env"] != environments[r["fid"]]
            or r["image_relpath"] != paths[r["fid"]]
            for r in rows
        ):
            raise ValueError("Neo4j object count or environment mismatch")
        invalid = session.run(
            """MATCH (a:ObjectObservation)-[:RELATES]->(b:ObjectObservation)
                                WHERE a.frame_id IN $ids AND
                                (a.frame_id <> b.frame_id OR a.environment_id <> b.environment_id)
                                RETURN count(*) AS n""",
            ids=sorted(expected),
        ).single()["n"]
        if invalid:
            raise ValueError("cross-frame/environment relation found")
        edges = session.run(
            """MATCH (a:ObjectObservation)-[r:RELATES]->()
                                WHERE a.frame_id IN $ids RETURN count(r) AS n""",
            ids=sorted(expected),
        ).single()["n"]
        if edges != sum(len(r.extraction.relationships) for r in records):
            raise ValueError("Neo4j edge count differs")
    points, offset = [], None
    while True:
        page, offset = qdrant.scroll(
            collection, limit=256, offset=offset, with_vectors=True, with_payload=True
        )
        points.extend(page)
        if offset is None:
            break
    mapping = {point_id(f.frame_id): i for i, f in enumerate(frames)}
    captions = {r.frame_id: r.extraction.caption for r in records}
    if {str(p.id) for p in points} != set(mapping) or len(points) != len(expected):
        raise ValueError("Qdrant deterministic point IDs/count differ from gallery")
    for p in points:
        i = mapping[str(p.id)]
        frame = frames[i]
        if (
            p.payload["frame_id"] != frame.frame_id
            or p.payload["environment_id"] != frame.environment_id
            or p.payload["image_relpath"] != frame.image_relpath
            or p.payload.get("caption") != captions[frame.frame_id]
            or p.payload.get("bundle_contract") != contract
            or not np.allclose(p.vector, bundle["vectors"][i], atol=1e-5)
        ):
            raise ValueError("Qdrant payload/vector row alignment mismatch")
    return {
        "frames": len(rows),
        "objects": sum(r["objects"] for r in rows),
        "relationships": edges,
        "vectors": len(points),
        "verified": True,
    }
