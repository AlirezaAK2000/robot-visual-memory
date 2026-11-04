"""Graph retrieval with explicit gallery/environment scopes and parameterized Cypher."""

import time

from .ingestion import database
from .schemas import Hit, QueryPlan, RetrievalResult


def compile_plan(plan: QueryPlan, environment_id, collection, k, min_score=1.0):
    """Only generated identifiers/integers enter Cypher text. All query content is parameterized.

    Label presence=2, each attribute/state on the same instance=1, each fully matched
    connected relation=3. Partial matches retain their earned score; at least one label
    must match. Relation endpoints must satisfy their anchor's attributes and states.
    """
    if k < 1 or k > 1000:
        raise ValueError("k must be 1..1000")
    params = {
        "env": environment_id,
        "collection": collection,
        "k": k,
        "threshold": min_score,
        "anchors": [a.model_dump() for a in plan.anchors],
        "predicates": [r.predicate.value for r in plan.relationships],
    }
    if not plan.anchors:
        return (
            "MATCH (f:Frame) WHERE false RETURN f.frame_id AS frame_id, 0.0 AS score, [] AS explanation",
            params,
        )

    def exists(anchor_index, *, full=False):
        label = f"$anchors[{anchor_index}].label"
        constraints = [f"{label} IN o.labels", "o.environment_id=$env", "o.frame_id=f.frame_id"]
        if full:
            constraints.extend(
                [
                    f"all(v IN $anchors[{anchor_index}].attributes WHERE v IN o.attributes)",
                    f"all(v IN $anchors[{anchor_index}].states WHERE v IN o.states)",
                ]
            )
        return (
            "EXISTS { MATCH (f)-[:OBSERVES]->(o:ObjectObservation) WHERE " + " AND ".join(constraints) + " }"
        )

    # Presentation: rank frames by visible object constraints and connected spatial relations.
    terms, explanations, label_checks = [], [], []
    for i, anchor in enumerate(plan.anchors):
        check = exists(i)
        label_checks.append(check)
        terms.append(f"CASE WHEN {check} THEN 2.0 ELSE 0.0 END")
        explanations.append(f"CASE WHEN {check} THEN ['anchor:{i}'] ELSE [] END")
        for field in ["attributes", "states"]:
            for j, _ in enumerate(getattr(anchor, field)):
                check = exists(i, full=True)
                terms.append(f"CASE WHEN {check} THEN 1.0 ELSE 0.0 END")
                explanations.append(f"CASE WHEN {check} THEN ['{field}:{i}:{j}'] ELSE [] END")
    for ri, relation in enumerate(plan.relationships):
        si, ti = relation.source, relation.target
        pattern = {
            "out": "(a)-[r:RELATES]->(b)",
            "in": "(a)<-[r:RELATES]-(b)",
            "either": "(a)-[r:RELATES]-(b)",
        }[relation.direction]
        constraints = [
            "a.frame_id=f.frame_id",
            "b.frame_id=f.frame_id",
            "a.environment_id=$env",
            "b.environment_id=$env",
            f"$anchors[{si}].label IN a.labels",
            f"r.predicate=$predicates[{ri}]",
        ]
        for var, idx in [("a", si), ("b", ti)]:
            if idx is not None:
                if var == "b":
                    constraints.append(f"$anchors[{idx}].label IN b.labels")
                constraints.extend(
                    [
                        f"all(v IN $anchors[{idx}].attributes WHERE v IN {var}.attributes)",
                        f"all(v IN $anchors[{idx}].states WHERE v IN {var}.states)",
                    ]
                )
        # near is symmetric even when a planner emits out/in.
        if relation.predicate.value == "near":
            pattern = "(a)-[r:RELATES]-(b)"
        check = (
            "EXISTS { MATCH (f)-[:OBSERVES]->(a:ObjectObservation), "
            "(f)-[:OBSERVES]->(b:ObjectObservation), "
            + pattern
            + " WHERE "
            + " AND ".join(constraints)
            + " }"
        )
        terms.append(f"CASE WHEN {check} THEN 3.0 ELSE 0.0 END")
        explanations.append(f"CASE WHEN {check} THEN ['relation:{ri}'] ELSE [] END")
    cypher = (
        "MATCH (:Gallery {collection:$collection})-[:CANDIDATE]->(f:Frame {environment_id:$env}) "
        "WHERE "
        + " OR ".join(label_checks)
        + " WITH f, "
        + " + ".join(terms)
        + " AS score, "
        + " + ".join(explanations)
        + " AS explanation WHERE score >= $threshold "
        "RETURN f.frame_id AS frame_id, score, explanation ORDER BY score DESC, frame_id ASC LIMIT $k"
    )
    return cypher, params


class GraphRetriever:
    def __init__(self, driver, planner, collection, min_score=1.0):
        self.driver, self.planner, self.collection, self.min_score = driver, planner, collection, min_score

    def retrieve(self, question, environment_id, k):
        start = time.perf_counter()
        db_seconds = 0.0
        try:
            plan, planning = self.planner.plan(question)
            cypher, params = compile_plan(plan, environment_id, self.collection, k, self.min_score)
            db_start = time.perf_counter()
            with self.driver.session(database=database(), default_access_mode="READ") as session:
                from neo4j import Query

                # Fixed read query, server-side timeout and bounded driver's transaction retry interval.
                rows = list(session.run(Query(cypher, timeout=10), params))
            db_seconds = time.perf_counter() - db_start
            hits = [
                Hit(
                    frame_id=r["frame_id"], score=float(r["score"]), diagnostics={"matched": r["explanation"]}
                )
                for r in rows
            ]
            return RetrievalResult(
                hits=hits,
                diagnostics={
                    **planning,
                    "plan": plan.model_dump(mode="json"),
                    "total_seconds": time.perf_counter() - start,
                    "database_seconds": db_seconds,
                    "error": None,
                },
            )
        except Exception as exc:
            return RetrievalResult(
                hits=[],
                diagnostics={
                    "total_seconds": time.perf_counter() - start,
                    "database_seconds": db_seconds,
                    "error": type(exc).__name__,
                },
            )
