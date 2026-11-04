"""Compact notebook interface for query plans and retrieved visual evidence."""

import json
from html import escape

import pandas as pd
from IPython.display import HTML, display
from tqdm.auto import tqdm

from .ingestion import clients
from .retrieval import GraphRetriever, compile_plan
from .review import extraction_review_html


class GallerySearch:
    """Keep display and connection management out of the presentation cells."""

    def __init__(self, bundle, planner, collection, environment, config, *, synthetic=False):
        self.bundle = bundle
        self.planner = planner
        self.collection = collection
        self.environment = environment
        self.model = config["extraction"]["model"]
        self.min_score = config["retrieval"].get("min_graph_score", 1.0)
        self.synthetic = synthetic
        self.frames = {f.frame_id: f for f in bundle["frames"]}
        self.records = {r.frame_id: r for r in bundle["records"]}

    def _show_plan(self, question, k, *, custom=False):
        plan, diagnostics = self.planner.plan(question)
        display(
            pd.DataFrame([a.model_dump() for a in plan.anchors], columns=["label", "attributes", "states"])
        )
        if plan.relationships:
            display(pd.DataFrame([r.model_dump(mode="json") for r in plan.relationships]))
        elif not custom:
            print("No relationship constraint was parsed for this question.")
        print(diagnostics.get("unparsed_terms_notice", ""))
        cypher, parameters = compile_plan(plan, self.environment, self.collection, k, self.min_score)
        summary = (
            "Inspect custom query Cypher and parameters"
            if custom
            else "Inspect generated Cypher and parameters"
        )
        display(
            HTML(
                f"<details><summary>{summary}</summary><pre>{escape(cypher)}</pre>"
                f"<pre>{escape(json.dumps(parameters, indent=2))}</pre></details>"
            )
        )
        return plan

    @staticmethod
    def _constraint(key, plan):
        parts = key.split(":")
        index = int(parts[1])
        if parts[0] == "anchor":
            return f"Object: {plan.anchors[index].label}"
        if parts[0] in {"attributes", "states"}:
            anchor = plan.anchors[index]
            return f"{anchor.label}: {getattr(anchor, parts[0])[int(parts[2])]}"
        relation = plan.relationships[index]
        source = plan.anchors[relation.source].label
        target = (
            plan.anchors[relation.target].label if relation.target is not None else "any visible neighbor"
        )
        return f"Relation: {source}, {relation.predicate.value}, {target} ({relation.direction})"

    def _show_hits(self, result, plan):
        if result.diagnostics.get("error"):
            raise RuntimeError(
                f"Graph search failed: {result.diagnostics['error']}. Check Neo4j and the setup cells."
            )
        display(
            pd.DataFrame(
                [
                    {
                        "rank": rank,
                        "frame_id": hit.frame_id,
                        "score": hit.score,
                        "matched_constraints": "; ".join(
                            self._constraint(key, plan) for key in hit.diagnostics.get("matched", [])
                        ),
                    }
                    for rank, hit in enumerate(result.hits, 1)
                ],
                columns=["rank", "frame_id", "score", "matched_constraints"],
            )
        )
        ids = [hit.frame_id for hit in result.hits]
        if ids:
            display(
                HTML(
                    extraction_review_html(
                        [self.frames[fid] for fid in ids],
                        [self.records[fid] for fid in ids],
                        self.bundle["root"],
                        model=self.model,
                    )
                )
            )
        else:
            print("No matching visual evidence found.")
        print("Returned frames can be partial matches. Inspect their images and matched constraints.")

    def examples(self, questions, retriever=None, *, k=5):
        plans, results = [], []
        notes = {
            "a5a88a20-7d66-4b5c-bbfe-3efae2da465a": "Front is not encoded; the search cannot identify the front door.",
            "deb29a55-6b43-48b6-b974-0c1cb8061cc9": "Upstairs is not encoded; the search has no floor/location filter.",
        }
        for number, question in enumerate(
            tqdm(questions, desc="Graph question examples", unit="question"), 1
        ):
            display(HTML(f"<h2>Example {number}: {escape(question.question)}</h2>"))
            print("Original query ID:", question.query_id)
            plan = self._show_plan(question.question, k)
            plans.append(plan.model_dump(mode="json"))
            if question.query_id in notes:
                print("Parser limitation:", notes[question.query_id])
            if self.synthetic:
                print("SYNTHETIC SMOKE: plan compiled; no database query or retrieval result simulated.")
                results.append({"query_id": question.query_id, "database_query_executed": False})
                continue
            result = retriever.retrieve(question.question, self.environment, k)
            results.append({"query_id": question.query_id, **result.model_dump()})
            self._show_hits(result, plan)
        return plans, results

    def ask(self, questions, *, k=5):
        """A fresh connection makes this cell safe to rerun after notebook cleanup."""
        if not questions or any(not isinstance(q, str) or not q.strip() for q in questions):
            raise ValueError("Enter at least one nonempty question.")
        if type(k) is not int or not 1 <= k <= 1000:
            raise ValueError("The result limit must be an integer between 1 and 1000.")
        driver = None
        results = []
        try:
            if not self.synthetic:
                driver = clients()
                retriever = GraphRetriever(driver, self.planner, self.collection, self.min_score)
            # Presentation: one question becomes a plan, a graph query, then ranked image evidence.
            for number, question in enumerate(questions, 1):
                question = question.strip()
                display(HTML(f"<h2>Your question {number}: {escape(question)}</h2>"))
                plan = self._show_plan(question, k, custom=True)
                if not plan.anchors:
                    print("No supported object was recognized. Try naming a visible object from the dataset.")
                    results.append({"question": question, "hits": [], "unsupported": True})
                elif self.synthetic:
                    print(
                        "SYNTHETIC SMOKE: query compiled; no database request or retrieval result simulated."
                    )
                    results.append(
                        {
                            "question": question,
                            "plan": plan.model_dump(mode="json"),
                            "database_query_executed": False,
                        }
                    )
                else:
                    result = retriever.retrieve(question, self.environment, k)
                    results.append({"question": question, **result.model_dump()})
                    self._show_hits(result, plan)
        finally:
            if driver is not None:
                driver.close()
        return results
