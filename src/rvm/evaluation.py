"""Fully judged retrieval metrics, leakage checks, latency passes, and portable result reports."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .annotation import annotation_status, validate_qrels
from .io import atomic_json, digest, file_hash, portable_config, read_json


def ranking_metrics(ranking, relevant, k):
    """Binary qrels. P denominator always k; Recall denominator all relevant gallery frames.

    MRR uses the entire supplied ranking (caller retrieves the complete gallery size).
    nDCG uses log2(rank+1) discounts and ideal min(k, number-relevant) positives.
    No-relevant galleries: P=0, Recall/MRR/nDCG undefined (None), excluded from primary means.
    """
    if k < 1:
        raise ValueError("k must be positive")
    if len(ranking) != len(set(ranking)):
        raise ValueError("duplicate returned frames")
    hit_positions = [i + 1 for i, frame_id in enumerate(ranking) if frame_id in relevant]
    hits = sum(frame_id in relevant for frame_id in ranking[:k])
    precision = hits / k
    if not relevant:
        return {"precision": precision, "recall": None, "mrr": None, "ndcg": None}
    dcg = sum(1 / math.log2(i + 2) for i, frame_id in enumerate(ranking[:k]) if frame_id in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return {
        "precision": precision,
        "recall": hits / len(relevant),
        "mrr": 1 / hit_positions[0] if hit_positions else 0.0,
        "ndcg": dcg / ideal,
    }


def validate_results(result, environment_id, frame_map, limit):
    ids = [h.frame_id for h in result.hits]
    if len(ids) > limit or len(ids) != len(set(ids)):
        raise ValueError("retriever returned too many/duplicate frames")
    if any(fid not in frame_map or frame_map[fid].environment_id != environment_id for fid in ids):
        raise ValueError("wrong-environment/unknown retrieval result")
    if any(not np.isfinite(h.score) for h in result.hits):
        raise ValueError("nonfinite retrieval score")


def freeze_settings(config):
    from .extraction import EXTRACTION_PROMPT
    from .normalization import rules_hash
    from .planning import PLAN_PROMPT

    source_hashes = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    return {
        "config": portable_config(config),
        "rules_hash": rules_hash(),
        "extraction_prompt_hash": digest(EXTRACTION_PROMPT),
        "planning_prompt_hash": digest(PLAN_PROMPT),
        "source_hashes": source_hashes,
    }


def freeze_development(config, destination, *, development_environment):
    payload = {"settings": freeze_settings(config), "development_environment": development_environment}
    atomic_json(destination, {**payload, "hash": digest(payload)})


def verify_frozen(config, path):
    frozen = read_json(path)
    payload = {"settings": frozen["settings"], "development_environment": frozen["development_environment"]}
    if frozen["settings"] != freeze_settings(config) or frozen["hash"] != digest(payload):
        raise ValueError(
            "development settings changed after freeze; retune only on development then refreeze"
        )
    return frozen["hash"]


def evaluate(
    systems,
    questions,
    frames,
    judgments,
    *,
    k_values=(1, 3, 5),
    warm_repeats=2,
    synthetic=False,
    show_progress=False,
):
    if set(systems) != {"graph", "caption_embedding"}:
        raise ValueError("evaluation must compare exactly graph and caption_embedding")
    if not k_values or min(k_values) < 1 or warm_repeats < 1:
        raise ValueError("positive k values and at least one warm repeat required")
    if any(q.split != "evaluation" for q in questions):
        raise ValueError("retrieval evaluation accepts evaluation questions only")
    # An annotation CSV may include other exported questions. Validate pairs for this query set here;
    # callers validate the complete CSV against the complete exported questions before selecting queries.
    selected_ids = {q.query_id for q in questions}
    judgments = [j for j in judgments if j.query_id in selected_ids]
    validate_qrels(judgments, frames, questions)
    frame_map = {f.frame_id: f for f in frames}
    status = {s["query_id"]: s for s in annotation_status(judgments, frames, questions, synthetic=synthetic)}
    relevant = {
        q.query_id: {
            j.frame_id
            for j in judgments
            if j.query_id == q.query_id
            and j.relevance == 1
            and j.status == "confirmed"
            and j.source == ("synthetic" if synthetic else "human")
        }
        for q in questions
    }
    rows, runs = [], []
    with tqdm(
        total=len(questions) * len(systems) * (warm_repeats + 1),
        desc="Evaluate retrieval",
        unit="pass",
        disable=not show_progress,
    ) as progress:
        for qi, query in enumerate(questions):
            # Only these three values reach a retriever. Full-gallery ranking gives untruncated MRR.
            limit = status[query.query_id]["gallery_size"]
            for name in ["graph", "caption_embedding"] if qi % 2 == 0 else ["caption_embedding", "graph"]:
                retriever = systems[name]
                first = None
                for repeat in range(warm_repeats + 1):
                    progress.set_postfix(method=name, pass_number=repeat + 1, refresh=False)
                    result = retriever.retrieve(query.question, query.environment_id, limit)
                    # Isolation violations are fatal, never converted into an empty successful result.
                    validate_results(result, query.environment_id, frame_map, limit)
                    progress.update(1)
                    if first is None:
                        first = result
                    runs.append(
                        {
                            "query_id": query.query_id,
                            "method": name,
                            "phase": "first_pass" if repeat == 0 else "warm",
                            "repeat": repeat,
                            "ranking": [h.model_dump() for h in result.hits],
                            **result.diagnostics,
                        }
                    )
                ranking = [h.frame_id for h in first.hits]
                state = status[query.query_id]
                for k in k_values:
                    row = {
                        "query_id": query.query_id,
                        "environment_id": query.environment_id,
                        "category": query.category,
                        "method": name,
                        "k": k,
                        "judgment_status": state["coverage"],
                        "gallery_size": limit,
                        "judged_frames": state["judged"],
                        "relevant_frames": state["relevant"],
                        "returned_count": min(k, len(ranking)),
                        "empty": not ranking,
                        "error": first.diagnostics.get("error"),
                        "failure_category": "retrieval_error"
                        if first.diagnostics.get("error")
                        else "annotation_pending"
                        if not state["complete"]
                        else "coverage_gap"
                        if not relevant[query.query_id]
                        else "empty_graph_or_no_match"
                        if not ranking
                        else "miss_at_k"
                        if not any(fid in relevant[query.query_id] for fid in ranking[:k])
                        else "hit",
                    }
                    if state["complete"]:
                        row.update(ranking_metrics(ranking, relevant[query.query_id], k))
                    else:
                        row.update(dict.fromkeys(["precision", "recall", "mrr", "ndcg"]))
                    rows.append(row)
    return pd.DataFrame(rows), runs, list(status.values())


def report_results(table, runs, status, output, *, provenance):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "per_query.csv", index=False)
    atomic_json(output / "runs.json", runs)
    atomic_json(output / "annotation_status.json", status)
    metrics = ["precision", "recall", "mrr", "ndcg"]
    # Identical evidence-bearing, fully judged query denominator for both pipelines.
    eligible = table[table.judgment_status == "has-evidence"]
    aggregate = eligible.groupby(["method", "k"])[metrics].mean().reset_index()
    category = eligible.groupby(["category", "method", "k"])[metrics].mean().reset_index()
    aggregate = aggregate.merge(
        eligible.groupby(["method", "k"]).query_id.nunique().rename("metric_queries").reset_index(),
        on=["method", "k"],
        how="left",
    )
    category = category.merge(
        eligible.groupby(["category", "method", "k"])
        .query_id.nunique()
        .rename("metric_queries")
        .reset_index(),
        on=["category", "method", "k"],
        how="left",
    )
    diagnostic = (
        table.groupby(["method", "k"])
        .agg(
            queries=("query_id", "count"),
            returned_mean=("returned_count", "mean"),
            empty_rate=("empty", "mean"),
            errors=("error", lambda x: x.notna().sum()),
        )
        .reset_index()
    )
    aggregate.to_csv(output / "aggregate.csv", index=False)
    category.to_csv(output / "by_category.csv", index=False)
    diagnostic.to_csv(output / "diagnostics.csv", index=False)
    wide = eligible.pivot(index=["query_id", "k", "category"], columns="method", values=metrics)
    differences = pd.DataFrame(index=wide.index)
    for metric in metrics:
        if (metric, "graph") in wide and (metric, "caption_embedding") in wide:
            differences[f"{metric}_graph_minus_caption"] = (
                wide[metric, "graph"] - wide[metric, "caption_embedding"]
            )
    differences.reset_index().to_csv(output / "paired_differences.csv", index=False)
    pd.DataFrame(
        [
            {
                "method": r["method"],
                "phase": r["phase"],
                "query_id": r["query_id"],
                **{
                    key: r.get(key)
                    for key in ["total_seconds", "database_seconds", "embedding_seconds", "planning_seconds"]
                },
            }
            for r in runs
        ]
    ).to_csv(output / "latency.csv", index=False)
    counts = {
        state: sum(s["coverage"] == state for s in status)
        for state in ["annotation-pending", "no-relevant", "has-evidence"]
    }
    atomic_json(
        output / "report.json",
        {
            "provenance": provenance,
            "query_coverage": counts,
            "evaluation_complete": counts["annotation-pending"] == 0,
            "primary_metric_queries": counts["has-evidence"],
            "notes": "Metrics only for fully judged human evidence-bearing galleries. "
            "Incomplete queries have null metrics; all-zero coverage cases reported separately. "
            "First pass is a query/model-cache proxy, not a DB/OS cold restart.",
        },
    )
    return aggregate, category, diagnostic, differences.reset_index()


def compare_display(query, runs, frames, root, k=5):
    from IPython.display import display
    from .preparation import contact_sheet
    from PIL import Image

    frame_map = {f.frame_id: f for f in frames}
    strips = []
    for name in ["graph", "caption_embedding"]:
        run = next(
            r
            for r in runs
            if r["query_id"] == query.query_id and r["method"] == name and r["phase"] == "first_pass"
        )
        gallery = [frame_map[h["frame_id"]] for h in run["ranking"][:k]]
        strips.append(contact_sheet(gallery, root, columns=1, title=name))
        if name == "graph":
            print(
                "Graph plan:",
                run.get("plan"),
                "matched criteria:",
                [h["diagnostics"] for h in run["ranking"][:k]],
            )
    combined = Image.new("RGB", (sum(im.width for im in strips), max(im.height for im in strips)), "white")
    x = 0
    for strip in strips:
        combined.paste(strip, (x, 0))
        x += strip.width
    print(query.question)
    display(combined)
