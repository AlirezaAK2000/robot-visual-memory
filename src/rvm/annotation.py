"""Human qrels: explicit unknowns, exhaustive gallery templates and resumable widgets."""

import csv
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io import atomic_json, read_json
from .preparation import contact_sheet
from .schemas import Judgment

CRITERION = (
    "Relevant (1): the frame contains visual evidence useful for answering this specific question, "
    "including useful partial evidence when multiple views are needed. Nonrelevant (0): inspected "
    "frame provides no useful evidence. Unjudged: not yet inspected; never infer a 0. "
    "Reference answers may aid humans but are never retriever inputs. Judge every selected frame "
    "in the question's environment. All-zero fully judged galleries are no-evidence coverage cases."
)


def save_qrels(path, judgments):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [Judgment.model_validate(j.model_dump() if isinstance(j, Judgment) else j) for j in judgments]
    if len({(j.query_id, j.frame_id) for j in rows}) != len(rows):
        raise ValueError("duplicate judgment pair")
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".qrels-")
    try:
        with os.fdopen(fd, "w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=list(Judgment.model_fields))
            writer.writeheader()
            writer.writerows(j.model_dump() for j in rows)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load_qrels(path):
    with Path(path).open(newline="") as stream:
        return [
            Judgment.model_validate(
                {**row, "relevance": None if row["relevance"] == "" else int(row["relevance"])}
            )
            for row in csv.DictReader(stream)
        ]


def validate_qrels(judgments, frames, questions):
    f = {frame.frame_id: frame for frame in frames}
    q = {question.query_id: question for question in questions}
    seen = set()
    for j in judgments:
        pair = (j.query_id, j.frame_id)
        if pair in seen:
            raise ValueError("duplicate qrel")
        seen.add(pair)
        if j.query_id not in q or j.frame_id not in f:
            raise ValueError("qrel references unknown query/frame")
        if f[j.frame_id].environment_id != q[j.query_id].environment_id:
            raise ValueError("qrel crosses environment")


def annotation_status(judgments, frames, query_ids, *, synthetic=False):
    accepted = {"synthetic"} if synthetic else {"human"}
    galleries = defaultdict(set)
    for f in frames:
        galleries[f.environment_id].add(f.frame_id)
    by_query = defaultdict(list)
    for j in judgments:
        by_query[j.query_id].append(j)
    result = []
    for query in query_ids:
        qid, env = query.query_id, query.environment_id
        rows = by_query[qid]
        valid = [
            j for j in rows if j.source in accepted and j.status == "confirmed" and j.relevance is not None
        ]
        complete = {j.frame_id for j in valid} == galleries[env]
        result.append(
            {
                "query_id": qid,
                "complete": complete,
                "judged": len(valid),
                "gallery_size": len(galleries[env]),
                "relevant": sum(j.relevance for j in valid),
                "coverage": "no-relevant"
                if complete and not any(j.relevance for j in valid)
                else "has-evidence"
                if complete
                else "annotation-pending",
            }
        )
    return result


def create_materials(bundle, frames, questions, config):
    from tqdm.auto import tqdm

    bundle = Path(bundle)
    directory = bundle / "annotation"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "qrels.csv"
    rng = np.random.default_rng(config["seed"])
    chosen = []
    # Seeded query selection, independent of answers and retrieval outcomes.
    for env in sorted({q.environment_id for q in questions if q.split == "evaluation"}):
        pool = sorted([q for q in questions if q.environment_id == env], key=lambda q: q.query_id)
        n = min(config["dataset"]["questions_per_eval_environment"], len(pool))
        chosen.extend(pool[int(i)] for i in rng.choice(len(pool), n, replace=False))
    atomic_json(directory / "query_ids.json", [q.query_id for q in chosen])
    if path.exists():
        judgments = load_qrels(path)
        validate_qrels(judgments, frames, questions)
        expected = {
            (q.query_id, f.frame_id) for q in chosen for f in frames if f.environment_id == q.environment_id
        }
        if {(j.query_id, j.frame_id) for j in judgments} != expected:
            raise ValueError(
                "Existing annotations use a different gallery/query selection; use a fresh bundle"
            )
    else:
        judgments = [
            Judgment(
                query_id=q.query_id, frame_id=f.frame_id, relevance=None, source="unjudged", status="unjudged"
            )
            for q in chosen
            for f in frames
            if f.environment_id == q.environment_id
        ]
        save_qrels(path, judgments)
    (directory / "CRITERION.txt").write_text(CRITERION)
    # Shared pages avoid duplicating the same images for every question.
    for env in sorted({q.environment_id for q in chosen}):
        gallery = [f for f in frames if f.environment_id == env]
        env_dir = directory / "review" / env.split("/")[-1]
        env_dir.mkdir(parents=True, exist_ok=True)
        for page, start in enumerate(
            tqdm(range(0, len(gallery), 12), desc="Build annotation contact sheets", unit="page")
        ):
            contact_sheet(gallery[start : start + 12], bundle, title=f"{env} / page {page}").save(
                env_dir / f"page-{page:03}.jpg"
            )
        atomic_json(env_dir / "frame_order.json", [f.model_dump() for f in gallery])
        atomic_json(env_dir / "questions.json", [q.model_dump() for q in chosen if q.environment_id == env])
    return chosen, judgments


def annotation_widget(bundle, frames, questions, page_size=6):
    import ipywidgets as w
    from IPython.display import display

    bundle = Path(bundle)
    qids = read_json(bundle / "annotation/query_ids.json")
    chosen = [q for q in questions if q.query_id in qids]
    if not chosen:
        raise ValueError("No evaluation queries selected")
    path = bundle / "annotation/qrels.csv"
    select = w.Dropdown(options=[(q.question, q.query_id) for q in chosen], description="Question")
    page = w.BoundedIntText(value=0, min=0, max=0, description="Page")
    output, message = w.Output(), w.HTML()
    controls = []

    def render(*_):
        output.clear_output()
        controls.clear()
        q = next(q for q in chosen if q.query_id == select.value)
        gallery = [f for f in frames if f.environment_id == q.environment_id]
        page.max = (len(gallery) - 1) // page_size
        judgments = {(j.query_id, j.frame_id): j for j in load_qrels(path)}
        with output:
            print(q.question, "\n", CRITERION)
            for frame in gallery[page.value * page_size : (page.value + 1) * page_size]:
                j = judgments[(q.query_id, frame.frame_id)]
                data = (bundle / frame.image_relpath).read_bytes()
                label = w.RadioButtons(
                    options=[("Unjudged", -1), ("Nonrelevant", 0), ("Relevant", 1)],
                    value=-1 if j.relevance is None else j.relevance,
                )
                notes = w.Text(value=j.notes, description="Notes")
                controls.append((q.query_id, frame.frame_id, label, notes))
                display(
                    w.VBox(
                        [w.HTML(frame.frame_id), w.Image(value=data, format="jpeg", width=480), label, notes]
                    )
                )

    def save(_):
        rows = {(j.query_id, j.frame_id): j for j in load_qrels(path)}
        for qid, fid, radio, notes in controls:
            rows[qid, fid] = Judgment(
                query_id=qid,
                frame_id=fid,
                relevance=None if radio.value == -1 else radio.value,
                source="unjudged" if radio.value == -1 else "human",
                status="unjudged" if radio.value == -1 else "confirmed",
                notes=notes.value,
            )
        save_qrels(path, rows.values())
        message.value = "Saved atomically. Save each page before changing page/question."

    button = w.Button(description="Save this page")
    button.on_click(save)
    select.observe(render, names="value")
    page.observe(render, names="value")
    render()
    return w.VBox([select, page, button, message, output])
