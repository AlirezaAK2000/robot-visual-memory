"""Human-readable, self-contained image/extraction review for the notebook pilot."""

import base64
import hashlib
import json
from html import escape
from pathlib import Path


def extraction_review_html(frames, records, root, *, model, synthetic=False):
    """Pair outputs by frame ID, preserving input order and displaying the exact exported image."""
    frames, records = list(frames), list(records)
    by_id = {r.frame_id: r for r in records}
    if (
        not frames
        or len(by_id) != len(records)
        or len({f.frame_id for f in frames}) != len(frames)
        or set(by_id) != {f.frame_id for f in frames}
    ):
        raise ValueError("Review requires exactly one extraction per input frame")

    cards = []
    for index, frame in enumerate(frames, 1):
        record = by_id[frame.frame_id]
        extraction = record.extraction
        image = (Path(root) / frame.image_relpath).read_bytes()
        if hashlib.sha256(image).hexdigest() != frame.image_sha256:
            raise ValueError(f"Review image hash mismatch: {frame.frame_id}")
        image_uri = "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")
        rows = []
        for obj in extraction.objects:
            attributes = (
                "<br>".join(
                    f"{escape(a.name)}: {escape(a.value) if a.value is not None else '<em>Unknown</em>'}"
                    for a in obj.attributes
                )
                or "<span class='rvm-muted'>None reported</span>"
            )
            states = (
                ", ".join(escape(s) for s in obj.states) or "<span class='rvm-muted'>None reported</span>"
            )
            rows.append(
                f"<tr><td><strong>{escape(obj.label)}</strong><br><code>{escape(obj.object_id)}</code></td>"
                f"<td>{attributes}</td><td>{states}</td></tr>"
            )
        objects = (
            "<div class='rvm-table'><table><thead><tr><th>Object / ID</th><th>Attributes</th>"
            "<th>States</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
            if rows
            else "<p class='rvm-muted'>No objects reported.</p>"
        )
        names = {o.object_id: f"{o.label} ({o.object_id})" for o in extraction.objects}
        relations = "".join(
            f"<li>{escape(names[r.source])} <strong>→ {escape(r.predicate.value.replace('_', ' '))} →</strong> "
            f"{escape(names[r.target])}</li>"
            for r in extraction.relationships
        )
        relationships = (
            f"<ul>{relations}</ul>" if relations else "<p class='rvm-muted'>No relationships reported.</p>"
        )
        raw = escape(json.dumps(extraction.model_dump(mode="json"), indent=2, ensure_ascii=False))
        cards.append(f"""
<article class="rvm-card">
  <header><h3>Frame {index} of {len(frames)}</h3>
    <span>Original sequence index {frame.sequence_index} · {frame.width} × {frame.height} px</span></header>
  <div class="rvm-columns">
    <figure><img src="{image_uri}" alt="Input frame {index}, sequence index {frame.sequence_index}">
      <figcaption>Exact image sent to the VLM<br>Frame ID: <code>{escape(frame.frame_id)}</code><br>
      Episode: {escape(frame.episode_history)}</figcaption></figure>
    <div class="rvm-extraction"><h4>Caption</h4><p class="rvm-caption">{escape(extraction.caption)}</p>
      <h4>Objects <span>({len(extraction.objects)})</span></h4>{objects}
      <h4>Relationships <span>({len(extraction.relationships)})</span></h4>{relationships}
      <details><summary>View structured JSON</summary><pre>{raw}</pre></details>
    </div>
  </div>
</article>""")

    status = "SYNTHETIC EXAMPLE · No model requests" if synthetic else f"Model: {model}"
    return """
<style>
.rvm-review {font:15px/1.6 system-ui,sans-serif;color:#182b3a;background:#f3f6fa;padding:24px;border-radius:16px;max-width:1400px;margin:auto;}
.rvm-review * {box-sizing:border-box;}
.rvm-review h2,.rvm-review h3,.rvm-review h4,.rvm-review p {margin:0 0 12px;color:inherit;}
.rvm-review h2 {font-size:26px;}.rvm-review h3 {font-size:19px;}.rvm-review h4 {font-size:16px;margin-top:20px;}
.rvm-review .rvm-status {color:#17645e;font-weight:600;}
.rvm-review .rvm-card {border:1px solid #d7e1ec;background:#fff;border-radius:12px;margin-top:24px;overflow:hidden;}
.rvm-review header {padding:16px 22px;background:#eaf1f8;}.rvm-review header h3 {margin:0;}
.rvm-review header span,.rvm-review figcaption,.rvm-review .rvm-muted {color:#526779;}
.rvm-review .rvm-columns {display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.15fr);gap:24px;padding:22px;}
.rvm-review figure {margin:0;min-width:0;}.rvm-review img {width:100%;height:auto;display:block;border-radius:8px;}
.rvm-review figcaption {font-size:12px;margin-top:12px;overflow-wrap:anywhere;}
.rvm-review .rvm-extraction {min-width:0;overflow-wrap:anywhere;}.rvm-review .rvm-extraction h4:first-child {margin-top:0;}
.rvm-review .rvm-caption {padding:14px 16px;border-left:3px solid #268479;background:#f1f8f6;border-radius:0 6px 6px 0;white-space:pre-wrap;}
.rvm-review .rvm-table {overflow-x:auto;}.rvm-review table {width:100%;border-collapse:collapse;font-size:13px;text-align:left;}
.rvm-review th,.rvm-review td {text-align:left;padding:9px 10px;border-bottom:1px solid #e3e9ef;vertical-align:top;}
.rvm-review th {background:#f4f7fa;}.rvm-review ul {padding-left:20px;}.rvm-review li {margin-bottom:8px;}
.rvm-review code {font-size:12px;}.rvm-review details {margin-top:20px;border-top:1px solid #e3e9ef;padding-top:12px;}
.rvm-review summary {cursor:pointer;color:#17645e;}.rvm-review pre {white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7fa;padding:14px;font-size:12px;color:#182b3a;}
@media(max-width:850px) {.rvm-review .rvm-columns {grid-template-columns:1fr;}.rvm-review {padding:12px;}}
</style>
<section class="rvm-review"><h2>Pilot extraction review</h2>
""" + (
        f"<p class='rvm-status'>{escape(status)} · {len(frames)} frames</p>"
        "<p>Compare each input with its caption, objects, attributes, states and relationship directions. "
        "Empty lists mean nothing was reported; they do not establish absence. "
        "Spatial left/above relationships are relative to the image view.</p>" + "".join(cards) + "</section>"
    )
