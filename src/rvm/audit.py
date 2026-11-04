"""Structural and deterministic search-readiness audit of collected graph records."""

from collections import Counter, defaultdict

from .bundle import validate_bundle
from .normalization import (
    LABEL_PARENTS,
    OBJECT_VOCABULARY,
    UNKNOWN,
    canonical,
    graph_vocabulary,
    object_properties,
)


def audit_dataset(root):
    bundle = validate_bundle(root)
    frames, records = bundle["frames"], bundle["records"]
    objects = [o for r in records for o in r.extraction.objects]
    relations = [e for r in records for e in r.extraction.relationships]
    labels = Counter(o.label for o in objects)
    attributes = [a for o in objects for a in o.attributes]
    groups = defaultdict(list)
    for f in frames:
        groups[f.image_sha256].append(f.frame_id)
    mappings = {label: canonical(label) for label in sorted(labels) if canonical(label) != label}
    reciprocal_near = 0
    disconnected = 0
    reverse_conflicts = Counter()
    cycles = Counter()
    for r in records:
        edges = {(e.source, e.predicate.value, e.target) for e in r.extraction.relationships}
        reciprocal_near += sum(p == "near" and (b, p, a) in edges for a, p, b in edges) // 2
        for predicate in ("left_of", "above", "on", "inside", "in_front_of"):
            reverse_conflicts[predicate] += (
                sum(p == predicate and (b, p, a) in edges for a, p, b in edges) // 2
            )
            adjacency = defaultdict(list)
            for a, p, b in edges:
                if p == predicate:
                    adjacency[a].append(b)
            active, done = set(), set()

            def visit(node):
                if node in active:
                    return True
                if node in done:
                    return False
                active.add(node)
                found = any(visit(other) for other in adjacency.get(node, []))
                active.remove(node)
                done.add(node)
                return found

            cycles[predicate] += int(any(visit(node) for node in list(adjacency)))
        touched = {v for a, _, b in edges for v in (a, b)}
        disconnected += sum(o.object_id not in touched for o in r.extraction.objects)
    properties = [object_properties(o) for o in objects]
    return {
        "frames": len(frames),
        "environments": sorted({f.environment_id for f in frames}),
        "objects": len(objects),
        "relationships": len(relations),
        "unique_raw_labels": len(labels),
        "predicate_counts": dict(Counter(e.predicate.value for e in relations)),
        "empty_graph_frames": sum(not r.extraction.objects for r in records),
        "invalid_or_dangling_edges": 0,
        "images_and_checksums_valid": True,
        "null_attribute_values": sum(a.value is None for a in attributes),
        "explicit_unknown_attribute_values": sum(
            a.value is not None and canonical(a.value) in UNKNOWN for a in attributes
        ),
        "observations_outside_old_parser_vocabulary": sum(
            LABEL_PARENTS.get(canonical(o.label), canonical(o.label)) not in OBJECT_VOCABULARY
            for o in objects
        ),
        "exact_duplicate_image_groups": [ids for ids in groups.values() if len(ids) > 1],
        "extra_exact_duplicate_images": sum(len(ids) - 1 for ids in groups.values()),
        "reciprocal_near_pairs": reciprocal_near,
        "isolated_observations": disconnected,
        "reverse_direction_conflicts": dict(reverse_conflicts),
        "frames_with_predicate_cycles": dict(cycles),
        "generic_object_observations": sum(
            canonical(o.label) in {"object", "item", "small object", "device", "decorative object"}
            for o in objects
        ),
        "normalized_label_mapping": mappings,
        "raw_label_counts": dict(labels.most_common()),
        "vocabulary": graph_vocabulary(records),
        "normalized_empty_labels": sum(not p["label"] for p in properties),
        "preprocessing": [
            "Stable environment/frame/local-object IDs prevent cross-frame object merges.",
            "Preserve captions, raw labels/states and typed attribute JSON alongside normalized search terms.",
            "Omit null/unknown attribute values from search; keep them in raw attribute JSON.",
            "Use stored label/attribute/state vocabulary in the offline query parser.",
            "Retain duplicate images and reciprocal near edges for source fidelity; near queries are undirected.",
        ],
        "limitations": [
            "Structural validity does not establish visual correctness of model extractions; inspect images and extraction review.",
            "No verified timestamps, room assignments, 3D positions or cross-frame physical-object identity are available.",
            "The graph stores frame-local observations; counting observations is not counting distinct household objects.",
            "Image-relative left_of/above and visible proximity do not define a global house map.",
            "Rules parsing can omit location, affordance and activity context; inspect its plan or provide a structured plan.",
            "Sampling and extraction omissions can prevent retrieval of evidence for some original questions.",
        ],
    }
