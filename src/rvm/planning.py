"""Question-only query planning. No imports or arguments expose reference answers/qrels."""

import re
import time

from .extraction import StructuredRunner
from .io import digest
from .normalization import ATTRIBUTES, OBJECT_VOCABULARY, STATES, SYNONYMS, canonical, rules_hash
from .schemas import Anchor, PlanRelation, Predicate, QueryPlan

PLAN_PROMPT = """Interpret a visual retrieval question into anchors and connected relationships.
Use ONLY the supplied question, no expected answer. Anchor labels are simple singular nouns;
attributes/states constrain visible objects. Do not constrain an unknown answer object.
For 'What is beside the kettle?' anchor kettle, near relation, target null, direction either.
For 'What is on the table?' anchor table, on relation, target null, direction in.
Vocabulary: left_of and above are image-relative, on=visible support, inside=containment,
near=symmetric visual proximity, in_front_of=visible occlusion order. Relationships reference
anchor array indexes; target=null denotes an unknown visible neighbor. Use direction out/in/either
relative to source anchor. Use empty lists if no supported visible anchor is present.
No world coordinates, unmentioned answer constraints, negation-as-absence or Cypher code.
"""


def normalize_plan(plan):
    return QueryPlan(
        anchors=[
            Anchor(
                label=canonical(a.label),
                attributes=[canonical(v) for v in a.attributes],
                states=[canonical(v) for v in a.states],
            )
            for a in plan.anchors
        ],
        relationships=plan.relationships,
    )


def validate_grounding(plan, question):
    """Reject invented answer constraints even if an otherwise schema-valid GPT plan emits them."""
    text = " ".join(re.sub(r"[^a-z0-9 ]", " ", question.lower()).split())
    for old, new in sorted(SYNONYMS.items(), key=lambda item: -len(item[0])):
        text = re.sub(r"\b" + re.escape(old) + r"\b", new, text)
    for anchor in plan.anchors:
        for term in [anchor.label, *anchor.attributes, *anchor.states]:
            if not re.search(r"\b" + re.escape(term) + r"s?\b", text):
                raise ValueError("query plan introduced an unmentioned object/attribute/state constraint")
    return plan


class RulesPlanner:
    """Deliberately small deterministic parser, with explicit unsupported-query diagnostics.

    Finds known nouns in text order; attributes attach to the nearest following noun.
    Relations connect the nearest mentioned noun on each side, or a wildcard neighbor
    to the one mentioned anchor. No label-only substitution for unsupported questions.
    """

    def plan(self, question):
        start = time.perf_counter()
        text = " ".join(re.sub(r"[^a-z0-9 ]", " ", question.lower()).split())
        # Longest synonym substitutions first, with word boundaries.
        for old, new in sorted(SYNONYMS.items(), key=lambda x: -len(x[0])):
            text = re.sub(r"\b" + re.escape(old) + r"\b", new, text)
        unsupported = bool(re.search(r"\b(not|never|without|absent|missing)\b", text))
        mentions = []
        occupied = set()
        for term in sorted(OBJECT_VOCABULARY, key=lambda x: (-len(x.split()), x)):
            for match in re.finditer(r"\b" + re.escape(term) + r"s?\b", text):
                if not any(i in occupied for i in range(match.start(), match.end())):
                    mentions.append((match.start(), match.end(), term))
                    occupied.update(range(match.start(), match.end()))
        mentions.sort()
        mentions = mentions[:6]
        anchors = []
        for i, (begin, end, term) in enumerate(mentions):
            preceding = text[mentions[i - 1][1] if i else 0 : begin].split()[-3:]
            following = text[end : mentions[i + 1][0] if i + 1 < len(mentions) else len(text)].split()[:3]
            # Exclude relational 'on' from state attachment; on/off states need GPT for disambiguation.
            anchors.append(
                Anchor(
                    label=canonical(term),
                    attributes=[v for v in preceding if v in ATTRIBUTES],
                    states=[v for v in preceding + following if v in STATES - {"on", "off"}],
                )
            )
        relations = []
        patterns = [
            (r"\b(beside|next to|near)\b", "near", "either"),
            (r"\b(left of)\b", "left_of", "out"),
            (r"\b(right of)\b", "left_of", "in"),
            (r"\b(above|over)\b", "above", "out"),
            (r"\b(below|under)\b", "above", "in"),
            (r"\b(on|on top of)\b", "on", "out"),
            (r"\b(inside|in)\b", "inside", "out"),
            (r"\b(in front of)\b", "in_front_of", "out"),
            (r"\b(behind)\b", "in_front_of", "in"),
        ]
        spans = []
        for pattern, pred, direction in sorted(patterns, key=lambda p: -len(p[0])):
            for m in re.finditer(pattern, text):
                if any(m.start() < b and m.end() > a for a, b in spans):
                    continue
                spans.append(m.span())
                left = [i for i, (_, end, _) in enumerate(mentions) if end <= m.start()]
                right = [i for i, (begin, _, _) in enumerate(mentions) if begin >= m.end()]
                if left:
                    source, target = left[-1], right[0] if right else None
                elif right:
                    source, target = right[0], None
                    direction = {"out": "in", "in": "out", "either": "either"}[direction]
                else:
                    continue
                relations.append(
                    PlanRelation(source=source, predicate=Predicate(pred), target=target, direction=direction)
                )
        plan = QueryPlan(
            anchors=[] if unsupported else anchors, relationships=[] if unsupported else relations[:8]
        )
        return validate_grounding(normalize_plan(plan), question), {
            "parser": "rules",
            "rules_hash": rules_hash(),
            "cache_hit": False,
            "planning_seconds": time.perf_counter() - start,
            "unsupported": unsupported or not anchors,
            "usage": None,
        }


class GPTPlanner:
    def __init__(self, cache_dir, settings):
        self.runner = StructuredRunner(cache_dir, settings)

    def plan(self, question):
        start = time.perf_counter()
        plan, key, provenance = self.runner.run(
            prompt=PLAN_PROMPT,
            content=[{"type": "input_text", "text": question}],
            schema=QueryPlan,
            identity={"question_hash": digest(question), "rules_hash": rules_hash()},
        )
        return validate_grounding(normalize_plan(plan), question), {
            "parser": "gpt",
            "cache_key": key,
            "cache_hit": provenance["cache_hit"],
            "planning_seconds": time.perf_counter() - start,
            "usage": [a.get("usage") for a in provenance["attempts"]] if not provenance["cache_hit"] else [],
            "prompt_hash": digest(PLAN_PROMPT),
            "rules_hash": rules_hash(),
        }
