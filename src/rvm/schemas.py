from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Predicate(str, Enum):
    left_of = "left_of"
    above = "above"
    near = "near"
    on = "on"
    inside = "inside"
    in_front_of = "in_front_of"


class Attribute(StrictModel):
    name: str
    value: str | None


class Object(StrictModel):
    object_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    attributes: list[Attribute]
    states: list[str]


class Relation(StrictModel):
    source: str
    predicate: Predicate
    target: str


class Extraction(StrictModel):
    caption: str = Field(min_length=1)
    objects: list[Object]
    relationships: list[Relation]

    @model_validator(mode="after")
    def graph_integrity(self):
        ids = [o.object_id for o in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate frame-local object IDs")
        seen = set()
        for r in self.relationships:
            if r.source not in ids or r.target not in ids:
                raise ValueError("dangling edge (cross-frame edges are forbidden)")
            if r.source == r.target:
                raise ValueError("self relation")
            edge = (r.source, r.predicate, r.target)
            if edge in seen:
                raise ValueError("duplicate edge")
            seen.add(edge)
        return self


class Frame(StrictModel):
    environment_id: str
    episode_history: str
    frame_id: str
    sequence_index: int = Field(ge=0)
    image_relpath: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    source_sha256: str
    image_sha256: str
    dhash: str
    split: Literal["development", "evaluation"]
    # Sequence indexes are not timestamps. RGB-only archive has no verified clock.
    timestamp: float | None = None


class Record(StrictModel):
    frame_id: str
    extraction: Extraction
    cache_key: str


class Question(StrictModel):
    query_id: str
    question: str
    environment_id: str
    episode_history: str
    category: str
    split: Literal["development", "evaluation"]


class Judgment(StrictModel):
    query_id: str
    frame_id: str
    relevance: Literal[0, 1] | None
    source: Literal["human", "silver", "synthetic", "unjudged"]
    status: Literal["confirmed", "proposed", "unjudged"]
    notes: str = ""

    @model_validator(mode="after")
    def valid_status(self):
        if self.status == "unjudged" and self.relevance is not None:
            raise ValueError("unjudged must have null relevance")
        if self.status == "confirmed" and (
            self.source not in {"human", "synthetic"} or self.relevance is None
        ):
            raise ValueError("confirmed judgments require human/synthetic source and binary relevance")
        return self


class Anchor(StrictModel):
    label: str
    attributes: list[str]
    states: list[str]


class PlanRelation(StrictModel):
    source: int = Field(ge=0)
    predicate: Predicate
    target: int | None  # None: some visible neighbor, never an inferred answer object
    direction: Literal["out", "in", "either"]


class QueryPlan(StrictModel):
    anchors: list[Anchor] = Field(max_length=6)
    relationships: list[PlanRelation] = Field(max_length=8)

    @model_validator(mode="after")
    def valid_indexes(self):
        for r in self.relationships:
            if r.source >= len(self.anchors) or (r.target is not None and r.target >= len(self.anchors)):
                raise ValueError("query relation anchor out of bounds")
        return self


class Hit(StrictModel):
    frame_id: str
    score: float
    diagnostics: dict = Field(default_factory=dict)


class RetrievalResult(StrictModel):
    hits: list[Hit]
    diagnostics: dict
