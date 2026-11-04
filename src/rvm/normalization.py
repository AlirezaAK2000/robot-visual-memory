"""Deterministic search normalization; raw extraction text is always preserved."""

import re

SYNONYMS = {
    "couch": "sofa",
    "bannister": "handrail",
    "banister": "handrail",
    "stairs": "staircase",
    "settee": "sofa",
    "television": "tv",
    "television set": "tv",
    "refrigerator": "fridge",
    "refrigerators": "fridge",
    "trash can": "bin",
    "garbage can": "bin",
    "teakettle": "kettle",
    "electric kettle": "kettle",
    "cupboard": "cabinet",
    "washbasin": "sink",
    "bookcase": "bookshelf",
    "grey": "gray",
    "wooden": "wood",
    "metallic": "metal",
}
OBJECT_VOCABULARY = frozenset(
    {
        "kettle",
        "chair",
        "sofa",
        "table",
        "tv",
        "fridge",
        "bin",
        "cabinet",
        "sink",
        "bed",
        "lamp",
        "door",
        "window",
        "shelf",
        "bookshelf",
        "picture",
        "mirror",
        "toilet",
        "bathtub",
        "shower",
        "oven",
        "microwave",
        "stove",
        "counter",
        "desk",
        "computer",
        "plant",
        "vase",
        "bottle",
        "cup",
        "plate",
        "bowl",
        "towel",
        "rug",
        "carpet",
        "curtain",
        "wall",
        "ceiling",
        "floor",
        "air conditioner",
        "fan",
        "pillow",
        "blanket",
        "clock",
        "basket",
        "dishwasher",
    }
)
ATTRIBUTES = frozenset(
    {
        "white",
        "black",
        "red",
        "blue",
        "green",
        "yellow",
        "brown",
        "gray",
        "orange",
        "pink",
        "purple",
        "wood",
        "metal",
        "glass",
        "plastic",
        "round",
    }
)
STATES = frozenset({"open", "closed", "on", "off", "empty", "full"})


def canonical(text):
    value = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    value = " ".join(value.split())
    if value in SYNONYMS:
        return SYNONYMS[value]
    if value.endswith("s") and not value.endswith(("ss", "us")) and value[:-1] in OBJECT_VOCABULARY:
        value = value[:-1]
    return SYNONYMS.get(value, value)


LABEL_PARENTS = {"armchair": "chair", "coffee table": "table", "dining table": "table"}
UNKNOWN = {"", "unknown", "none", "null", "not visible", "unclear", "unspecified"}
STATE_ALIASES = {"lit": "illuminated", "partly visible": "partially visible", "dark screen": "screen dark"}


def search_values(value):
    """Keep exact descriptions; add conservative component terms for simple phrases.

    Uncertain alternatives/likeness remain exact phrases. No inferred color, material or state.
    """
    value = canonical(value)
    if value in UNKNOWN:
        return []
    terms = {value}
    if not re.search(r"\b(or|like|possibly|perhaps|appears|maybe)\b", value):
        terms.update(word for word in value.split() if word in ATTRIBUTES)
    return sorted(terms)


def object_properties(obj):
    label = canonical(obj.label)
    return {
        "label": label,
        "raw_label": obj.label,
        "labels": sorted(
            {
                label,
                LABEL_PARENTS.get(label, label),
                " ".join(re.sub(r"[^a-z0-9 ]+", " ", obj.label.lower()).split()),
            }
        ),
        "attributes": sorted(
            {v for a in obj.attributes if a.value is not None for v in search_values(a.value)}
        ),
        "attribute_pairs": sorted(
            {
                canonical(a.name) + "=" + canonical(a.value)
                for a in obj.attributes
                if a.value is not None and canonical(a.value) not in UNKNOWN
            }
        ),
        "states": sorted(
            {STATE_ALIASES.get(canonical(s), canonical(s)) for s in obj.states if canonical(s) not in UNKNOWN}
        ),
        "raw_states": obj.states,
    }


def graph_vocabulary(records):
    properties = [object_properties(o) for r in records for o in r.extraction.objects]
    return {
        "labels": sorted({v for p in properties for v in p["labels"]}),
        "attributes": sorted({v for p in properties for v in p["attributes"]}),
        "states": sorted({v for p in properties for v in p["states"]}),
    }


def rules_hash():
    from .io import digest

    return digest(
        {
            "synonyms": SYNONYMS,
            "vocabulary": sorted(OBJECT_VOCABULARY),
            "attributes": sorted(ATTRIBUTES),
            "states": sorted(STATES),
            "version": 2,
            "states_aliases": STATE_ALIASES,
            "label_parents": LABEL_PARENTS,
            "implementation": __import__("pathlib").Path(__file__).read_text(),
        }
    )
