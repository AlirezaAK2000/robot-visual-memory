"""Frozen pilot vocabulary. Edit only using the development environment, then freeze."""

import re

SYNONYMS = {
    "couch": "sofa",
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
    "coffee table": "table",
    "dining table": "table",
    "armchair": "chair",
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


def object_properties(obj):
    return {
        "label": canonical(obj.label),
        "attributes": sorted({canonical(a.value) for a in obj.attributes if a.value is not None}),
        "states": sorted({canonical(s) for s in obj.states}),
    }


def rules_hash():
    from .io import digest

    return digest(
        {
            "synonyms": SYNONYMS,
            "vocabulary": sorted(OBJECT_VOCABULARY),
            "attributes": sorted(ATTRIBUTES),
            "states": sorted(STATES),
            "version": 1,
        }
    )
