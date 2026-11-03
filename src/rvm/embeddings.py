"""Only local Hugging Face embeddings. A single contract for captions and queries."""

import warnings
from pathlib import Path

import numpy as np

from .io import atomic_json, digest, read_json


class CaptionEmbedder:
    def __init__(self, settings, *, snapshot=None, cache_folder=None):
        from sentence_transformers import SentenceTransformer

        self.settings = dict(settings)
        if settings["pooling"] != "cls" or settings["dtype"] != "float32" or not settings["normalize"]:
            raise ValueError("Pilot contract requires CLS pooling, normalized float32 vectors")
        if snapshot:
            if read_json(Path(snapshot) / "rvm-model.json") != self.contract():
                raise ValueError("offline snapshot contract differs from bundle")
        self.model = SentenceTransformer(
            str(snapshot) if snapshot else settings["model"],
            revision=None if snapshot else settings["revision"],
            cache_folder=str(cache_folder) if cache_folder else None,
            local_files_only=bool(snapshot),
            trust_remote_code=False,
        )
        self.model.max_seq_length = settings["max_length"]
        if self.model.get_sentence_embedding_dimension() != settings["dimension"]:
            raise ValueError("model dimension differs from configuration")
        pool = self.model[1]
        if not pool.pooling_mode_cls_token or pool.pooling_mode_mean_tokens:
            raise ValueError("unexpected pooling in pinned model")

    def contract(self):
        return {
            k: self.settings[k]
            for k in [
                "model",
                "revision",
                "dimension",
                "max_length",
                "query_instruction",
                "pooling",
                "normalize",
                "dtype",
            ]
        }

    def encode(self, texts, *, query=False):
        prepared = [self.settings["query_instruction"] + t if query else t for t in texts]
        lengths = [len(self.model.tokenizer.encode(t, truncation=False)) for t in prepared]
        over = [i for i, n in enumerate(lengths) if n > self.settings["max_length"]]
        if over:
            warnings.warn(
                f"Embedding truncation refused for rows {over}: lengths {[lengths[i] for i in over]}"
            )
            raise ValueError("text exceeds embedding token budget; revise captions on development data")
        vectors = self.model.encode(
            prepared,
            batch_size=self.settings["batch_size"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        validate_vectors(vectors, len(texts), self.settings["dimension"])
        return vectors

    def export_snapshot(self, target):
        self.model.save(str(target), safe_serialization=True)
        atomic_json(Path(target) / "rvm-model.json", self.contract())


def validate_vectors(vectors, count, dimension):
    if vectors.shape != (count, dimension) or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise ValueError("invalid vector shape, dtype, or finite values")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError("vectors must be unit-normalized")


def embed_bundle(root, settings, *, cache_folder=None):
    import torch
    from .schemas import Record

    root = Path(root)
    frames = read_json(root / "frames.json")
    records = {r["frame_id"]: Record.model_validate(r) for r in read_json(root / "records.json")}
    embedder = CaptionEmbedder(settings, cache_folder=cache_folder)
    vectors = embedder.encode([records[f["frame_id"]].extraction.caption for f in frames])
    np.save(root / "caption_vectors.npy", vectors, allow_pickle=False)
    atomic_json(root / "vector_frame_ids.json", [f["frame_id"] for f in frames])
    atomic_json(
        root / "embedding.json",
        {
            **embedder.contract(),
            "batch_size": settings["batch_size"],
            "caption_hashes": [digest(records[f["frame_id"]].extraction.caption) for f in frames],
            "token_lengths": [
                len(embedder.model.tokenizer.encode(records[f["frame_id"]].extraction.caption))
                for f in frames
            ],
            "device": str(embedder.model.device),
            "truncation_count": 0,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
    )
    if settings.get("export_snapshot"):
        embedder.export_snapshot(root / "model_snapshot")
    return vectors
