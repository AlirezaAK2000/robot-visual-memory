from types import SimpleNamespace

import numpy as np
import pytest

from rvm.embeddings import CaptionEmbedder, validate_vectors


def test_query_preprocessing_and_truncation_without_model_download():
    # Exercise the actual encode path with a deterministic tokenizer/model test double.
    embedder = CaptionEmbedder.__new__(CaptionEmbedder)
    embedder.settings = {
        "query_instruction": "query prefix: ",
        "max_length": 8,
        "batch_size": 2,
        "dimension": 3,
    }
    seen = []
    progress_flags = []

    def encode(texts, **kwargs):
        seen.extend(texts)
        progress_flags.append(kwargs["show_progress_bar"])
        assert kwargs["normalize_embeddings"] is True
        return np.array([[1, 0, 0] for _ in texts], dtype=np.float32)

    embedder.model = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda text, **kw: text.split()), encode=encode
    )
    embedder.encode(["a caption"], show_progress=True)
    embedder.encode(["question"], query=True)
    assert seen == ["a caption", "query prefix: question"]
    assert progress_flags == [True, False]  # batch captions show progress; single query stays quiet
    with pytest.warns(UserWarning, match="truncation refused"):
        with pytest.raises(ValueError, match="token budget"):
            embedder.encode(["word " * 10])
    assert len(seen) == 2  # overlong text never reaches model.encode


@pytest.mark.parametrize(
    "array",
    [
        np.array([[np.nan, 0]], dtype=np.float32),
        np.array([[2, 0]], dtype=np.float32),
        np.array([[1, 0]], dtype=np.float64),
    ],
)
def test_bad_vectors(array):
    with pytest.raises(ValueError):
        validate_vectors(array, 1, 2)
