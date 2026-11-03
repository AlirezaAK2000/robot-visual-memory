"""Derive the Kaggle lock; the Kaggle image supplies PyTorch and its CUDA runtime.

Install this lock with --no-deps; all other transitive requirements stay hashed/pinned.
"""

import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
blocks = re.split(
    r"(?=^[A-Za-z0-9][A-Za-z0-9_.-]*==)", (root / "requirements-core.lock").read_text(), flags=re.M
)
kept = []
for block in blocks:
    name = block.split("==")[0]
    if name in {"torch", "triton"} or name.startswith("nvidia-"):
        continue
    kept.append(block)
(root / "requirements-kaggle.lock").write_text(
    "# Derived by scripts/kaggle_lock.py. Install with --no-deps.\n"
    "# PyTorch, Triton and NVIDIA runtime are provided by the Kaggle image; versions logged at export.\n"
    + "".join(kept)
)
