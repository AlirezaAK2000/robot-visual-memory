# Verification actually performed

## Progress reporting and metadata download repair — 2026-11-03

The notebooks now expose progress for directory/archive inventory and extraction, archive hashing,
image inspection and pairwise duplicate checks, gallery preprocessing, preview video encoding,
annotation materials/ZIP creation, concurrent VLM frame extraction, BGE caption batches, provenance
copying, bundle checksums/validation, compact export and retrieval passes. Unknown member totals
remain count/rate displays. Concurrent frame completion updates preserve manifest order; cache hits
and API attempts for completed frames are shown separately.

The reported metadata failure was reproduced against the exact pinned official URL: its gzip
response declared 93,582 wire bytes while Requests yielded 476,360 decoded bytes. The old code
compared these different lengths. Downloads now request identity encoding, handle compressed
responses using wire-length validation, and resume only identity-encoded partials with a valid
validator/range. Legacy/compressed partials restart safely, transient interruptions retry at most
three times, and exhausted safe partials remain resumable on the next run.

- **89 tests passed, 2 skipped** (22.84 seconds); Ruff, compileall and whitespace checks passed.
- Download regressions include an actual local gzip HTTP server using Requests/urllib3, automatic
  resume after truncation, invalid/encoded/changed/unsatisfiable ranges, legacy partials, and retry
  exhaustion followed by successful recovery. No original RGB archive was downloaded.
- A real metadata-first run downloaded and parsed all 1,636 official questions plus the pinned
  data README. A second run reused the same files and returned identical provenance.
- Both notebooks executed successfully in explicit synthetic mode, retaining progress-widget outputs
  and the existing three-frame review/export steps. Tests also verify out-of-order worker progress,
  ordered results, cache/retry counters, failure propagation and embedding progress selection.
- No paid model requests, live Kaggle execution or Docker services were used for these checks.

## Three-frame extraction review — 2026-11-03

The preparation notebook processes only the first three sampled frames for the extraction pilot
(all if fewer). It displays each input image beside the extracted caption, objects, attributes,
states and directed relationships, with expandable JSON. It also saves a self-contained HTML
report and validated records under `WORK/pilot-review`. Full extraction keeps its review gates
and reuses matching pilot cache entries.

- **77 tests passed, 2 skipped** in the pinned environment (19.33 seconds).
- Both notebooks executed in synthetic mode. The preparation output includes three review cards
  and a saved three-record JSON file. Authored synthetic data is clearly labeled, with no API calls.
- Checks cover output/image pairing even when records arrive out of order, exact image bytes and
  hashes, HTML escaping, missing outputs, readable relation direction, unknown attributes and empty
  graphs. Gallery tests verify three pilot IDs and all-frame behavior for a two-frame gallery.
- Ruff and Git whitespace checks passed. No real RGB data, paid VLM requests or live Kaggle run
  were used for this validation.

## GitHub notebook bootstrap — 2026-11-03

The standalone preparation notebook now checks out the public repository directly into
`/kaggle/working`; uploading a project source ZIP is no longer required. Generated data remains
in `/kaggle/working/rvm-single-episode`. The compact selected-data export is still a ZIP.

- **75 tests passed, 2 skipped** in the pinned Python environment (17.34 seconds).
- Local Git fixtures verify a nonempty flat checkout, notebook/checkpoint preservation, explicit
  updates, same-commit reruns, tracked-edit and untracked-collision protection, origin checks,
  tags and full commit SHAs. Notebook checks verify that the current helper is embedded verbatim.
- A real unauthenticated GitHub checkout into a temporary nonempty directory passed: project files
  were flat, an existing notebook survived, and a second run reused the resolved commit.
- Both notebooks executed successfully in explicit synthetic mode; retained copies are under
  `artifacts/verification/single-episode/`. Ruff, compileall and Git whitespace checks passed.
- No live Kaggle session, real RGB processing, or paid model requests were run. The existing
  Docker/real-bundle availability skips remain. Prior ZIP-based checks below are historical.

## Single-episode update — 2026-11-03

The current configuration selects exactly one episode, the shortest collected HM3D RGB history.
The preparation notebook inventories the available collection before selection, exports an original
frame-count histogram/CSV, partitions the chosen sequence into disjoint windows (at most 200
representatives), and displays sampled and original-sequence MP4 previews before any paid extraction.
Results explicitly use `single_episode_exploratory`; no held-out-environment claim is made.

- **68 tests passed, 2 skipped** in the pinned Python environment (16.83 seconds).
- New checks cover directory/ZIP/streaming-tar inventories, deterministic minimum/tie selection,
  missing episodes, duplicate archive entries, histogram counts, positive contiguous disjoint windows,
  the 200-frame cap (including an actual 417-frame synthetic gallery), retention of all short sequences,
  MP4 decoding/frame order, and single-episode bundle/window/video integrity.
- Both notebooks are executed in explicit synthetic mode by `scripts/verify_notebooks.py` and saved
  below `artifacts/verification/single-episode/`. The examples use three authored synthetic episodes
  with 9, 5 and 7 frames; the selected episode has 5 frames. They are **not OpenEQA statistics**.
- Ruff, compileall, and dependency consistency checks passed; universal/Kaggle locks include pinned
  Matplotlib and imageio-ffmpeg dependencies. The latter supplies the ffmpeg video encoder.
- No real RGB archive or extracted collection was provided locally. The **real histogram, actual
  shortest episode ID, and real review videos are generated on Kaggle from the user's attached or
  downloaded collection**, not fabricated in this workspace. No paid model requests were started.
- Docker-backed integration and real-export checks remain skipped for the availability reasons below.

The original-sequence comparison video stays in Kaggle WORK. Only the selected-image MP4, its
index/hash metadata, the inventory/histogram and window map enter the compact export. Video FPS is
explicit review playback speed; source capture timestamps are unknown.

## Initial multi-episode verification — 2026-11-02

Verified on 2026-11-02 in this workspace. Source notebooks deliberately retain empty execution
counts; executed **synthetic-only** copies are separate artifacts, not real benchmark runs.

| Check | Result |
|---|---|
| Full pytest suite, isolated Python 3.12 environment | **48 passed, 2 skipped**, 9.30 seconds |
| Notebook JSON schemas and code-cell parsing | Passed for both notebooks |
| Both notebooks, ordered clean-kernel synthetic execution | Passed; output copies retained |
| Pydantic graph/manifest/qrel validation | Passed, including deliberate corruptions |
| Known-ranking P@k/Recall/MRR/nDCG tests | Passed, including missing positions and zero-relevant cases |
| Unjudged/silver/partial qrels | Passed; incomplete metrics remain null |
| Wrong-environment/unknown/duplicate retrieval results | Correctly rejected |
| Qdrant official Python client in-process fixture | Passed filtering, repeat upsert, deterministic IDs and brute-force comparison |
| Cache reuse/invalidation, persistent budget, raw failure checkpoints | Passed using an explicit fake API client |
| Safe archive traversal/link/expansion and resumable HTTP contracts | Passed without downloading the original RGB archive |
| Missing images, vector-row/caption-source mismatch and ZIP fresh extraction | Passed, including deliberate corruptions |
| Frozen development settings and portable path sanitization | Passed |
| Ruff and Python compileall | Passed |
| Dependency consistency | `uv pip check`: all 139 installed project packages compatible |
| Compose official JSON-schema validation | Passed against the fetched compose-spec schema |
| Container tag existence / multiarchitecture manifest digests | All three verified through Docker Registry API |
| Actual Docker Compose startup, Neo4j Cypher execution, Neo4j MCP handshake | **Not run: Docker executable/application unavailable** |
| Real OpenEQA compact-bundle smoke | **Skipped: no real selected-image export supplied** |
| Paid VLM extraction / optional GPT query planning | **Not run; no paid requests started** |
| Real BGE weights and offline snapshot round trip | **Attempted, blocked by Hugging Face CDN timeout/DNS failure** |

The first notebook kernel attempt was blocked by sandbox restrictions on localhost sockets. The
suite and retained execution then passed with local kernel sockets permitted. Initial dependency
downloads also timed out; targeted retries completed the pinned environment, including PyTorch 2.9.1.
The BGE network failure was not substituted with synthetic embeddings in a real pipeline. Tests use
explicit synthetic vectors/test doubles and make no model-quality or OpenEQA performance claims.

Published images verified and pinned:

```text
neo4j:5.26.12-community
sha256:9f75e8df4325a24f00fdd7a8c0bcce650a58375049b1058e496e8b43d6c36b37
qdrant/qdrant:v1.16.1
sha256:db1c735496dfa982ef27576a17b624e48e6b46a140bcdc2ac34e39d186204ef5
neo4j/mcp:v2.0.0
sha256:3b65fa5a096e8ac547c3f22be074e8fdd108b505c394d664e4d55e3d5981d58d
```

Registry verification caught that `neo4j/mcp:2.0.0` does not exist; the Compose service uses the
verified **v2.0.0** tag. Static/schema/registry checks do not establish live service connectivity.
The Docker-backed test remains provided and must be run on a Docker-equipped machine before
claiming the deployment is runtime-tested.

Review artifacts:

- `artifacts/rvm-source.zip`: Kaggle-ready package, configuration, dependency locks and source notebooks.
- `artifacts/synthetic-fixture.zip`: tiny drawings/authored graphs/vectors/qrels, marked synthetic-test-only.
- `artifacts/verification/01_prepare_openeqa_kaggle.synthetic-executed.ipynb`.
- `artifacts/verification/02_evaluate_retrieval_local.synthetic-executed.ipynb`.
- `artifacts/verification/evaluation-smoke/`: **synthetic replay** result files for report-format inspection.

The paths above record the initial verification run. Current single-episode execution copies and
synthetic histogram/video outputs are under `artifacts/verification/single-episode/`.

Reproduce local checks:

```bash
source .venv/bin/activate
pytest -q
ruff check src scripts tests
python -m compileall -q src scripts tests
python scripts/verify_notebooks.py
```

Complete the external checks when their inputs become available:

```bash
docker compose up -d --build --wait neo4j qdrant
RVM_RUN_DB_TESTS=1 pytest -q -m integration
RVM_REAL_BUNDLE=data/robot-visual-memory-compact pytest -q -m real
```

The integration test uses an isolated synthetic Qdrant collection and refuses to reuse preexisting
synthetic graph frames. It imports twice, verifies graph/vector counts and isolation, runs Cypher,
injects a cross-frame edge to check detection, and cleans up only its fixture nodes/collection.
No real labels or scores have been fabricated; no full dataset has been downloaded on the laptop.
