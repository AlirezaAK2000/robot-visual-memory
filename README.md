# Robot Visual Memory: Graph Retrieval vs Caption Embedding Retrieval

A small OpenEQA episodic-memory pilot comparing **exactly two frame retrieval pipelines**:
Neo4j scene-graph retrieval and Qdrant caption embedding retrieval. Both search the same selected
RGB gallery in the question's environment. This is a pipeline comparison: captions and graphs
are different extraction outputs, so results cannot isolate representation alone as the cause.

Default scope: **one HM3D episode**, chosen as the shortest episode among the collected histories.
The notebook first inventories every available HM3D episode and exports a histogram and exact counts.
It partitions the selected sequence into disjoint windows and takes one midpoint per window,
up to **200 frames**. If the episode has at most 200 frames, every frame is retained. Three original
questions (or all, if fewer) are selected for annotation. The model remains `gpt-6-luna`.
Pilot review and evaluation use the same episode: results are **single-episode exploratory**, with
no held-out environment/generalization claim. The two retrieval methods and relevance metrics remain.
No answer generation, answer grading, tracking, world map, simulator, frontend, or third retriever.
Evenly spaced frame sampling prepares the shared gallery; it is not a retrieval baseline.

## Files

```text
notebooks/01_prepare_openeqa_kaggle.ipynb    Kaggle acquisition → gallery → VLM → BGE → annotations → ZIP
notebooks/02_evaluate_retrieval_local.ipynb local clients → rankings → metrics/latency → reports
compose.yaml                             Neo4j, Qdrant, optional official Neo4j MCP profile
Dockerfile.qdrant                         pinned Qdrant plus curl for /readyz health check
config.example.yaml, .env.example         experiment settings and local credentials
pyproject.toml, requirements-*.txt/lock   package metadata and reproducible dependencies
src/rvm/schemas.py                       strict schemas and graph validation
src/rvm/preparation.py                   official acquisition, inspection, deterministic selection
src/rvm/mirror.py                        pinned mirror TAR verification and resumable episode collection
src/rvm/episode_selection.py             episode inventory, histogram, disjoint windows and MP4 previews
src/rvm/extraction.py                    strict VLM requests, cache, budgets, raw provenance
src/rvm/review.py                        input images beside readable pilot captions and graphs
src/rvm/embeddings.py                    local BGE captions/queries, token checks, optional snapshot
src/rvm/annotation.py                    contact sheets, widgets, resumable human qrels
src/rvm/bundle.py, io.py                  integrity, checksums, safe extraction, portable export
src/rvm/normalization.py, planning.py     frozen synonyms and question-only query plans
src/rvm/ingestion.py, retrieval.py        database clients and exactly two retrieval methods
src/rvm/evaluation.py                    metrics, isolation checks, reports and image comparison
src/rvm/fixture.py, cli.py                synthetic fixture and command-line utilities
scripts/import_bundle.py                 import CLI wrapper
scripts/build_notebooks.py               reviewable notebook cell sources
scripts/kaggle_bootstrap.py              embedded GitHub checkout into /kaggle/working itself
scripts/kaggle_lock.py, verify_notebooks.py lock derivation and retained synthetic execution
tests/                                  offline, notebook, optional DB and real-bundle checks
docs/SOURCES.md                          primary documentation and pinned design decisions
docs/VERIFICATION.md                     checks actually run and their limitations
```

## 1. Upload the notebook and run on Kaggle

Download/import `notebooks/01_prepare_openeqa_kaggle.ipynb` from this GitHub repository into Kaggle.
Enable Internet; optionally select a GPU. A CPU works more slowly. No project ZIP, source Dataset
attachment or local Python setup is needed for this preparation phase.

The notebook clones/checks out `https://github.com/AlirezaAK2000/robot-visual-memory.git` **directly
into `/kaggle/working`**, without creating a `robot-visual-memory` subfolder:

```text
/kaggle/working/
  .git/
  config.example.yaml
  pyproject.toml
  requirements-kaggle.txt
  src/
  notebooks/
  scripts/
  rvm-single-episode/       # generated data, previews and checkpoints
```

The Git bootstrap is embedded in the uploaded notebook and uses only the standard library, so it
works before any project modules exist. It uses in-place Git init/fetch/checkout to accommodate
Kaggle's existing notebook/output files. It never forces overwrites, resets files, or deletes outputs.
The default `REPOSITORY_REF='main'` can instead be a tag or commit SHA. The resolved commit is recorded
in `provenance/source-code.json`. Public repository access works without a GitHub token.

Reruns reuse the checked-out commit. Set `UPDATE_REPOSITORY=True` to fetch the requested ref explicitly;
updates refuse tracked local edits or conflicting untracked files. If imported code was updated,
restart the kernel and run again with `UPDATE_REPOSITORY=False`. Changing REPOSITORY_REF also requires
that explicit update. The notebook changes its working directory to PROJECT_ROOT before installation.

`WORK` defaults to `/kaggle/working/rvm-single-episode`.
Time-consuming preparation steps show notebook-friendly progress bars: archive downloads/hashing,
inventory/extraction, image inspection and duplicate checks, gallery preprocessing, video encoding,
annotation-kit creation, VLM extraction, caption embedding and final ZIP/checksum validation. The API
bar counts completed frames, including cache hits, while preserving manifest order; its attempt count
includes retries for completed frames. Streaming directory/TAR scans show counts/rate/elapsed time
because their member totals are unknown. The local evaluation notebook also shows bundle validation
and retrieval-pass progress. Model downloads use Hugging Face's built-in progress reporting.

Use a fresh WORK directory when switching from an older multi-episode export; existing annotations
must never be silently applied to a different gallery.
Do not place secrets under WORK or in YAML/notebook cells. Add `OPENAI_API_KEY` in Kaggle Secrets
and grant notebook access; an environment variable also works. The code never prints/exports keys.

The universal local lock pins/hashes all transitive dependencies. The derived Kaggle lock pins the
same Python packages while deliberately preserving the Kaggle image's PyTorch/CUDA stack; install
it with `--no-deps` as the bootstrap does. Exact Kaggle torch/CUDA/device/package versions are recorded
in provenance. This avoids replacing its working GPU runtime; it does not promise bitwise-identical
GPU/CPU inference across different Kaggle images. Regenerate locks using the command in the core
lock's header, then `python scripts/kaggle_lock.py`.

The notebook keeps the pinned official question metadata. **The original Dropbox RGB link has
expired** (verified 2026-11-03). For Kaggle downloads, the default provider now uses the third-party
[AIGeeksGroup/OpenEQA mirror](https://huggingface.co/datasets/AIGeeksGroup/OpenEQA/tree/main), pinned to
commit `9b127cf58e564fe2468bb140c10b20dd937a97f8`. Its HM3D files are separate episode TARs rather
than the original combined archive. Original Dropbox byte identity is **unverified**; acquisition
provenance records the mirror, revision and each TAR's SHA-256 from pinned LFS metadata. The official
README/MD5 remain reference information and are not used to validate differently packaged mirror TARs.

Choose one route in the notebook configuration:

- **Kaggle mirror download:** set `ALLOW_RGB_DOWNLOAD=True`, `RGB_PROVIDER='huggingface'` and
  `ATTACHED_RGB=None`. This public mirror currently needs no Hugging Face secret.
- **Existing original archive:** set `ATTACHED_RGB=Path('/kaggle/input/<dataset>/open-eqa-hm3d-frames-v0.tgz')`.
  Its documented MD5 is checked. Download permission can remain disabled.
- **Existing extracted data:** set `ATTACHED_RGB` to the root containing `hm3d-v0/<episode>/*-rgb.png`.

The pinned mirror exposes 63 HM3D TARs, totaling 12,563,097,600 bytes. The notebook collects all available
HM3D episodes that match QA metadata before building the frame-count histogram and selecting the
shortest. It downloads one episode TAR, verifies SHA-256, extracts its RGB files, saves a completion
checkpoint and removes that temporary TAR before proceeding. This keeps storage near the complete
PNG collection plus one episode TAR, instead of keeping two complete copies. Reruns skip completed
episodes after checking their expected file sizes; missing outputs trigger that episode's recovery.
Only the selected shortest episode is subsequently decoded/preprocessed and sent to the VLM. ScanNet
is excluded. Check available Kaggle disk space before starting; no RGB data is downloaded on the laptop.

Downloads stream with progress and make up to three attempts for interrupted transfers or HTTP
429/500/502/503/504. Retries start at the stable source URL, rather than reusing a signed CDN URL,
and wait 5 then 10 seconds, or longer when a valid Retry-After header requests it. If the server
requests more than 60 seconds, the downloader stops with instructions to wait and rerun instead
of retrying early. HTTP 401/403/404 errors stop immediately. To recover from a CDN 503, rerun the
same collection cell with the same WORK directory; completed episodes are skipped and safe partials
are retained. A server that stays unavailable can still exhaust the bounded retry limit. Validated
identity-encoded byte ranges can resume; legacy/compressed partials restart safely. HTTP wire
Content-Length is never compared to decoded file size. Links, devices, traversal and excessive
archive expansion are rejected. Do not change the input collection while the notebook is running.

For an attached original archive, inventory scans headers without extracting all images, then
extracts only the shortest episode. Extracted attachments and the materialized mirror collection
are counted in place. Selection uses actual collected frame counts and reports missing episodes;
it does not infer full history coverage or original-archive equivalence from a mirror's file sizes.
If mirror access also fails, attach an existing RGB collection via Add Input. An OpenAI key does
not grant access to expired dataset links.

Before sampling, `review/episode_frame_counts.csv` records exact RGB counts for every collected HM3D
history with QA metadata; `review/episode_frame_count_histogram.png` plots their distribution. Missing
metadata histories are listed separately and excluded from the histogram, never treated as empty
episodes. Selection takes the lowest positive count, breaking ties by history ID. With a partial
attachment, the minimum is explicitly restricted to that input; no claim about unseen episodes.
An optional `environment_ids` list restricts the inventory scope before selecting its minimum.

Only the chosen episode is decoded and checked for image validity, resolutions, exact duplicates,
near-duplicate pairs (64-bit dHash Hamming distance ≤4), and contact sheets. Invalid images stop
preparation rather than silently substituting another episode. No question text, answer, relevance,
or retrieval outcome participates in episode or frame selection.

For N original frames and a cap M<=200, make B=min(N,M) disjoint windows with boundaries
`floor(i*N/B)` for i=0..B. Each window `[start,end)` contributes index `start+(end-start-1)//2`.
The windows cover the full sequence, differ in width by at most one, and produce B unique ordered
frames. `sampling_windows.json` records every boundary, selected original index, filename and frame ID.
The RGB archive has no verified capture timestamps, so these are sequence-index time proxies, not
invented seconds or real-time playback. If N<=200 and M=200, all source frames are kept.

Before any paid extraction, the notebook displays two H.264 MP4s: the selected exported images in
`review/sampled_frames.mp4`, and all frames from the selected original episode in
`WORK/original_episode_preview.mp4`. Original sequence indexes are overlaid, aspect ratio is preserved,
and `preview_fps` controls review speed only. The original preview stays on Kaggle; only the sampled
preview enters the compact ZIP. Review both and set `SAMPLING_REVIEWED=True` before enabling the VLM
pilot. A video review helps spot missing views; it does not certify question coverage.

Frame IDs hash history + original filename; the manifest preserves original sequence indexes,
source/export image hashes, dHash, dimensions and portable image paths. Images are EXIF-transposed,
converted to RGB and resized with aspect ratio preserved (default max side 1024), then JPEG encoded.
RGB histories have no verified timestamps, so timestamps are null rather than fabricated.

## 2. Complete annotations and extraction; download the compact export

The notebook produces `annotation-kit.zip` **before any paid requests**. It contains selected images,
question/manifest metadata, separate reference answers, paginated contact sheets and blank qrels.
Seeded query selection uses question IDs from the one selected episode, independently of outcomes.

Human relevance criterion: **a frame contains visual evidence useful for answering the particular
question, including useful partial evidence when multiple views are needed**. Judge every selected
frame in that query's environment. Reference answers may help annotation; they are not frame-level
ground truth by themselves. Do not derive labels from extracted captions/graphs or their retrievals.

Use the notebook widget (save each page before navigating), or edit `annotation/qrels.csv`:

```csv
query_id,frame_id,relevance,source,status,notes
<original-question-id>,<frame-id>,1,human,confirmed,useful evidence
<original-question-id>,<another-frame>,0,human,confirmed,inspected no useful evidence
<original-question-id>,<uninspected-frame>,,unjudged,unjudged,
```

CSV quoting/newlines in notes are supported. Rerunning preserves matching judgments; a changed
gallery/query selection requires a fresh WORK directory. All-zero fully inspected queries remain
explicit `no-relevant` coverage cases. Silver suggestions, if externally supplied, must use
`source=silver,status=proposed`; they never satisfy a human ground-truth requirement. No automatic
silver annotation is enabled. Human labels are never fabricated.

After video review, set `SAMPLING_REVIEWED=True` and `RUN_PILOT=True`. The extraction pilot processes
only the first **three sampled frames** from the same episode (all if fewer are available). Each
exact input image appears beside its caption, object/attribute/state table and directed relationships,
with expandable structured JSON. The notebook also saves a self-contained `WORK/pilot-review/extractions.html`
and `records.json` so you can download and inspect the preview. Empty/unknown outputs are explicit.
Inspect captions, states, object IDs and spatial edge direction before full extraction. This is image
review on the evaluation episode, not independent development data. Keep evaluation questions/answers
out of prompt tuning and label the final experiment exploratory. One request produces two distinct
fields: a natural-language caption (target ≤180 words) and a frame-local graph. Empty lists/null
attributes allow unknowns; non-detection is not absence. Strict JSON Schema output plus Pydantic
reject duplicate IDs, missing fields, dangling/self/duplicate edges and unknown predicates.

After reviewing, set `PILOT_REVIEWED=True` and `RUN_FULL_EXTRACTION=True`, rerun in order. These flags
are deliberate controls for a paid run, not a requirement to seek additional assistant approval.
Matching pilot results reuse cache entries. Default concurrency is 2, with at most 3 attempts/request.
Requests reserve output tokens against a persistent request/output budget before calling the API.
This guards attempts/reserved output tokens, **not money or all input tokens**. SDK hidden retries
are disabled; capped exponential backoff is implemented explicitly. The cache key includes model,
prompt/schema, image hash, output limit and preprocessing settings. Atomic files and locks prevent
duplicate matching concurrent work. Failed attempts/raw responses and successful normalized records
include usage, timing, prompt/image hashes and model identifiers. No precise currency estimate is
claimed; use the logged actual usage and current pricing if you need an accounting calculation.

The caption is the primary vector text, never graph serialization. BGE embedding inference runs on
Kaggle using CUDA when available, CPU otherwise. Query prefix, revision, pooling, normalization,
dimension, dtype and token budget are exported. Overlong text raises with row IDs/counts; no silent
truncation. Check the extraction pilot's caption token lengths before processing the full gallery. Set
`EXPORT_OFFLINE_MODEL=True` to export only the small model snapshot, not the Hugging Face cache.

Run the final export cell after saving annotations. It validates a fresh extraction of the ZIP and
checks every selected image reference. Download only
`/kaggle/working/rvm-single-episode/robot-visual-memory-compact.zip` through Kaggle Output. No full archive, keys,
unrelated caches or absolute Kaggle paths enter the portable bundle. The bundle contains:

- `images/`, `frames.json`, `selection.json`, `records.json` (captions and graph records).
- `sampling_windows.json`, `review/episode_frame_counts.csv`, histogram PNG/JSON, inventory JSON,
  and the sampled MP4 plus its frame-index/hash provenance. The full original episode preview is not bundled.
- `caption_vectors.npy` (float32) and `vector_frame_ids.json` in manifest order.
- `embedding.json`, optional `model_snapshot/`, `development-freeze.json`.
- `questions.json` with original IDs/categories/history/splits; answers only under `annotation/`.
- Qrels, annotation query IDs, status/review materials and separate reference-answer metadata.
- `provenance/` raw/normalized extraction results, attempts, acquisition/inspection and actual usage.
- `bundle.json` schema/version/provenance manifest and `checksums.json`.

Incomplete human labels are allowed at export and explicitly mark it **annotation-pending**.
Finish them in Kaggle and re-export, or save a completed human CSV separately on the laptop and set
QRELS_OVERRIDE in notebook 02. Keep the immutable exported checksums intact.

## 3. Start databases and import the compact bundle

On your laptop, use Python 3.11 or 3.12. From the repository root, install the local environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-local.txt
python -m pip install --no-deps -e .
```

Install Docker with Compose v2. The deployment pins Neo4j Community 5.26.12, Qdrant 1.16.1, and
official Neo4j MCP 2.0.0. Docker runtime validation is conditional on Docker availability; see the
verification report. Neo4j installs the matching APOC core plugin at first startup (requires network).

```bash
cp .env.example .env
# Edit .env and set a local NEO4J_PASSWORD (do not commit .env).
docker compose up -d --build --wait neo4j qdrant
docker compose ps
mkdir -p data
rvm import ~/Downloads/robot-visual-memory-compact.zip --destination data/robot-visual-memory-compact
rvm verify data/robot-visual-memory-compact
# Idempotency: repeat against the extracted directory, not the same occupied unzip destination.
rvm import data/robot-visual-memory-compact
```

Equivalent importer: `python scripts/import_bundle.py <bundle-directory-or-zip> [--destination ...]`.
`rvm validate <bundle>` performs local integrity checks without databases. Import/verify reports
are written beside the bundle, not inside its checksummed contents. `.env` is loaded automatically.
The importer defaults to the collection recorded in the export (`robot_visual_memory_single_episode`
for this config). An explicit `--collection` overrides it; use the same name in notebook 02.

Neo4j uses unique Environment, Frame, ObjectObservation and Gallery keys. Objects are identified by
the tuple (environment, frame, local ID), never globally by labels or assumed cross-view identity.
Object attributes have both normalized primitive lists for matching and JSON strings for full
named-attribute preservation; no unsupported nested Neo4j property values. Directed `RELATES`
edges carry a validated `predicate`. They cannot cross frame/environment. Images remain files.
Qdrant has one deterministic UUID5 point per frame, normalized cosine vectors, matching caption,
environment and relative image path payloads. Answers/qrels/questions are never indexed.

An exact gallery/model/content contract is enforced. Repeat imports replace each frame's graph
atomically and upsert the same vector IDs. Count, vector alignment, edge isolation and image checks
run afterward. A different export requires a new collection name; changed content for an existing
frame in Neo4j requires a separate Neo4j database/deployment rather than silently changing another
gallery. Community's simple default database is `neo4j`. Use a separate Compose project/ports/volumes
for independent incompatible experiments. Interrupted imports can be rerun; evaluation first verifies
that both stores are complete. Concurrent imports/evaluation are unsupported in this small pilot.

Host Python clients use `bolt://localhost:7687` and `http://localhost:6333`. Inside Compose, MCP uses
`bolt://neo4j:7687`. Published ports bind to 127.0.0.1. Persistent named volumes preserve data;
`docker compose down` stops services and keeps volumes. Removing volumes deletes stored data.

## 4. Optional Codex chat through official Neo4j MCP

Ingestion/evaluation **never call MCP**. The optional official server uses STDIO, read-only tools,
APOC metadata access, no exposed MCP TCP port and telemetry disabled. It is launched per Codex
connection with `run --rm -T`; do not start it as a detached STDIO daemon. Codex owns its lifetime;
closing the connection closes stdin and removes the one-off container. Keep Neo4j running first.

Get your absolute project directory with `pwd`, then add this to your user Codex MCP configuration
(`~/.codex/config.toml`), replacing the example path with that output:

```toml
[mcp_servers.robot_visual_memory]
command = "docker"
args = ["compose", "--project-directory", "/absolute/path/to/robot-visual-memory", "--profile", "mcp", "run", "--rm", "-T", "--no-deps", "neo4j-mcp"]
startup_timeout_sec = 60
tool_timeout_sec = 30
```

`.env` is read by Compose in the project directory; no password is put in TOML or command arguments.
Test launching with `docker compose --profile mcp run --rm -T --no-deps neo4j-mcp` (it waits for MCP
input, so a waiting process is expected). Restart/reconnect the Codex MCP connection after configuration.
The server's read-only setting removes writes for this chat connection; Community credentials are
shared with the local importer and are not a separate database-level authorization boundary.

Example chat request: “Using robot_visual_memory, inspect the graph schema, then return five frame IDs,
environment IDs and image_relpath values containing a kettle, restricted to my chosen environment.”
An illustrative read query for a known frame is:

```cypher
MATCH (f:Frame {frame_id: $frame_id})
RETURN f.environment_id, f.frame_id, f.image_relpath
```

Resolve `image_relpath` against the unpacked bundle root. Ask the local client's image-viewing tool
to open that file. Neo4j MCP returns paths/data; it does **not** automatically render images, mount
your bundle into the chat, or provide a custom frontend.

## 5. Run local retrieval evaluation

```bash
source .venv/bin/activate
python -m ipykernel install --user --name rvm --display-name 'Robot Visual Memory'
jupyter lab notebooks/02_evaluate_retrieval_local.ipynb
```

Select the Robot Visual Memory kernel. Set BUNDLE_PATH to the unpacked compact bundle directory,
or set `RVM_BUNDLE` before launching Jupyter. Use the same collection name as import. Optional
LOCAL_SNAPSHOT points to the exported `model_snapshot/`; otherwise only the small pinned BGE model
downloads from Hugging Face. No original OpenEQA images or image VLM extraction run locally.

The graph planner defaults to deterministic offline rules. GPT planning is optional, uses only
the question, caches by question/prompt/model/rules, uses bounded retries/budgets, and records online
usage/latency. Freeze parser/model choices after the image pilot using `freeze_development` (the
legacy function/filename is retained, with `development_environment=null` in single-episode mode).
The notebook verifies config, prompt/rule and retrieval source hashes. Keep
config changes (including operational model choices) synchronized rather than bypassing this check.

Graph plans support up to six anchors with attributes/states and eight directed connected constraints.
Predicates: `left_of`, `above` (viewer/image coordinates), `near` (symmetric), `on` (support), `inside`
(containment), `in_front_of` (visible occlusion evidence). No metric world geometry is asserted.
Synonyms/singularization are shared by ingestion/planning in normalization.py and frozen before query evaluation.
Rules parse known nouns, nearby attributes/states and nearest relational noun pairs. A single mentioned
anchor plus relation produces a wildcard neighbor: “What is beside the kettle?” never inserts an
expected answer noun. Negated queries are unsupported; non-detection is not absence. Complex syntax,
unlisted labels, disjunctions and pronouns can fail; inspect plans and count failures rather than
claiming a general language parser. GPT is an alternative parser within the same graph pipeline.

Cypher is parameterized, read-only and scoped to Gallery → Frame/environment, with both relationship
endpoints restricted to that exact frame. Scoring: 2 per matched anchor label, 1 per matching
attribute/state on that labeled instance, and 3 per connected relation whose endpoint constraints
match. Relations require attributes/states on their own endpoints; independent partial anchor hits
can still score. No per-label object count inflation. At least one anchor label and the configured
minimum score are required; insufficient matches produce fewer results, never embedding fallback.
Ties sort by frame ID. Failures produce empty results plus explicit error diagnostics. Graph queries
have a 10-second server timeout; direct read queries do not retry automatically. GPT retries are bounded.
Both parsers reject unmentioned object/attribute/state constraints; a schema-valid GPT plan cannot
silently insert a guessed answer as a retrieval constraint.

Caption queries apply the exported query instruction and embed locally. The measured search is
**Qdrant**, exact cosine over the small environment gallery, followed by score/frame-ID tie ordering.
The full environment is searched so ties at the k boundary remain deterministic. A NumPy dot product
exists only in tests for correctness. Distinct frame IDs and environment filters are checked again
by the evaluator. Raw graph/cosine scores are not comparable across methods.

For full MRR the evaluator asks each retriever for up to the entire environment's selected gallery;
top-k metrics are sliced from those same rankings for k=1,3,5. This retrieval depth is identical for
both pipelines. Primary per-query/aggregate/category metrics require an exhaustively judged gallery:

| Metric | Definition |
|---|---|
| Precision@k | Relevant returned frames among first k / k, including missing positions in denominator |
| Recall@k | Relevant returned frames among first k / all relevant frames in this selected gallery |
| MRR | Reciprocal of first relevant rank in the entire returned ranking, zero if never returned |
| nDCG@k | Binary discounted gain, discount 1/log2(rank+1), divided by ideal min(k, relevant-count) gain |

All primary means use the same complete, human-confirmed, evidence-bearing queries for both methods.
No-relevant galleries have P=0, undefined Recall/MRR/nDCG and separate coverage counts. Incomplete or
pooled judgments suppress all four metrics for that query; unjudged frames never become nonrelevant.
If only some selected queries are complete, eligible-subset metrics are explicitly accompanied by
incomplete status and denominator counts; do not present them as complete benchmark results.
Eligible query errors/empty results are scored as empty rankings, with error/empty rates reported.

First-pass and warm passes log total retrieval, embedding/planning and database time separately.
BGE initialization is separate. First-pass timing is a cache proxy, **not fully cold DB/OS latency**;
restart services/kernel and record startup for a true cold-start experiment. Planning cache hits
have no new API usage; cached provenance is not added as new cost. Offline extraction/indexing usage
and import timings are separate from online planning costs. Rules and BGE have no hosted inference fee.

Outputs: per-query CSV, aggregate/category CSV, paired graph-minus-caption differences, coverage and
diagnostics, latency CSV, full rankings/plans/matched constraints JSON, and config/version/checksum
provenance. Side-by-side image displays support review. Suggested manual failure labels: gallery
coverage, caption omission, object/attribute/state extraction, relationship direction, parser/synonym
gap and distractor. Model explanations are debugging evidence, never label generators.

## Checks and a three-day workflow

```bash
pytest -q
rvm fixture artifacts/synthetic
rvm validate artifacts/synthetic
# Optional databases (only synthetic records; cleans up its own synthetic data):
RVM_RUN_DB_TESTS=1 pytest -q -m integration
# Optional real-data schema/image smoke, requires a real compact export:
RVM_REAL_BUNDLE=data/robot-visual-memory-compact pytest -q -m real
```

Offline checks include graph integrity, cross-environment results/qrels, frame-scoped identities,
vector alignment, cache reuse/budget exhaustion, safe archives, missing images, known metric rankings,
unjudged/silver handling, notebook schemas and full synthetic notebook execution. The Qdrant client
test explicitly uses its in-process mode and compares against brute force; it is not a production
benchmark. The optional DB test ingests twice, checks counts/isolation, runs real Cypher/Qdrant, injects
a cross-frame edge to ensure detection, and removes only clearly synthetic fixture records afterward.
Never run synthetic fixtures under the real experiment's collection or interpret their scores as results.

Day 1: attach data, inventory episode lengths, inspect the histogram, select the shortest episode,
review its original/sampled videos and extraction pilot, then freeze settings. Day 2: finish bounded
extraction/embeddings, judge a small exhaustive query subset, export and import the compact gallery.
Day 3: verify database isolation/repeat import, run the exploratory single-episode evaluation,
review failures and export results. If labels/access remain incomplete, report that status and stop
short of unsupported benchmark claims. See docs/VERIFICATION.md for the actual checks run here.
