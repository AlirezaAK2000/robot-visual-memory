# Primary sources and fixed choices

Verified 2026-11-02. Runtime smoke tests are a separate matter; see VERIFICATION.md.

- [OpenEQA repository](https://github.com/facebookresearch/open-eqa) and
  [official data instructions](https://github.com/facebookresearch/open-eqa/blob/main/data/README.md).
  Pinned revision `cfa3fce4595c1622bb2f8a38ae2ca9aae9eb685b`. The prepared RGB link is third-party
  Dropbox, approximately 12 GB; documented MD5 `286aa5d2fda99f4ed1567ae212998370`.
  The acquisition code resolves the link from that pinned README instead of inventing image URLs.
  Official extraction source names RGB frames `*-rgb.png`; QA metadata uses `question_id`,
  `episode_history`, `question`, `answer`, `category`. Image availability/host access is not guaranteed.
- **RGB source update, 2026-11-03:** the original Dropbox shared link returns a "Link Expired" page;
  `dl=1` and `raw=1` do not restore access, and the content host returns HTTP 403. Kaggle downloads
  now use the third-party [AIGeeksGroup/OpenEQA episode archive mirror](https://huggingface.co/datasets/AIGeeksGroup/OpenEQA),
  pinned to `9b127cf58e564fe2468bb140c10b20dd937a97f8`. Its public API lists 63 HM3D TARs
  (12,563,097,600 bytes). Each TAR has a pinned LFS SHA-256, checked before extraction. This is
  different packaging; byte equality/completeness relative to the unavailable original archive
  is unverified. The histogram describes the actual collected mirror frames. Official QA metadata
  remains pinned to OpenEQA. Only small metadata/header probes were performed locally, not a full
  RGB mirror download. [Dropbox's documented download parameters](https://help.dropbox.com/share/force-download)
  change preview/download behavior but do not revive an expired shared link.
- [GPT-6 Luna model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna)
  establishes image input, Responses endpoint, structured outputs and Batch support.
  The pilot default is `gpt-6-luna` (updated 2026-11-03); the model remains configurable.
  [Structured output guide](https://developers.openai.com/api/docs/guides/structured-outputs)
  documents strict JSON Schema formats and Pydantic parsing. This project preserves the raw response
  before local validation; refusals/incomplete output fail explicitly. No current-price assumption.
- [BGE-small model card](https://huggingface.co/BAAI/bge-small-en-v1.5),
  [revision API](https://huggingface.co/api/models/BAAI/bge-small-en-v1.5), and
  [pinned pooling configuration](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/1_Pooling/config.json).
  MIT license; revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`; 384 dimensions;
  512-token sequence limit; CLS pooling; normalized vectors; query-only instruction
  `Represent this sentence for searching relevant passages: `. Passage captions have no prefix.
  The local embedder validates the dimension/pooling, rejects over-budget text and exports the contract.
- [Qdrant documentation](https://qdrant.tech/documentation/),
  [filtering](https://qdrant.tech/documentation/concepts/filtering/), and
  [points](https://qdrant.tech/documentation/concepts/points/) establish environment payload filters,
  vector search and integer/UUID point IDs. The pilot uses deterministic UUID5 IDs and cosine distance.
- [Official Neo4j MCP documentation](https://neo4j.com/docs/mcp/current/) and
  [official release source](https://github.com/neo4j/mcp/tree/v2.0.0).
  Verified release 2.0.0 configuration source: `NEO4J_MCP_URI`, `USERNAME`, `PASSWORD`, `DATABASE`,
  `TRANSPORT_MODE=stdio`, `READ_ONLY=true`, `TELEMETRY=false` (all with `NEO4J_MCP_` prefix).
  The registry publishes multiarchitecture `neo4j/mcp:v2.0.0` (the `v` is required). All three base
  image manifest digests were verified against the registry and pinned. Use the official image directly;
  no Dockerfile.mcp or custom MCP server is needed. APOC metadata procedures are enabled on Neo4j.
- [Codex MCP setup](https://learn.chatgpt.com/docs/extend/mcp) documents STDIO command/args configuration.
  The Compose profile is a per-client STDIO process, not a detached network service.

Dependency versions are deliberately pinned rather than selected as “latest.” Lock resolution has a
fixed cutoff recorded in its generated header. The small Hugging Face model is the only embedding
provider; there are no hosted embedding API calls in this project.

Single-episode preparation uses the documented [Matplotlib histogram API](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html)
and [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) for standalone histogram PNGs and H.264
preview videos. The latter provides platform wheels containing ffmpeg; preview FPS is explicitly
playback speed and does not supply missing recording timestamps. No second image dataset is used.
