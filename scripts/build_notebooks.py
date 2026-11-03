"""Generate readable notebooks from reviewable cell sources. No notebook execution here."""

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip() + "\n"}


def code(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": dedent(text).strip() + "\n",
    }


def save(name, cells):
    for i, cell in enumerate(cells):
        cell["id"] = f"cell-{i:03d}"
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "cells": cells,
    }
    path = ROOT / "notebooks" / name
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1) + "\n")


save(
    "01_prepare_openeqa_kaggle.ipynb",
    [
        md("""
    # Robot Visual Memory — 1. Prepare OpenEQA on Kaggle

    Build one shared RGB candidate gallery, two independent VLM outputs per frame, local BGE
    embeddings, and human retrieval judgments. This notebook never evaluates generated answers.
    Inventory the collected HM3D histories, plot their original frame-count distribution, and choose
    the shortest episode (ties by history ID). Use only that episode for this exploratory pilot.
    Split its ordered sequence into disjoint windows and keep one midpoint per window, at most 200
    frames. When it has 200 or fewer frames, keep every frame. Selection never reads QA answers.
    A single-episode pilot has no held-out environment; review and results are labeled accordingly.

    Upload this notebook to Kaggle, enable Internet, and optionally select a GPU. The bootstrap
    checks out the GitHub project directly into `/kaggle/working`, with no nested repository folder.
    No source ZIP or source Dataset attachment is needed. The original Dropbox RGB link has expired;
    the default download provider is a pinned third-party Hugging Face mirror of episode PNG TARs.
    Mirror data is about 12.6 GB; original Dropbox archive byte identity is unverified.
    Attached archives/extracted datasets are also supported. You need temporary disk for compressed
    data plus extraction; this implementation extracts only selected histories' RGB members.
    Read the source dataset's terms before redistribution; keep imagery/artifacts private as appropriate.

    Slow steps show notebook-friendly progress bars: downloads and archive hashing, inventory and
    extraction, image checks/preprocessing, video encoding, API frame completion, caption embedding,
    and export validation/ZIP creation. Streaming archive/directory scans have no known member total,
    so they show counts, elapsed time and rate rather than an estimated percentage.
    """),
        code("""
    # Configuration — edit this cell before running; no paid requests are enabled by default.
    import os
    from pathlib import Path
    SMOKE = os.getenv("RVM_NOTEBOOK_SMOKE") == "1"  # synthetic offline CI only, never benchmark results
    PROJECT_ROOT = Path('/kaggle/working') if Path('/kaggle').exists() else Path.cwd()
    REPOSITORY_URL = 'https://github.com/AlirezaAK2000/robot-visual-memory.git'
    REPOSITORY_REF = 'main'  # a branch, tag, or commit SHA; resolved commit is recorded in provenance
    UPDATE_REPOSITORY = False  # reruns reuse the checkout; True explicitly fetches the ref again
    WORK = Path(os.getenv("RVM_WORK", str(PROJECT_ROOT/'rvm-single-episode') if Path('/kaggle').exists() else str(PROJECT_ROOT/'artifacts/kaggle-single-episode'))).resolve()
    CONFIG_FILE = None  # defaults to PROJECT_ROOT/config.example.yaml; optional attached edited YAML
    ATTACHED_RGB = None  # Path('/kaggle/input/my-rgb/open-eqa-hm3d-frames-v0.tgz') or extracted root
    ALLOW_RGB_DOWNLOAD = False  # enable only inside Kaggle with enough disk
    RGB_PROVIDER = 'huggingface'  # pinned episode TAR mirror; original 'dropbox' link has expired
    SAMPLING_REVIEWED = False  # review the sampled and original preview videos before paid extraction
    RUN_PILOT = False  # first three sampled frames (all if fewer); needs Kaggle Secrets/env key
    PILOT_REVIEWED = False  # set True AFTER reviewing the pilot output and prompt
    RUN_FULL_EXTRACTION = False  # explicitly enable after pilot review; capped persistent budget applies
    SHOW_ANNOTATOR = True
    EXPORT_OFFLINE_MODEL = False
    print('Mode:', 'SYNTHETIC SMOKE — NOT BENCHMARK' if SMOKE else 'real OpenEQA preparation')
    """),
        md("""
    ## Bootstrap
    Clone/check out the project **directly in `/kaggle/working`**. Project paths such as `src/`,
    `notebooks/` and `config.example.yaml` sit immediately under that directory. If Kaggle has already
    created notebook/output files, the bootstrap uses Git init/fetch/checkout in place. It never
    forces overwrites or deletes outputs. Reruns keep the first resolved commit unless you explicitly
    set UPDATE_REPOSITORY=True; updates refuse tracked local edits and conflicting untracked files.
    Repository access must be available from Kaggle; public access works without a GitHub token.
    The actual commit SHA is saved in export provenance.

    Install the reusable package. Kaggle keeps its supplied PyTorch
    GPU runtime; exact realized package versions are recorded in the export. On CPU the same code runs.
    Never put a key in a notebook cell or export directory. Reruns use the same WORK directory for caches.
    """),
        code(
            (ROOT / "scripts/kaggle_bootstrap.py").read_text()
            + dedent("""
    import sys
    if SMOKE:
        # Offline verification explicitly uses the local source tree and never clones/downloads.
        SOURCE = next((p for p in [Path.cwd(), Path.cwd().parent] if (p/'pyproject.toml').exists()), None)
        if SOURCE is None:
            raise RuntimeError('Synthetic smoke mode requires the local project root')
        source_provenance = {'synthetic_smoke': True, 'repository_url': REPOSITORY_URL, 'commit': None}
    else:
        SOURCE = PROJECT_ROOT.resolve()
        source_provenance = checkout_repository(REPOSITORY_URL, REPOSITORY_REF, SOURCE, update=UPDATE_REPOSITORY)
        if (source_provenance['previous_commit'] is not None
            and source_provenance['previous_commit'] != source_provenance['commit']
            and any(name == 'rvm' or name.startswith('rvm.') for name in sys.modules)):
            raise RuntimeError('Repository updated. Restart the kernel, set UPDATE_REPOSITORY=False, and run in order to load the new code.')
    SOURCE = SOURCE.resolve()
    os.chdir(SOURCE)
    WORK.mkdir(parents=True, exist_ok=True)
    print('Project root:', SOURCE)
    print('Repository commit:', source_provenance['commit'], 'reused:', source_provenance.get('reused_checkout', False))
    if not SMOKE:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-deps', '-r', 'requirements-kaggle.txt'], cwd=SOURCE)
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-deps', '-e', '.'], cwd=SOURCE)
        import torch
        if int(torch.__version__.split('.')[0]) < 2:
            raise RuntimeError('Use a Kaggle image with PyTorch 2.x; the notebook preserves its CUDA runtime')
        print('Torch runtime:', torch.__version__, 'CUDA available:', torch.cuda.is_available())
    sys.path.insert(0, str(SOURCE/'src'))
    from rvm.io import load_config, read_json, atomic_json, digest, export_zip
    from rvm.schemas import Frame, Question, Record
    config = load_config(CONFIG_FILE or SOURCE/'config.example.yaml')
    config['dataset']['input_path'] = str(ATTACHED_RGB) if ATTACHED_RGB else config['dataset']['input_path']
    config['dataset']['allow_download'] = ALLOW_RGB_DOWNLOAD
    config['dataset']['rgb_provider'] = RGB_PROVIDER
    config['embedding']['export_snapshot'] = EXPORT_OFFLINE_MODEL
    BUNDLE = WORK/'compact-bundle'
    atomic_json(BUNDLE/'provenance/source-code.json', source_provenance)
    print('Source ready; bundle:', BUNDLE)
    """)
        ),
        md("""
    ## Metadata first; inventory every collected episode before choosing one
    Question metadata and the original archive reference/MD5 come from the pinned official README.
    With RGB_PROVIDER='huggingface', mirror TARs use a separately pinned commit and per-TAR SHA-256
    checks from its LFS metadata. This is different archive packaging; we cannot prove equality with
    the expired original Dropbox archive. Acquisition provenance records the actual provider.
    Access failures stop with the exact Add Input route. No ScanNet, depth, meshes or simulator.
    The source archive stays on Kaggle. HTTP downloads retry interrupted streams up to three times
    and resume only validated identity-encoded byte ranges. Compressed responses are decoded without
    comparing wire Content-Length to decoded file size. Legacy/compressed partials automatically
    restart safely; you do not need to delete the failed metadata partial manually.
    Archive inventory streams member headers without
    extracting all episodes. It needs a second archive scan to extract only the shortest episode.
    The mirror instead collects PNG episodes one TAR at a time, verifies/extracts each and removes
    its temporary TAR, avoiding two full collection copies. Per-episode checkpoints survive reruns.
    All available HM3D episodes with QA metadata are collected before computing their frame counts;
    only the shortest episode is subsequently preprocessed and sent to the VLM. No ScanNet download.
    For an attached extracted directory we count the RGB files in place. Missing metadata episodes
    are listed separately: the minimum is over the **collected** episodes, not unseen/missing data.
    """),
        code("""
    import pandas as pd
    from tqdm.auto import tqdm
    from IPython.display import display, Image as DisplayImage, Video
    from rvm.preparation import (metadata_first, select_environments, locate_rgb_source, acquire_rgb,
                                 discover_rgb, inspect_images, prepare_gallery, contact_sheet)
    from rvm.episode_selection import inventory_episodes, save_inventory, make_preview_video
    if config['dataset'].get('environment_selection') != 'smallest' or config['dataset']['environments'] != 1:
        raise ValueError('This notebook requires environment_selection=smallest and environments=1')
    if SMOKE:
        from rvm.fixture import create_episode_collection
        source, qa_rows = create_episode_collection(WORK/'synthetic-collection')
        archive_info = {'synthetic': True}
    else:
        qa_rows, archive_info = metadata_first(config, WORK)
        display(pd.DataFrame(qa_rows).assign(dataset=lambda d: d.episode_history.str.split('/').str[0])
                .groupby(['dataset', 'category']).size().rename('questions').to_frame())
        print('Metadata histories:', len({r['episode_history'] for r in qa_rows}))
        source = locate_rgb_source(config, WORK, archive_info, qa_rows=qa_rows)
        if 'rgb_acquisition' in archive_info:
            print('RGB provider:', archive_info['rgb_acquisition']['provider'])
            print('Mirror commit:', archive_info['rgb_acquisition']['revision'])
            print('Original Dropbox archive byte identity is unverified; actual mirror counts follow.')
    inventory, inventory_summary = inventory_episodes(source, qa_rows, config)
    histogram = save_inventory(inventory, inventory_summary, BUNDLE/'review', synthetic=SMOKE)
    display(pd.DataFrame(inventory).sort_values(['frame_count','environment_id']).reset_index(drop=True))
    display(DisplayImage(filename=str(histogram)))
    print('Inventory checkpoint:', inventory_summary['collected_episodes'], 'collected episodes')
    if inventory_summary['missing_episodes']:
        print('Missing from this input:', inventory_summary['missing_episodes'])
        print('Shortest selection is restricted to collected episodes; no global-minimum claim')
    environments = select_environments(qa_rows, config, inventory=inventory)
    print('Selected shortest episode:', environments[0])
    """),
        md("""
    ## Extract only the shortest episode and sample disjoint sequence windows
    One deterministic midpoint is retained per contiguous nonoverlapping window. Windows cover the
    whole ordered history, have nearly equal widths, and never produce more than 200 selected frames.
    Prepared RGB histories have no verified timestamps, so these are sequence-index windows; we do
    not invent second-based timestamps or metric motion. If N<=200, all N frames are preserved.
    The exact window boundaries and chosen original indexes are saved in `sampling_windows.json`.
    """),
        code("""
    rgb_root = acquire_rgb(config, WORK, archive_info, environments, source=source)
    paths_by_env = discover_rgb(rgb_root, environments)
    counted = next(row['frame_count'] for row in inventory if row['environment_id'] == environments[0])
    if len(paths_by_env[environments[0]]) != counted:
        raise ValueError('Extracted RGB count differs from inventory; input changed or extraction is stale')
    inspection = inspect_images(paths_by_env)
    display(pd.DataFrame(inspection))
    if any(row['invalid'] for row in inspection):
        raise ValueError('Invalid source images in the shortest episode; inspect/repair data, no silent substitution')
    frames, questions = prepare_gallery(paths_by_env, qa_rows, environments, config, BUNDLE)
    atomic_json(BUNDLE/'provenance/acquisition.json', archive_info)
    atomic_json(BUNDLE/'provenance/inspection.json', inspection)
    assert len(environments) == 1 and 1 <= len(frames) <= 200
    display(pd.DataFrame([f.model_dump() for f in frames]).groupby(['environment_id','split']).size().to_frame('selected'))
    display(pd.DataFrame([q.model_dump() for q in questions]).groupby(['split','category']).size().to_frame('questions'))
    for env in environments:
        display(contact_sheet([f for f in frames if f.environment_id == env][:12], BUNDLE, title=env))
    print('Gallery checkpoint:', len(frames), 'images; aspect-preserving JPEG preprocessing recorded')
    """),
        md("""
    ## Review the sampled video before extraction
    The first MP4 uses the actual exported sampled images, in original order. The second shows all
    frames of this one episode for comparison and stays outside the compact bundle. Frame indexes
    are overlaid. Playback FPS is a configurable review speed, not a claim about recording timing.
    Inspect both videos for missing views; visual review cannot guarantee that every question remains
    answerable. Do not change the sampling using reference answers or evaluation questions.
    Set `SAMPLING_REVIEWED=True` only after review, then enable the VLM pilot when ready.
    """),
        code("""
    sampled_video = make_preview_video([BUNDLE/f.image_relpath for f in frames], BUNDLE/'review/sampled_frames.mp4',
                                      sequence_indexes=[f.sequence_index for f in frames],
                                      fps=config['dataset']['preview_fps'], synthetic=SMOKE)
    original_video = make_preview_video(paths_by_env[environments[0]], WORK/'original_episode_preview.mp4',
                                       fps=config['dataset']['preview_fps'], synthetic=SMOKE)
    print('Sampled gallery:', len(frames), 'of', counted, 'frames')
    print('All source frames retained' if len(frames) == counted else 'Review sampling loss against the original sequence below')
    display(Video(str(sampled_video), embed=True, width=720))
    display(Video(str(original_video), embed=True, width=720))
    print('Sampled video exported in bundle/review; original video remains in WORK only')
    """),
        md("""
    ## Export annotation materials immediately
    OpenEQA answers are **not frame qrels**. Exhaustively judge each selected query's gallery within
    its environment. Useful partial visual evidence counts. Judged 0 and unjudged are different.
    All-zero fully inspected galleries are coverage failures, never silently dropped. Humans may
    open `annotation/reference_answers.json`; retrieval code never loads it. No silver labels are
    generated here. Silver proposals, if added externally, must use `source=silver,status=proposed`.

    The early ZIP contains selected images, question IDs, separate answers, CSV templates and paginated
    review sheets. It is an **annotation kit**, not an ingestible finished bundle. Download it even if
    extraction is postponed. Save each widget page before switching. Reopening resumes from CSV.
    """),
        code("""
    from rvm.annotation import create_materials, annotation_widget, load_qrels, annotation_status
    annotation_queries, judgments = create_materials(BUNDLE, frames, questions, config)
    if SMOKE:
        from rvm.fixture import complete_prepared_fixture
        complete_prepared_fixture(BUNDLE)
    # Explicit allowlist; no caches, keys or original archive can enter the review kit.
    import zipfile
    KIT = WORK/'annotation-kit.zip'
    with zipfile.ZipFile(KIT, 'w', zipfile.ZIP_DEFLATED) as z:
        kit_paths = [p for p in sorted(BUNDLE.rglob('*')) if p.is_file()]
        for path in tqdm(kit_paths, desc='Write annotation kit', unit='file'):
            rel = path.relative_to(BUNDLE)
            if path.is_file() and (rel.parts[0] in {'images', 'annotation', 'review'} or rel.as_posix() in {'frames.json','questions.json','selection.json','sampling_windows.json'}):
                z.write(path, rel)
    print('Annotation kit available now:', KIT)
    if SHOW_ANNOTATOR and not SMOKE:
        display(annotation_widget(BUNDLE, frames, questions))
    """),
        md("""
    ## Three-frame extraction preview — inspect inputs and outputs together
    One strict structured request per frame produces a natural caption and a validated graph.
    First process only the first three sampled frames (all if fewer than three are available).
    Display each exact input image beside its caption, object/attribute/state table and readable
    relationships. Expand the JSON only when needed. A standalone HTML report and the validated
    pilot records are also saved in WORK/pilot-review for downloading/reopening.
    Inspect hallucinations, missing states, relation direction and verbosity on these images
    from this same episode. This review is not a separate held-out development environment.
    Spatial `left_of/above` are viewer-relative; `on/inside` require visible evidence; no metric
    location or cross-frame object identity is claimed. Unknown values and empty graphs are allowed.
    Changing prompt/model/preprocessing invalidates matching cache entries. Actual usage and failed
    attempts are logged; the guard limits request attempts and reserved output tokens, not currency.
    The extraction bar advances as frames finish, including cache hits. Its API-attempt counter covers
    completed frames (including their retries), so it is not a live count of in-flight requests.
    """),
        code("""
    from rvm.extraction import StructuredRunner, extract_frames, EXTRACTION_PROMPT, usage_summary
    from rvm.io import file_hash
    CACHE = WORK/'extraction-cache'
    selection = read_json(BUNDLE/'selection.json')
    preprocessing = selection.get('preprocessing', {'synthetic': True})
    pilot_frames = [f for f in frames if f.frame_id in selection['pilot_frame_ids']]
    assert len(pilot_frames) == min(3, len(frames))
    if SMOKE:
        pilot_records = [Record.model_validate(r) for r in read_json(BUNDLE/'records.json')
                         if r['frame_id'] in {f.frame_id for f in pilot_frames}]
    else:
        if not SAMPLING_REVIEWED:
            raise RuntimeError('Review the sampled/original videos above, then set SAMPLING_REVIEWED=True before paid extraction')
        if not RUN_PILOT:
            raise RuntimeError('Annotation kit and videos are ready. Set RUN_PILOT=True to process the three-frame preview (all if fewer).')
        runner = StructuredRunner(CACHE, config['extraction'])
        pilot_records = extract_frames(pilot_frames, BUNDLE, runner, preprocessing=preprocessing, show_progress=True)
    from IPython.display import HTML
    from rvm.review import extraction_review_html
    preview = extraction_review_html(pilot_frames, pilot_records, BUNDLE,
                                    model=config['extraction']['model'], synthetic=SMOKE)
    REVIEW = WORK/'pilot-review'
    REVIEW.mkdir(parents=True, exist_ok=True)
    (REVIEW/'extractions.html').write_text('<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Pilot extraction review</title></head><body>' + preview + '</body></html>', encoding='utf-8')
    atomic_json(REVIEW/'records.json', [r.model_dump(mode='json') for r in pilot_records])
    display(HTML(preview))
    print('Review report:', REVIEW/'extractions.html')
    print('Review all three image/output pairs before setting PILOT_REVIEWED=True. Full extraction remains separately gated.')
    if not SMOKE:
        from rvm.embeddings import CaptionEmbedder
        print('Loading BGE model; Hugging Face displays download progress for uncached files...')
        pilot_embedder = CaptionEmbedder(config['embedding'], cache_folder=WORK/'hf-cache')
        pilot_embedder.encode([r.extraction.caption for r in pilot_records], show_progress=True)
        print('Pilot caption tokens:', [len(pilot_embedder.model.tokenizer.encode(r.extraction.caption)) for r in pilot_records])
    print('Pilot checkpoint: review these records before enabling full extraction')
    """),
        md("""
    ## Freeze the pilot and process the shared gallery
    After review, set PILOT_REVIEWED and RUN_FULL_EXTRACTION to True and rerun from configuration.
    Freeze the settings after image review; keep evaluation questions/answers out of extraction tuning.
    Matching pilot extractions are reused. Results are single-episode exploratory, not held-out results.
    Errors stop this step; successful atomic checkpoints survive reruns. No failed frame is silently
    removed from the common gallery. Review actual usage before raising the persistent budget.
    """),
        code("""
    from rvm.evaluation import freeze_development
    if not SMOKE:
        if not (PILOT_REVIEWED and RUN_FULL_EXTRACTION):
            raise RuntimeError('Review the extraction pilot, then explicitly enable PILOT_REVIEWED and RUN_FULL_EXTRACTION')
        freeze_development(config, BUNDLE/'development-freeze.json', development_environment=None)
        records = extract_frames(frames, BUNDLE, runner, preprocessing=preprocessing, show_progress=True)
        atomic_json(BUNDLE/'records.json', [r.model_dump(mode='json') for r in records])
        atomic_json(BUNDLE/'provenance/extraction-usage.json', usage_summary(CACHE))
        # All attempt records, including failed validation responses; never request headers/API keys.
        for attempt in tqdm(list((CACHE/'attempts').glob('*.json')), desc='Save request provenance', unit='attempt'):
            atomic_json(BUNDLE/'provenance/attempts'/attempt.name, read_json(attempt))
    print('Extraction checkpoint:', len(read_json(BUNDLE/'records.json')), 'validated frame records')
    """),
        md("""
    ## Caption embeddings — BGE-small, local inference
    MIT-licensed `BAAI/bge-small-en-v1.5`, pinned revision, 384 dimensions, CLS pooling,
    normalized float32, 512-token limit including special tokens. The documented retrieval prefix
    is applied to queries only. Captions stay natural prose and never include reference answers.
    Overlong captions raise an error with row IDs/token counts rather than silently truncate.
    Optional model export contains only a small SentenceTransformer snapshot for offline local queries.
    """),
        code("""
    from rvm.embeddings import embed_bundle
    if not SMOKE:
        vectors = embed_bundle(BUNDLE, config['embedding'], cache_folder=WORK/'hf-cache', show_progress=True)
        print('Embedding checkpoint:', vectors.shape, vectors.dtype)
    else:
        print('Synthetic fixture vectors retained; no model inference')
    """),
        md("""
    ## Final judgments and portable export
    Use the widget above or edit qrels.csv with `relevance=0/1,source=human,status=confirmed` for
    every inspected pair. Leave blanks unjudged. You may export with annotation pending; the local
    notebook will display retrievals/latency but suppress ground-truth metrics for incomplete queries.
    Run this final cell again after saving annotations to seal fresh checksums. The bundle contains
    only selected images and portable paths, plus separate annotation/reference metadata and provenance.
    """),
        code("""
    from rvm.bundle import finalize_bundle
    validated = finalize_bundle(BUNDLE, synthetic=SMOKE, show_progress=True)
    display(pd.DataFrame(read_json(BUNDLE/'annotation/status.json')))
    ZIP = export_zip(BUNDLE, WORK/'robot-visual-memory-compact.zip', show_progress=True)
    print('Export checkpoint:', ZIP, round(ZIP.stat().st_size/1024**2, 2), 'MiB')
    print('Status:', validated['manifest']['benchmark_status'])
    print('ZIP was extracted and revalidated in a fresh temporary directory. Download only this compact ZIP.')
    """),
    ],
)

save(
    "02_evaluate_retrieval_local.ipynb",
    [
        md("""
    # Robot Visual Memory — 2. Graph vs caption embedding retrieval

    Exactly two pipelines over the same compact gallery. No original dataset download, image
    extraction, answer generation or MCP. Start Compose and import the Kaggle export first.
    Graph/caption outputs differ, so this compares retrieval pipelines, not a controlled proof that
    graph representation alone is superior. The one-episode configuration is exploratory; there is
    no held-out environment. Extraction/retrieval settings remain frozen before running the queries.
    """),
        code("""
    # Configuration — use a real exported directory/ZIP after database import.
    import os, sys, time
    from pathlib import Path
    SMOKE = os.getenv('RVM_NOTEBOOK_SMOKE') == '1'
    ROOT = next((p for p in [Path.cwd(), Path.cwd().parent] if (p/'pyproject.toml').exists()), None)
    if ROOT is None:
        raise RuntimeError('Launch Jupyter from the project root or notebooks directory')
    ROOT = ROOT.resolve()
    sys.path.insert(0, str(ROOT/'src'))
    BUNDLE_PATH = Path(os.getenv('RVM_BUNDLE', str(ROOT/'data/robot-visual-memory-compact')))
    OUTPUT = Path(os.getenv('RVM_RESULTS', str(ROOT/'results/retrieval')))
    LOCAL_SNAPSHOT = None  # optional Path to bundle/model_snapshot, otherwise pinned small HF model download
    QRELS_OVERRIDE = None  # optional completed human CSV outside immutable exported bundle
    FREEZE_FILE = None  # defaults to BUNDLE/development-freeze.json
    CONFIG_OVERRIDE = None  # optional YAML matching the frozen settings from the episode pilot
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print('Mode:', 'SYNTHETIC SMOKE — NOT BENCHMARK' if SMOKE else 'real retrieval evaluation')
    """),
        md("""
    ## Validate the compact gallery and annotation status
    Bundle checksums, images, dimensions, graph edges and vector-row alignment are checked before
    retrieval. Answers stay unopened. Qrels are loaded only by evaluation, never by either retriever.
    A separate human CSV can complete an annotation-pending export without modifying its checksums.
    Synthetic fixture mode below is only for offline notebook CI and cannot produce real benchmark claims.
    """),
        code("""
    import pandas as pd
    from IPython.display import display
    from rvm.bundle import open_bundle, validate_bundle
    from rvm.io import load_config, read_json, file_hash, digest
    from rvm.annotation import load_qrels, annotation_status, validate_qrels
    if SMOKE:
        from rvm.fixture import create_single_episode_fixture
        BUNDLE_PATH = create_single_episode_fixture(OUTPUT/'synthetic-fixture', config_path=ROOT/'config.example.yaml')
    BUNDLE = open_bundle(BUNDLE_PATH, show_progress=True)
    bundle = validate_bundle(BUNDLE, show_progress=True)
    if bundle['manifest']['synthetic'] and not SMOKE:
        raise ValueError('Synthetic fixture is forbidden in real evaluation mode')
    frames = bundle['frames']
    config = load_config(CONFIG_OVERRIDE) if CONFIG_OVERRIDE else read_json(BUNDLE/'selection.json').get('config', load_config(ROOT/'config.example.yaml'))
    selected_qids = read_json(BUNDLE/'annotation/query_ids.json')
    questions = [q for q in bundle['questions'] if q.query_id in selected_qids]
    judgments = load_qrels(QRELS_OVERRIDE) if QRELS_OVERRIDE else bundle['judgments']
    validate_qrels(judgments, frames, bundle['questions'])
    display(pd.DataFrame(annotation_status(judgments, frames, questions, synthetic=SMOKE)))
    evaluation_scope = bundle['manifest'].get('evaluation_scope', 'held_out_environments')
    print('Candidate images:', len(frames), 'Selected evaluation questions:', len(questions))
    print('Evaluation scope:', evaluation_scope)
    if evaluation_scope == 'single_episode_exploratory':
        print('EXPLORATORY: pilot review and retrieval use the same episode; no held-out environment claims')
    print('No reference-answer file was loaded')
    """),
        md("""
    ## Connect the two Python-client retrievers
    The graph parser is configurable (`rules` by default, or cached bounded GPT plans). Plans use
    the question alone; GPT planning usage/latency is logged separately. The rules vocabulary,
    synonyms, partial-match policy and limitations are in README. Graph retrieval never falls back
    to vectors. Caption queries use the exact exported BGE contract and Qdrant environment filter.
    Model initialization time is measured separately from online query time.
    """),
        code("""
    from rvm.evaluation import verify_frozen
    from rvm.normalization import rules_hash
    from rvm.planning import RulesPlanner, GPTPlanner
    from rvm.retrieval import GraphRetriever, CaptionRetriever
    from rvm.embeddings import CaptionEmbedder
    model_load_seconds = 0.0
    if not SMOKE:
        freeze_hash = verify_frozen(config, FREEZE_FILE or BUNDLE/'development-freeze.json')
        if bundle['manifest']['rules_hash'] != rules_hash():
            raise ValueError('Normalization differs from frozen export')
        from rvm.ingestion import clients, verify_import
        driver, qdrant = clients()
        display(verify_import(BUNDLE, driver, qdrant, config['retrieval']['collection']))
        start = time.perf_counter()
        print('Loading BGE model; Hugging Face displays download progress for uncached files...')
        embedder = CaptionEmbedder(bundle['embedding'], snapshot=LOCAL_SNAPSHOT,
                                   cache_folder=ROOT/'artifacts/hf-cache')
        model_load_seconds = time.perf_counter()-start
        parser_name = config['retrieval']['parser']
        if parser_name not in {'rules','gpt'}:
            raise ValueError('Unknown graph parser')
        planner = RulesPlanner() if parser_name == 'rules' else GPTPlanner(OUTPUT/'plan-cache', config['extraction'])
        systems = {'graph': GraphRetriever(driver, planner, config['retrieval']['collection'], config['retrieval']['min_graph_score']),
                   'caption_embedding': CaptionRetriever(qdrant, embedder, config['retrieval']['collection'])}
    else:
        # Canned results test notebook/report plumbing, not database retrieval or model quality.
        from rvm.schemas import RetrievalResult, Hit
        class SyntheticReplay:
            def __init__(self, reverse=False): self.reverse = reverse
            def retrieve(self, question, environment_id, k):
                ids = [f.frame_id for f in frames if f.environment_id == environment_id]
                if self.reverse: ids.reverse()
                return RetrievalResult(hits=[Hit(frame_id=fid,score=float(len(ids)-i)) for i,fid in enumerate(ids[:k])],
                                       diagnostics={'total_seconds':0.0,'database_seconds':0.0,'error':None,
                                                    'synthetic_replay':True})
        systems = {'graph': SyntheticReplay(), 'caption_embedding': SyntheticReplay(True)}
        freeze_hash = 'SYNTHETIC-REPLAY-NOT-BENCHMARK'
    print('Retriever checkpoint: exactly', list(systems), 'model load seconds:', model_load_seconds)
    """),
        md("""
    ## Rankings, metrics and latency
    Retrieve up to the environment's full selected gallery size for untruncated MRR; compute
    Precision/Recall/nDCG at k=1,3,5. P@k always divides by k, penalizing missing positions even
    when the gallery is smaller than k. Recall divides by the total number of relevant gallery
    images. Binary nDCG uses log2(rank+1) discounts. Duplicate/wrong-environment results fail validation.

    Primary means use the **same fully human-judged, evidence-bearing queries** for both methods.
    No-relevant queries have P=0 and undefined Recall/MRR/nDCG and are reported as coverage cases.
    Partial/pooled/silver-only galleries have null metrics: unjudged is never treated as 0.
    Retrieval errors yield empty results and are penalized in eligible primary metrics, with errors reported.

    First pass and warm passes include query embedding or planning; DB-only times are separate.
    First pass is a practical query/cache proxy: Neo4j/Qdrant/OS caches are not flushed, and BGE weights
    have already loaded. For actual cold-start experiments restart services and kernel, record startup,
    and run once. Do not claim these proxy times are fully cold database latencies.
    """),
        code("""
    from rvm.evaluation import evaluate, report_results
    from rvm.extraction import usage_summary, usage_delta
    planning_usage_before = usage_summary(OUTPUT/'plan-cache')
    table, runs, status = evaluate(systems, questions, frames, judgments,
                                   k_values=config['retrieval']['k_values'],
                                   warm_repeats=config['retrieval']['warm_repeats'], synthetic=SMOKE,
                                   show_progress=True)
    aggregate, by_category, diagnostics, paired = report_results(table, runs, status, OUTPUT, provenance={
        'synthetic': SMOKE, 'evaluation_scope': evaluation_scope, 'freeze_hash': freeze_hash, 'config_hash': digest(config),
        'bundle_manifest_hash': file_hash(BUNDLE/'bundle.json'), 'bundle_checksums_hash': file_hash(BUNDLE/'checksums.json'),
        'qrels_hash': file_hash(QRELS_OVERRIDE or BUNDLE/'annotation/qrels.csv'),
        'versions': bundle['manifest']['versions'], 'model_load_seconds': model_load_seconds,
        'online_query_usage': 'See runs.json and plan-cache attempts; rules/BGE have no hosted embedding cost',
        'online_planning_usage': usage_delta(planning_usage_before, usage_summary(OUTPUT/'plan-cache')),
        'offline_extraction_usage': read_json(BUNDLE/'provenance/extraction-usage.json')
            if (BUNDLE/'provenance/extraction-usage.json').exists() else None,
    })
    if any(not s['complete'] for s in status):
        print('INCOMPLETE EVALUATION: complete human judgments before full-gallery ground-truth claims')
    if SMOKE: print('SYNTHETIC REPLAY — all displayed numbers are test-only')
    display(aggregate, by_category, diagnostics, paired, table)
    latency = pd.read_csv(OUTPUT/'latency.csv')
    display(latency.groupby(['method','phase']).total_seconds.agg(['count','mean','median','max']))
    print('Results exported:', OUTPUT)
    """),
        md("""
    ## Side-by-side evidence and failure review
    Scores are pipeline-specific; compare ranked evidence and metrics, never raw score magnitudes.
    Graph explanations report matched plan constraints, not truth labels. Record failure causes such
    as sampled-gallery coverage, caption omission, object/state extraction error, relation direction,
    synonym/parser gaps, or distractors in a separate review file; do not relabel from model outputs.
    """),
        code("""
    from rvm.evaluation import compare_display
    for query in questions[:3]:
        compare_display(query, runs, frames, BUNDLE, k=max(config['retrieval']['k_values']))
    display(table.groupby(['method','failure_category']).size().to_frame('query-k cases'))
    if not SMOKE:
        driver.close()
        qdrant.close()
    print('Evaluation finished; consult report.json for annotation completeness and provenance')
    """),
    ],
)
