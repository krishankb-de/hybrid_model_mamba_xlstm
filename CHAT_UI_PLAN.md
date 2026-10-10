# CHAT_UI_PLAN — CXR report-generation chat UI, with image retrieval

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task by task. The tracked checkboxes are the `- [ ] **Pn-X**` lines, which `scripts/mamba3_state.py` parses; the numbered steps under each box are that task's TDD cycle.

**Branch:** `chat_ui` (cut from `h100_efficiency` @ `5c45527` on 2026-10-01) · **State:** `chat_ui_state.json` · **Helper:** `venv/bin/python scripts/mamba3_state.py --plan chat_ui show|tick|note|phase|sync` · **Spec:** `docs/chat_ui/CHAT_UI_SPEC.md` (the 2026-09-30 design doc and the 2026-10-01 image additions, verbatim)

**Goal:** A Claude-style chat page. The user gives a chest X-ray, watches `preprocess › encode › retrieve › generate › label › score` run, sees similar training X-rays and matching reports, and gets a stored, replayable, exportable session. It is served from the cluster at zero cost, and no MIMIC-derived byte leaves the cluster.

**Architecture:** One CPU `sbatch` job runs FastAPI (engine, gallery, SQLite store) and, as a second process in `.venv_chexbert`, the CheXbert labeller. The laptop only forwards a port over SSH. The page is vanilla ES modules served by FastAPI. The engine is a thin layer over code the repo already trusts (`load_report_generation_module`, `beam_search_cached`, the `evaluate_cxr_retrieval` loaders, `bootstrap_compare.chexbert_f1`, `repair_report`). The only edit to thesis code is an optional `on_step` callback, pinned by parity tests.

**Tech Stack:** Python 3.11 (cluster `.venv`) and 3.14 (laptop `venv/`) · FastAPI, uvicorn, python-multipart, stdlib `sqlite3` · PyTorch 2.11, open_clip BiomedCLIP ViT-B/16 · f1chexbert in `.venv_chexbert` · vanilla HTML/CSS/JS with no build step and no CDN · `node --test` (node 25 on the laptop) · SLURM partition `pot-hpi-aisc-batch`.

> **▶ Approved 2026-10-01** (checkbox **P0-F**). The user answered the open questions (§3a), authorised autonomous job submission, and chose subagent-driven execution with a review after every task. **No deletion and no major change to anything that already exists (cluster checkout, shared venvs, data, outputs) without the user's explicit go-ahead.**
>
> **⚠ No published number moves.** The app imports the thesis pipeline. Its one edit to it is an additive `on_step=None` callback with parity tests. Nothing is retrained and no metric in `analysis/` changes. **No merge into `h100_scaling` or `main` without an explicit instruction.**

---

## 0. Resume protocol (a new session does this first)

1. `git branch --show-current` must print `chat_ui`.
2. `venv/bin/python scripts/mamba3_state.py --plan chat_ui show` prints progress per phase and the **next unticked box**. `... show P5` lists one phase's boxes.
3. Read `chat_ui_state.json` keys `status`, `next_action`, `decisions`, `open_questions`, then the phase section below. Facts the plan relies on are under `verified_facts`, each with its file and line.
4. Work one task at a time with the task loop (§8). After every meaningful change: tick with evidence, update `next_action`, commit.
5. Cluster work goes through `scripts/chat_remote.sh` from the Mac (P0-G): `sync` (rsync, never `--delete`), `submit <wrapper> [VAR=value …]`, `state <jobid>`, `summary <log>` (fixed patterns only, R7). The login node runs nothing scripted. Record job ids as evidence.
6. If `chat_ui_state.json` is lost, `venv/bin/python scripts/mamba3_state.py --plan chat_ui sync` rebuilds `phases` from this file's checkboxes. Evidence is lost then; recover it from `git log`.

## 1. What the user gets

| User-visible feature | Where |
|---|---|
| Attach an X-ray by click, drag-drop or paste, with a preview before sending | P6-A |
| Streamed stage timeline and report (Findings / Impression), copy, raw and repair toggles | P3, P4 |
| The X-ray stays in the chat after reload or reopen; the 224×224 model input shown beside it | P6-B |
| Similar training X-rays with similarity and "n/14 labels agree" | P5, P6-C |
| Matching reports from the 13D image and text encoders; own-report rank for test studies | P5, P6-D |
| Full-screen viewer, side by side, zoom, pan, brightness, contrast | P6-E |
| CheXbert-14 label chips; ROUGE-L / BLEU / CheXbert against a reference (private mode) | P5 |
| Sessions: list, replay, export (JSON, Markdown), delete; settings drawer | P3, P4 |
| Served from the cluster, reached from the laptop, survives preemption | P1, P7 |
| Laptop dev loop with no weights and no data (tiny engine, tiny gallery) | P2, P5 |

## 2. Global Constraints

- **DUA (Class R, `analysis/ARCHIVE_MANIFEST.md`).** Checkpoints, gallery files, MIMIC images, report text, study and subject ids, CheXbert labels of MIMIC reports, test-split images, and the session database stay on the cluster. The browser receives generated text and derived numbers. MIMIC text, ids and images reach the browser only in **private** mode. Nothing MIMIC-derived is committed, copied to the laptop, or put on a free cloud tier.
- **Login node `lx01` runs nothing scripted.** Every cluster step is one `sbatch …` or one `ssh lx01 <single command>`.
- **SLURM invariants.** `#SBATCH --partition=pot-hpi-aisc-batch` (renamed 2026-09-30, enforced by `test_no_slurm_script_uses_the_retired_aisc_batch_partition`), `#SBATCH --account=aisc`, `#SBATCH --qos=aisc` on CPU-only jobs (the proven CPU combination, `tests/test_willi_parity.py:3145-3159`), `#SBATCH --exclude=ga03,gx17v1,gx13v1`, GPUs as `--gpus=N` and never `--gres`, `--requeue` plus `--open-mode=append` on long preemptible jobs, logs at `logs/%x_%j.log`, and `cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"`.
- **Compute nodes are offline.** `SCRATCH_ROOT=/sc/scratch/$USER/hybrid_xmamba_h100`, `HF_HOME="${SCRATCH_ROOT}/.hf"`, `HF_HUB_OFFLINE=1`.
- **Zero cost.** No paid service, no cloud GPU.
- **Published protocol is the default.** `decode=beam`, `beam_size=3`, `max_new_tokens=100`, empty `input_ids` (no BOS), fp32, no autocast, `prefix_k` from `run_metadata.json` through `load_report_generation_module`. Transform: `Resize((224,224)) → Grayscale(3) → ToTensor → Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])`. Report text: `tokenizer.decode(ids, skip_special_tokens=True)`, whitespace-collapsed with `" ".join(s.split())` exactly as `write_hyps_refs` does.
- **Models.** Default: `hybrid_150m_m3_rrg` + `outputs/h100_report_gen_m3_tower13d_s42/checkpoints/last.ckpt` (cached decode available). Selectable: `hybrid_150m_v2_rrg` + `outputs/h100_report_gen_full_ext_4gpu_tower13d/checkpoints/last.ckpt` (13D, uncached only). Retrieval: `outputs/h100_kd_150m_v2_full_data_lr3e6/checkpoints/last.ckpt`, the 13D image and text encoders (user decision, 2026-10-01).
- **Paths.** `CHAT_HOME=/sc/home/$USER/chat_sessions/` holds the DB, uploads, endpoint file, token file and `gallery/<build_id>/`. It is never inside the repo tree. Dataset: `/sc/home/$USER/dataset/mimic_full/{train,validate,test}.parquet`.
- **Option bounds.** `beam_size` 1–8 · `max_new_tokens` 16–200 · `k_images` 0–12 (default 4) · `k_reports` 0–10 (default 3) · at most 4 accepted turns at once · upload ≤ 20 MB, ≥ 64×64 px, ≤ 50 MP, PNG/JPEG/WEBP.
- **Copy.** Banner: "Research prototype — not for clinical use." `message_stop.disclaimer`: "Research prototype; not for clinical use."
- **Retention.** 45 days in private mode (user, 2026-10-01), 7 days in public mode; `RETENTION_DAYS` overrides.
- **Cluster workflow (P0-G; modelled on `~/Desktop/Projects/leg_x_hybrid/legal_ai_hybrid_mamba_xlstm_model`).** Host alias `hpi-hpc` (login node lx01, user `krishankumar.bhushan`). Chat-UI code lives in its own cluster directory `CLUSTER_REPO=/sc/home/krishankumar.bhushan/hybrid_chat_ui`, filled by `rsync` from the Mac **without `--delete`**. The thesis checkout `MAIN_REPO=/sc/home/krishankumar.bhushan/hybrid_mamba_xlstm` (git, branch `h100_efficiency`, updated by the user's `git pull`) is never written to, except new `results/chat_*` subdirectories through the `results` symlink. `CLUSTER_REPO` reaches `outputs/`, `results/`, `.venv`, `.venv_chexbert` through symlinks into `MAIN_REPO`. Commands sent to lx01 are single commands from this list: `sbatch`, `squeue --me`, `sacct`, `scancel <own chat job>`, `cat`/`grep`/`tail` on summary lines only (R7), `ls`, `du`, `df`, `mkdir`. Never `python`, `bash x.sh`, `source`, heredocs, `rm`, or `git` on the cluster.
- **Shared environments are not modified.** The app's web deps go into overlay directories (`pip install --target`) inside `CLUSTER_REPO` and are put on `PYTHONPATH` by the chat wrappers only; `MAIN_REPO/.venv` and `.venv_chexbert` stay byte-identical.
- **R7 applies to every agent.** No MIMIC-derived content enters an agent's context: never `cat`/`tail`/pull a log, dump or file that can contain report text, study ids or images; read only lines matching the fixed summary patterns (`scripts/chat_remote.sh summary`). Wrappers write raw script output to files under `results/` and print only summary lines to the job log.
- **No deletion, no major change** to existing files, data, checkpoints, venvs or the cluster checkout without the user's go-ahead. New files and new directories are fine. `scancel` only on this plan's own jobs.
- **Code hygiene.** No PEP 604 (`X | Y`) or PEP 585 (`dict[...]`) syntax in runtime code. `app/` joins `SCAN_ROOTS` in `tests/test_willi_parity.py` (P2-A). Comment density and naming follow the surrounding repo code.
- **Gate before any tick or commit.** `bash scripts/validate.sh` exits 0. From P4-B it also runs `node --test`.
- **Commits.** One per task, message `"<ID>: <what>"`, kept short, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never stage `Colab_Distillation_E2E_Test.ipynb` or `ISBI Paper /` (pre-existing, unrelated work).

## 3. Spec deltas (verified 2026-10-01; the plan follows these, not the spec)

| # | Spec says | Code says (evidence) | Plan does |
|---|---|---|---|
| D1 | `--partition=aisc-batch` | Renamed to `pot-hpi-aisc-batch`; sbatch rejects the old name (commit `5c45527`; `test_no_slurm_script_uses_the_retired_aisc_batch_partition`) | New name everywhere |
| D2 | Tiny engine is the `--smoke-test` model with `["mamba3","mlstm"]` | The smoke test builds `["mamba","mlstm"]`, `vocab_size=100`, `use_tfla=False` (`scripts/evaluate_report_generation.py:819-834`). Mamba-1 has no `step()`, so it cannot run the cached path | Tiny decoder = the config M6-D proves token-identical between cached and uncached beam search (`tests/test_mamba3_numerics.py:1925-1934`) |
| D3 | `thumbs/` holds 191,462 pre-made 256 px JPEGs | The packed corpus already stores one 320×320 JPEG per study (`scripts/build_mimic_cxr_local.py --size 320`; manifest: "the packed 320px corpus") | Serve the stored JPEG. No thumbnail build, and no 191k small files on NFS |
| D4 | Two towers or two passes | Report-gen trains with `vit_unfreeze_blocks: 0` (frozen) from `IMAGE_ENCODER_CKPT=…full_data_lr3e6`, and Phase 8+ contrastive checkpoints have no `img_proj` (`scripts/evaluate_cxr_retrieval.py:220-229`). So the decoder's tower **should equal** the 13D tower | Measure it: P5-B hashes both towers. If equal, one `forward_features` pass gives the patch grid, the image→image vector and the image→report query |
| D5 | Own report "ranks 3 of 194,125", first-of-group | The chapter's authoritative protocol is strict paired index inside the test split, `compute_retrieval_metrics(groups=None)` (`scripts/evaluate_cxr_retrieval.py:460-510`), N = 2,663 | Badge = rank among the 2,663 test reports with strict pairing; the dedup-aware rank goes in the tooltip. The list itself ranks train+test report groups |
| D6 | Engine output "byte-identical to the thesis dumps" when served on CPU | The dumps were decoded on an H100 with the uncached path (`results/report_gen_m3_test_split_s42/`). CPU fp32 rounds differently, and beam search flips on near-ties (V5-A changed 227 of 400 reports with no metric move) | Byte identity is a hard gate **on the same device**. CPU-vs-GPU drift is measured in P1-C and shown on every card, never claimed to be zero |
| D7 | Stop aborts the fetch; the server stores `aborted` | A dropped connection (tunnel hiccup) would then also abort. The polling fallback exists for exactly that case | Explicit `POST /v1/messages/{id}/cancel`. A disconnect never cancels |
| D8 | Public mode uses one shared bearer token | Every visitor would see every other visitor's sessions and uploads | Public sessions are scoped by a per-browser `X-Client-Id` |
| D9 | Private mode on localhost may run without a token; the API binds `0.0.0.0` | A compute node's `0.0.0.0` is reachable by every cluster user, credentialed or not | The server refuses a non-loopback bind without a token (R6). P1-B tests a loopback bind with `ssh -J` first |
| D10 | §4 option `retrieval_k` | §6a replaces it with `k_images` and `k_reports` | `retrieval_k` is rejected (`extra="forbid"`) |
| D11 | Greedy via `generate(top_k=1)` | That needs a second streaming hook | Greedy = `beam_size=1` through the same hooked beam search; a test proves token equality with `greedy_decode` |
| D12 | `hashlib.sha256(Path(ckpt).read_bytes())` | Reads 2.4 GB into RAM on every start | Streamed hash, cached by (path, size, mtime) in `CHAT_HOME/cache/sha256.json` |
| D13 | The labeller exposes `get_label(text)` | Real and in use: `F1CheXbert().get_label(text)`, `.target_names`, `.target_names_5_index` (`scripts/score_chexbert_standalone.py:96-108`) | Used as is. P5-C and P5-F check it exactly against the published `chexbert_labels.json` |
| D14 | Per-neighbour `label_agreement: 0.86` | Most of the 14 labels are negative for every report, so 13/14 is common even for unrelated images | Show "n/14 labels agree" (the user's wording) **and** the positive labels that differ |
| D15 | DICOM "converted in the browser or rejected" | Browser-side DICOM needs a parser library; no CDN and no vendoring are planned | Rejected with "export as PNG or JPEG" (detected by `DICM` at byte 128) |
| D16 | Four ES modules | The image features need focused files | Add `composer.js` (image intake) and `viewer.js` (viewer) |
| D17 | `--time=2-00:00:00` | The longest wrapper in the repo uses 36 h; 2 days is untested on this partition | `24:00:00` plus `--requeue`; pass `sbatch --time=…` on the command line to try longer |
| D18 | "Retrieval floor" in the demo | The floor's per-study output already exists: `results/retrieval_floor_test_split/hyps.txt` (stock BiomedCLIP; `scripts/submit_v3_chain.sh:62`) | Show that line for test studies (private), exact by construction |
| D19 | A prototype `cxr_chat_app.zip` exists | Not found (searched the repo, `~/Desktop`, `~/Downloads` and Spotlight on 2026-10-01) | Build from scratch (user, U1) |
| D20 | Test-split picker, but no endpoint lists studies | The picker needs a list | `GET /v1/test-studies` (private only) |
| D21 | `--engine mock` replays a recorded log | The tiny engine already runs a turn in under a second | Dropped |
| D22 | SQLite "in WAL mode" under `/sc/home` | `/sc/home` is NFS (`uv venv --clear` was unreliable there, `H100_SCALING_PLAN.md` 11B). WAL's shared-memory index is unsafe on network filesystems | `PRAGMA locking_mode=EXCLUSIVE` before `journal_mode=WAL`: WAL without shared memory, one process owns the DB. Commit cost measured in P1-B |
| D23 | Only non-`GET` requests need the token | Transcripts and images are `GET`s, and an `<img src>` cannot send a bearer header | When a token is set, every `/v1/*` route needs it; the page fetches images with headers into object URLs |
| D24 | Labeller started inside the server job's environment on fixed port 8001 | CheXbert runs with the default HF cache and `HF_HUB_OFFLINE=0` (`scripts/score_chexbert_h100.sh:44-45`, no `HF_HOME`); the server job sets `HF_HOME=${SCRATCH_ROOT}/.hf` and offline mode; compute nodes are shared | Start it with `env -u HF_HOME HF_HUB_OFFLINE=0` on a free loopback port. CheXbert on CPU is ~0.16 s per report (job 2525606), which sizes P5-C |

## 3a. User decisions (2026-10-01; these settle the open questions)

| # | Question | Decision | Where it lands |
|---|---|---|---|
| U1 | Reuse `cxr_chat_app.zip`? | No. Build from scratch. | D19 |
| U2 | Public mode content | Similarity scores only: no neighbour or report labels, no agreement, no group sizes, no images, ids or text | P3-C `PUBLIC_DROP`, P6-C, P6-D |
| U3 | Public tunnel demo | Private demo now; public tunnel is a later stage, only on the user's word | P9-D stays deferred; public-mode code is still built and tested |
| U4 | Commands vs drawer | Use the recommended settings: the drawer plus the text commands, defaults = the published protocol | P3-D, P4-D |
| U5 | Default model | Mamba-3 s42 (`hybrid_150m_m3_rrg`), which scored better; the 13D incumbent stays selectable ("a mix between the two"); an optional side-by-side compare mode is P9-F | §2 Models, P9-F |
| U6 | Private retention | 45 days, then delete (public stays 7) | §2, P8-B |
| U7 | CPU too slow | Target 8 s per turn. If CPU cannot meet it, serve from a GPU job automatically, no further question | P1-D decision tree, P7-B GPU wrapper |
| U8 | Job submission | Agents submit cluster jobs themselves, following the legal-AI project's ssh + rsync workflow; no deletion or major change without a go-ahead | §2 Cluster workflow, P0-G |
| U9 | Execution | Subagent-driven: a fresh implementer per task and a reviewer after each | §8 |

## 4. Pre-registered rules

- **R1 — DUA by construction.** One module, `app/redact.py`, shapes every event before it is stored and sent, and every export. Public payloads never contain `image_url`, `study_id`, `subject_id`, `gallery_row`, `txt_row`, `group`, `group_size`, MIMIC report text, per-neighbour or per-report labels, `neighbor_agreement`, `reference`, `test_row`, `identical_to` or `true_report_rank`: retrieval in public mode is rank and similarity only (U2). Field-by-field tests, plus a catch-all test that no private string value survives into a public payload.
- **R2 — Protocol fidelity on the same device.** Defaults reproduce the published protocol. Engine output must equal `evaluate_report_generation.py` byte for byte on the same node and device (hard). On GPU, the engine's uncached output must equal the published dump (hard). CPU-vs-GPU drift is measured and disclosed on every card, never claimed to be zero.
- **R3 — Thesis code untouched**, except `on_step=None` on `beam_search_cached` and `beam_search_decode`, each with parity tests showing identical tokens with and without the callback.
- **R4 — Predict first.** Each cluster job's prediction is written here and in `chat_ui_state.json["predictions"]` before submission. A wrong prediction is recorded beside the number.
- **R5 — Evidence or no tick.** A box is ticked only with evidence: a test name, a job id with its log path, or a metric.
- **R6 — No unauthenticated exposure.** The server refuses to start on a non-loopback address without a token. Public mode also needs `--mode public`, a token file and the rate limit. Tunnels are stopped when a demo ends.
- **R7 — No MIMIC text in an agent's context.** Anything sent to an agent leaves the cluster, which the DUA forbids for MIMIC data. Agents never read report text, reference text, study or subject ids, or images from MIMIC: no `cat`/`tail` of dumps or raw logs, no pulling `results/` or job logs to the Mac. Cluster scripts write raw output to files and print only summary lines (`RESULT {json}`, counts, timings, hashes, `ERROR …` without content); agents read logs only through `scripts/chat_remote.sh summary`, which greps a fixed pattern list. Tests use synthetic data only.
- **R8 — Additive only.** No deletion and no major change to anything that already exists — files, data, checkpoints, the cluster checkout, shared venvs — without the user's explicit go-ahead. New files, new directories, symlinks inside `CLUSTER_REPO`, and `scancel` of this plan's own jobs are allowed.

## 5. Architecture and file map

```
browser ──fetch POST (SSE body) · GET polling──▶ laptop :8000
   preferred:  ssh -N -J lx01 -L 8000:127.0.0.1:PORT NODE    (server bound to loopback)
   fallback:   ssh -N -L 8000:NODE:PORT lx01                  (server bound to the node; token required)
                                         ▼
 ┌──────────────── one CPU sbatch job on NODE (pot-hpi-aisc-batch) ────────────────┐
 │ python -m app.server (.venv)                                                     │
 │   ├─ Pipeline ── Engine(s): tower → prefix → decoder (beam search, on_step)      │
 │   │           ── Gallery: fp32 copies of the image/report embeddings + metadata  │
 │   │           ── LabelerClient ──HTTP──▶ 127.0.0.1:8001                          │
 │   ├─ Store: CHAT_HOME/chat.db (SQLite) + CHAT_HOME/uploads/<sid>/<sha256>/       │
 │   └─ /static: the single-page app                                                │
 │ uvicorn app.labeler:app (.venv_chexbert, 127.0.0.1:8001, one warm F1CheXbert)    │
 └──────────────────────────────────────────────────────────────────────────────────┘
```

| Path | Responsibility | Task |
|---|---|---|
| `app/__init__.py` | Package marker. Imports nothing, because `.venv_chexbert` imports `app.labeler` | P1-A |
| `app/requirements.txt` | fastapi, uvicorn, python-multipart, httpx | P2-A |
| `app/tunnel/probe_server.py`, `probe_client.py` | Stdlib streaming probe and its laptop client | P1-A |
| `app/tiny.py` | Tiny decoder config, `TinyTokenizer`, `TinyTower` | P2-B, P2-D |
| `app/imaging.py` | Upload sniffing and decoding (16-bit, EXIF, bounds), the published transform, the model-input image | P2-C |
| `app/engine.py` | `Engine`, `RealEngine`, `TinyEngine`; stages preprocess, encode, generate; provenance | P2-D |
| `app/schemas.py`, `app/ids.py` | `Options`, error envelope; sortable ids | P3-A |
| `app/store.py` | SQLite schema, events, uploads, export, retention sweep | P3-B |
| `app/redact.py` | Private/public field policy | P3-C |
| `app/commands.py` | Text-command parser | P3-D |
| `app/pipeline.py` | Stage orchestration, events, cancel | P3-D, P5-E |
| `app/server.py` | FastAPI factory, routes, auth, runner, SSE bridge, CLI | P3-D, P7-B |
| `app/labels.py` | `CHEXBERT_14`, `LabelerClient`, `RuleLabeler`, `label_agreement` | P5-A |
| `app/labeler.py` | CheXbert microservice (runs in `.venv_chexbert`) | P5-A |
| `app/gallery.py` | `Gallery`: load, image→image, image→report, own rank, duplicates, test studies | P5-D |
| `app/scoring.py` | `score_pair` over the thesis metric functions | P5-E |
| `app/static/index.html`, `styles.css` | Layout, banner, themes | P4-A |
| `app/static/api.js`, `state.js` | SSE parser, stream, poll, cancel, image loader; reducers and replay | P4-B |
| `app/static/render.js` | Card builders | P4-C, P6 |
| `app/static/app.js` | Wiring, sessions, drawer, routing | P4-D |
| `app/static/composer.js` | Click / drag / paste intake with preview | P6-A |
| `app/static/viewer.js` | Full-screen viewer | P6-E |
| `app/tunnel/tunnel.sh`, `public_demo.sh` | Laptop forward loop; cloudflared wrapper | P7-C |
| `app/README.md` | Runbook | P8-E |
| `scripts/chat_probe_h100.sh` | Reachability probe job | P1-A |
| `scripts/chat_cpu_decode_probe_h100.sh` | CPU decode speed and drift job | P1-C |
| `scripts/chat_engine_golden.py`, `_h100.sh`, `_gpu_h100.sh` | Engine vs script vs published dump | P2-E |
| `scripts/build_retrieval_gallery.py`, `_h100.sh` | Gallery build (and `--tiny`) | P5-B |
| `scripts/label_gallery_reports.py`, `_h100.sh`, `_cpu_h100.sh` | CheXbert labels for the gallery (GPU canary; CPU shard fallback) | P5-C |
| `scripts/chat_retrieval_gates.py`, `_h100.sh` | Live-path retrieval and labeller gates | P5-F |
| `scripts/chat_remote.sh`, `scripts/chat_cluster.env.example`, `.rsync-exclude-chat` | Mac-side sync (no `--delete`), submit, state, R7-safe log summary | P0-G |
| `scripts/chat_cluster_setup_h100.sh` | Cluster workspace: symlinks into the thesis checkout, web-dep overlays, input checks | P0-G |
| `scripts/chat_app_smoke_h100.sh` | Cluster import smoke for `app.server` / `app.labeler` (overlays) | P7-A |
| `scripts/serve_chat_h100.sh`, `scripts/serve_chat_gpu_h100.sh` | The server job (CPU, or GPU when P1-D says so) | P7-B |
| `scripts/check_no_restricted_files.sh`, `install_hooks.sh` | Pre-commit hygiene | P9-A |
| modify `hybrid_xmamba/models/hybrid_lm.py` | `on_step` on `beam_search_cached` | P2-B |
| modify `scripts/evaluate_report_generation.py` | `on_step` on `beam_search_decode` | P2-B |
| modify `scripts/validate.sh` | Add the `node --test` gate | P4-B |
| modify `tests/test_willi_parity.py` | `SCAN_ROOTS` += `app`; wrapper invariants | P1, P2-A, P5, P7 |
| tests | `tests/test_app_*.py`, `tests/app_helpers.py`, `tests/frontend/*.test.mjs` | each task |

## 6. Contracts (names and shapes every task uses)

### 6.1 Options (`app/schemas.py`)

```python
class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")          # D10: retrieval_k is rejected
    model: Optional[str] = None                          # None = the server's default engine
    decode: Literal["beam", "greedy"] = "beam"
    beam_size: Annotated[int, Field(ge=1, le=8)] = 3
    max_new_tokens: Annotated[int, Field(ge=16, le=200)] = 100
    cached_decode: bool = True
    compile: bool = False                                # refused unless the server allows it (P1-D)
    k_images: Annotated[int, Field(ge=0, le=12)] = 4
    k_reports: Annotated[int, Field(ge=0, le=10)] = 3
    label: bool = True
    reference: Optional[str] = Field(default=None, max_length=20000)   # private mode only
    display_repair: bool = False
    test_row: Optional[Annotated[int, Field(ge=0)]] = None             # private mode only (picker)
```

### 6.2 Events

A frame is `event: <name>\ndata: <one-line JSON>\n\n`. A keep-alive comment `: ping\n\n` goes out every 15 s and is not stored. Every stored and sent `data` carries `seq` (1, 2, … per message, no gaps).

Order of one turn:

```
message_start
stage_start/stage_end × preprocess, encode, retrieve
stage_start(generate) · content_block_start · content_block_delta × steps · content_block_stop · stage_end(generate)
stage_start/stage_end × label, score
message_stop
```

A skipped stage emits only `stage_end` with `{"stage": s, "skipped": "<reason>"}`. On error: `error`, then `message_stop` with `status:"error"`. On cancel: `message_stop` with `status:"aborted"`.

| Event | `data` besides `seq` |
|---|---|
| `message_start` | `message_id, user_message_id, session_id, mode, model{…card…}, options{…resolved…}, image{sha256, filename, source: "upload"\|"test_split"\|"previous", urls{original, thumb, model_input}}` |
| `stage_start` | `stage, index` |
| `stage_end` | `stage, ms, detail{…}`, or `stage, skipped` |
| `content_block_start` | `index: 0, content_block{type: "report", text: ""}` |
| `content_block_delta` | `index: 0, delta{type: "beam_snapshot", step, text}` |
| `content_block_stop` | `index: 0` |
| `warning` | `code, message` (for example `reference_ignored_public`, `not_a_command`) |
| `error` | `type: "error", error{type: validation_error\|model_error\|overloaded_error, message}` |
| `message_stop` | `message_id, status: done\|error\|aborted, total_ms, report, display_report, truncated_mid_sentence, disclaimer` |

### 6.3 Stage details

```
preprocess  {"format","mode","input_px":[w,h],"exif_transposed","resized_to":[224,224],"grayscale_to_3ch":true,
             "normalize":"biomedclip_clip_mean_std","image_sha256","source",
             "test_row"? (private), "identical_to"? {"split","row"} (private)}
encode      {"patch_grid":[197,768],"pooled_dim":512,"prefix_tokens":32,"device","one_pass":true}
retrieve    {"image_neighbors":[{"rank","similarity","gallery_row","image_url","study_id","labels":{name:0|1}}],
             "report_matches":[{"rank","similarity","group","group_size","report","labels"}],
             "true_report_rank"?:{"rank","of","rank_dedup","hit_at_10","protocol"},
             "gallery":{"build_id","images","report_rows","report_groups","towers_identical"}}
generate    {"decode","beam_size","tokens","stopped":"budget","cached_decode","compiled","prefill_ms",
             "per_token_ms","device","threads","drift_note"}
label       {"chexbert_14":{name:0|1},"positives":[…],
             "neighbor_agreement":[{"rank","agree","of":14,"both_positive","neighbor_only","generated_only"}]}
score       {"rouge_l","bleu_1","bleu_4","chexbert_14_micro_f1"?,"exact_match_14"?,"reference_source":"user"|"test_split",
             "reference_chexbert_14"?:{name:0|1},
             "published"?:{"model_report","floor_report","live_equals_published"}}   (published: test rows, private)
```

*Added at P4-C (2026-10-04):* `score.reference_chexbert_14` holds the reference's own 14 labels (P5-E already labels the reference for the CheXbert parts). The UI uses it to mark agree/disagree on each label chip; without it, the chips show no marks. It is private-mode only, because public mode drops the whole score event.

*Added at P4-F (2026-10-08):*
- **Display repair.** With `display_repair` on, each `content_block_delta.text` is the *stream view* of the current best beam: repeated sentences are dropped, and so is the start of a repeat. `display_report` is `repair_report(report, dedup="all", truncate=True)`. `message_stop.report` is always the raw protocol text.
- **Server features.** `GET /v1/models` also returns `features: {"retrieval": bool, "labels": bool}`. Each is true only when the pipeline actually runs that stage. P5-E must set them when it wires the gallery and the labeller.

### 6.4 Store (`CHAT_HOME/chat.db`)

```sql
PRAGMA locking_mode=EXCLUSIVE;   -- before WAL: no shared-memory index, so safe on NFS for one process (D22)
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  title TEXT NOT NULL DEFAULT '', mode TEXT NOT NULL CHECK (mode IN ('private','public')),
  client_id TEXT, deleted_at TEXT);
CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
  seq_in_session INTEGER NOT NULL, role TEXT NOT NULL CHECK (role IN ('user','assistant')),
  created_at TEXT NOT NULL, text TEXT NOT NULL DEFAULT '', mode TEXT NOT NULL,
  image_sha256 TEXT, image_filename TEXT, test_row INTEGER, options_json TEXT,
  status TEXT NOT NULL CHECK (status IN ('running','done','error','aborted')),
  report TEXT, display_report TEXT, provenance_json TEXT, total_ms REAL,
  UNIQUE (session_id, seq_in_session));
CREATE TABLE IF NOT EXISTS events (
  message_id TEXT NOT NULL REFERENCES messages(id), seq INTEGER NOT NULL, ts TEXT NOT NULL,
  event TEXT NOT NULL, data_json TEXT NOT NULL, PRIMARY KEY (message_id, seq));
CREATE TABLE IF NOT EXISTS artifacts (
  message_id TEXT NOT NULL REFERENCES messages(id), kind TEXT NOT NULL,
  path TEXT NOT NULL, restricted INTEGER NOT NULL DEFAULT 0);
```

Files: `CHAT_HOME/uploads/<session_id>/<sha256>/{original.<png|jpg|webp>, thumb.jpg, model_input.png}`. Test-split images are never copied; they are served from the dataset path, in private mode only.

### 6.5 Gallery files (`CHAT_HOME/gallery/<build_id>/`, Class R)

| File | Contents |
|---|---|
| `img_emb.npy` | float16 (191,462 × 512): train images, L2-normalised, 13D tower |
| `test_img_emb.npy` | float32 (2,663 × 512): test images (gate and offline checks) |
| `txt_emb.npy` | float16 (N_txt × 512): train rows then test rows, 13D text encoder |
| `txt_emb_test.npy` | float32 (2,663 × 512): test reports in `test.parquet` order (the own-rank gallery) |
| `txt_groups.npy`, `group_order.npy`, `group_starts.npy` | int64: duplicate-group ids (`group_ids_from_texts`), rows sorted by group, group boundaries |
| `txt_test_groups.npy` | int64: groups among test rows only (dedup-aware rank) |
| `txt_split.npy`, `txt_split_row.npy` | 0 = train, 1 = test; row index inside that split's parquet |
| `img_txt_row.npy` | int64: image row → its own report row |
| `img_meta.parquet` | row, study_id, subject_id, dicom_id, view, image (absolute path), file_sha256 |
| `test_meta.parquet` | test_row, study_id, subject_id, view, image, file_sha256 |
| `report_texts.txt` | One sanitised `Findings: … Impression: …` per report row |
| `labels.npy`, `label_names.json` | uint8 (N_txt × 14) CheXbert-14, written by P5-C |
| `manifest.json` | build_id, created, git_sha, job_id, checkpoint paths and file SHA-256, `tower_sha256` (13D), `decoder_tower_sha256`, `towers_identical`, `img_proj_present`, counts, transform, tokenizer settings, `labels_status`, `gate_rk` |

### 6.6 Python interfaces

```python
# app/engine.py
@dataclass
class StageResult:
    detail: Dict[str, Any]
    ms: float

@dataclass
class Prepared:
    image: "Image.Image"          # 8-bit RGB after load_upload (EXIF applied)
    model_input: "Image.Image"    # 224×224 RGB, exactly what the tower sees before ToTensor
    pixel_values: torch.Tensor    # (1, 3, 224, 224)
    sha256: str
    facts: Dict[str, Any]

@dataclass
class Encoded:
    patch_grid: torch.Tensor      # (1, 197, D_patch)
    pooled: torch.Tensor          # (D_joint,), L2-normalised: the image→image and image→report query
    prefix: torch.Tensor          # (1, k, D_model)

@dataclass
class Generated:
    token_ids: List[int]
    report: str
    display_report: str
    truncated_mid_sentence: bool

class Cancelled(Exception): ...

class Engine:
    name: str
    def preprocess(self, data: bytes) -> Tuple[StageResult, Prepared]: ...
    def encode(self, prepared: Prepared) -> Tuple[StageResult, Encoded]: ...
    def generate(self, enc: Encoded, opts: "Options", on_snapshot: Callable[[int, str], None],
                 cancel: threading.Event) -> Tuple[StageResult, Generated]: ...
    def card(self) -> Dict[str, Any]: ...
    def tower_sha256(self) -> str: ...

def build_engine(kind: str, **kw: Any) -> Engine: ...   # "tiny" | "real"

# app/labels.py
CHEXBERT_14: List[str]
class LabelerUnavailable(RuntimeError): ...
class LabelerClient:
    def __init__(self, url: str, timeout: float = 10.0): ...
    def label(self, texts: List[str]) -> List[List[int]]: ...
    def healthy(self) -> bool: ...
class RuleLabeler:
    def label(self, texts: List[str]) -> List[List[int]]: ...
    def healthy(self) -> bool: ...
def label_agreement(generated: Sequence[int], neighbor: Sequence[int]) -> Dict[str, Any]: ...

# app/gallery.py
class GalleryMismatch(RuntimeError): ...
class Gallery:
    @classmethod
    def open(cls, root: Path, expect_tower_sha256: Optional[str]) -> "Gallery": ...
    def image_neighbors(self, query: np.ndarray, k: int) -> List[Dict[str, Any]]: ...
    def report_matches(self, query: np.ndarray, k: int) -> List[Dict[str, Any]]: ...
    def own_report_rank(self, query: np.ndarray, test_row: int) -> Dict[str, Any]: ...
    def find_identical(self, file_sha256: str) -> Optional[Dict[str, Any]]: ...   # {"split", "row"}
    def image_path(self, gallery_row: int) -> Path: ...
    def test_study(self, test_row: int) -> Dict[str, Any]: ...                    # image, study_id, reference
    def list_test_studies(self, query: str = "", limit: int = 50) -> List[Dict[str, Any]]: ...

# app/pipeline.py
@dataclass
class TurnJob:
    session_id: str
    user_message_id: str
    message_id: str
    text: str
    upload: Optional[bytes]
    filename: Optional[str]
    options: "Options"

class Pipeline:
    def __init__(self, engines: Dict[str, Engine], default_model: str, store: "Store", mode: str,
                 gallery: Optional["Gallery"] = None, labeler: Optional[Any] = None,
                 published: Optional["PublishedDumps"] = None, drift_note: str = ""): ...
    def run(self, job: TurnJob, emit: Callable[[str, Dict[str, Any]], None],
            cancel: threading.Event) -> None: ...
```

## 7. Review Focus

The five failure modes the spec implies but never tests, most likely first. Each has a test in its owning task.

1. **16-bit grayscale PNG** (the usual DICOM export). `convert("RGB")` clips it to 255, so the model sees a white square while the browser preview looks normal. Expected: rescale to 8-bit first. → `tests/test_app_imaging.py::test_16bit_png_is_rescaled_not_saturated` (P2-C).
2. **A compute node's `0.0.0.0` is reachable by every cluster user.** A tokenless "private" server would show MIMIC text to uncredentialed colleagues. Expected: refuse to start. → `tests/test_app_api.py::test_refuses_non_loopback_bind_without_token` (P3-D) and the serve-wrapper parity test (P7-F).
3. **Shared public token.** Visitor B lists visitor A's sessions and uploads. Expected: scoping per browser. → `tests/test_app_store.py::test_public_sessions_are_scoped_to_their_client` (P3-B) and `tests/test_app_api.py::test_public_client_cannot_read_another_clients_session` (P3-D).
4. **A dropped stream (tunnel hiccup) treated as Stop.** The turn aborts. Expected: the turn finishes and polling recovers it. → `tests/test_app_api.py::test_dropped_stream_does_not_cancel_the_turn` (P3-D).
5. **EXIF-rotated phone photo.** The model sees a different orientation from the user. Expected: `exif_transpose` before the transform. → `tests/test_app_imaging.py::test_exif_rotation_is_applied_before_the_model_sees_it` (P2-C).

## 8. Task loop (every task)

1. Write the failing tests shown in the task.
2. Run them and confirm the failure is the expected one.
3. Implement the minimum shown.
4. Run the task's tests until green.
5. `bash scripts/validate.sh` (background it locally; 3–10 min). It must print `All gates passed.`
6. Tick with evidence: `venv/bin/python scripts/mamba3_state.py --plan chat_ui tick <ID> --evidence test=<file::name> --note "<date>: <ID> <one line>"`, then update `next_action` in `chat_ui_state.json`.
7. Commit only the task's files: `git add <files> CHAT_UI_PLAN.md chat_ui_state.json && git commit -m "<ID>: <what>"` (with the Co-Authored-By line).

Cluster tasks replace step 4 with "submit, then record the job id, log path and metric". Wherever a task says "on lx01: `sbatch X`" or "`sbatch X`", run `bash scripts/chat_remote.sh sync && bash scripts/chat_remote.sh submit X [VAR=value …]` from the Mac (P0-G), read results only with `state <id>` and `summary <log>` (R7), and never wait by polling in a loop: submit, record the job id in the state file (`note`), continue with laptop work, and check back later. At a phase boundary: `venv/bin/python scripts/mamba3_state.py --plan chat_ui phase P<n> --status "<one line>"`.

Execution (U9): a fresh implementer subagent per task and a reviewer subagent after each; implementers never spawn subagents. Every dispatch carries R7 and R8.

Shared test helpers live in `tests/app_helpers.py` (created in P2-C, extended later): `png_bytes(w, h, mode="L") -> bytes`, `jpeg_bytes(w, h) -> bytes`, `iter_sse(chunks: Iterable[str]) -> Iterator[Dict]` (the Python twin of `parseSSE`).

---

## 9. Phases

### P0 — Plan-of-record bootstrap

Gate: plan, spec and state file exist on `chat_ui`; the helper parses this plan; `validate.sh` is green; **the user approves**.

- [x] **P0-A** Spec saved verbatim as `docs/chat_ui/CHAT_UI_SPEC.md` (design doc 2026-09-30 + image additions 2026-10-01); allow-listed in `.gitignore`.
- [x] **P0-B** This plan: phases P0–P9, contracts (§6), spec deltas D1–D24 verified against the code, Review Focus, pre-registered rules.
- [x] **P0-C** `chat_ui_state.json` (resume protocol, paths, verified facts, predictions); `--plan chat_ui` registered in `scripts/mamba3_state.py`, whose `show` now prints the next box.
- [x] **P0-D** `CLAUDE.md` bootstrap paragraph for branch `chat_ui`.
- [x] **P0-E** Parity test `test_chat_ui_plan_set_is_registered_and_its_ids_parse` added; `bash scripts/validate.sh` exits 0.
- [x] **P0-F** User reviews this plan and says go. Then commit the P0 files on `chat_ui` and set `current_phase` to P1.

On approval:

```bash
git add CHAT_UI_PLAN.md chat_ui_state.json docs/chat_ui/CHAT_UI_SPEC.md scripts/mamba3_state.py \
        tests/test_willi_parity.py .gitignore CLAUDE.md
git commit -m "P0: chat UI plan-of-record, state file, helper registration"
venv/bin/python scripts/mamba3_state.py --plan chat_ui tick P0-F --note "<date>: user approved; P0 committed"
```

- [x] **P0-G** Cluster workspace: `scripts/chat_remote.sh` (sync without `--delete`, submit, state, summary), `.rsync-exclude-chat`, `scripts/chat_cluster.env.example`, setup job; first sync and setup job green.

**Files:** create `scripts/chat_remote.sh`, `scripts/chat_cluster.env.example`, `.rsync-exclude-chat`, `scripts/chat_cluster_setup_h100.sh`, `tests/test_chat_remote.py`; create the local, gitignored `scripts/chat_cluster.env`; modify `.gitignore` (`scripts/chat_cluster.env`, `.sync_stamp`), `tests/test_willi_parity.py`.
**Produces:** the four `chat_remote.sh` subcommands every later cluster step uses; `.sync_stamp` (`<UTC time> <git HEAD sha> <clean|dirty>`) in `CLUSTER_REPO`, which the engine's provenance reads where there is no `.git` (P2-D); on the cluster, `CLUSTER_REPO/{outputs,results,.venv,.venv_chexbert}` symlinks into `MAIN_REPO`, and the overlay directories `CLUSTER_REPO/.chat_deps` (for `.venv`) and `CLUSTER_REPO/.chat_deps_chexbert` (for `.venv_chexbert`) holding `fastapi>=0.115 uvicorn>=0.30 python-multipart>=0.0.9 httpx>=0.27` (the second without `python-multipart`/`httpx`).

`scripts/chat_cluster.env.example` (the real `scripts/chat_cluster.env` has the same content and is gitignored):

```bash
# CHAT_UI_PLAN.md P0-G. Copy to scripts/chat_cluster.env (gitignored).
CLUSTER_HOST=hpi-hpc
CLUSTER_REPO=/sc/home/krishankumar.bhushan/hybrid_chat_ui
MAIN_REPO=/sc/home/krishankumar.bhushan/hybrid_mamba_xlstm
SCRATCH_ROOT=/sc/scratch/krishankumar.bhushan/hybrid_xmamba_h100
```

`.rsync-exclude-chat` (root paths anchored with a leading `/`, as the legal-AI project learned in its job 2588703):

```
# rsync rules for scripts/chat_remote.sh sync. No --delete is ever used (R8).
/.git/
/venv/
/.venv/
/.venv_chexbert/
/.chat_deps/
/.chat_deps_chexbert/
/outputs/
/results/
/logs/
/data/
/cluster/
/output_willi_server/
/hpi_results_logs/
/ISBI Paper /
/.superpowers/
/scripts/chat_cluster.env
__pycache__/
.pytest_cache/
*.egg-info/
*.ckpt
*.pt
*.pth
*.parquet
*.npy
*.npz
*.db
.DS_Store
```

```bash
#!/usr/bin/env bash
# chat_remote.sh — the Mac-side door to the cluster for CHAT_UI_PLAN.md (P0-G).
#
#   bash scripts/chat_remote.sh sync                          # rsync this tree -> $CLUSTER_REPO (never --delete)
#   bash scripts/chat_remote.sh submit <wrapper> [VAR=value ...] [-- <sbatch args>]   # prints the job id
#   bash scripts/chat_remote.sh state <jobid>                 # sacct one-liner
#   bash scripts/chat_remote.sh summary <log path relative to $CLUSTER_REPO>         # R7-safe lines only
#   bash scripts/chat_remote.sh queue                         # squeue --me
#
# Modelled on the legal-AI project's sync_to_cluster.sh, minus --delete (R8: the cluster side is
# additive only) and minus pulling logs to the Mac (R7: logs can hold MIMIC text). Reads
# CLUSTER_HOST, CLUSTER_REPO, MAIN_REPO, SCRATCH_ROOT from scripts/chat_cluster.env.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${CHAT_CLUSTER_ENV:-$REPO_ROOT/scripts/chat_cluster.env}"
[[ -f "$ENV_FILE" ]] || { echo "missing $ENV_FILE: copy scripts/chat_cluster.env.example" >&2; exit 1; }
# shellcheck source=/dev/null
source "$ENV_FILE"
: "${CLUSTER_HOST:?}" "${CLUSTER_REPO:?}" "${MAIN_REPO:?}" "${SCRATCH_ROOT:?}"

# Lines a summary may show (R7). Everything else in a log stays on the cluster.
SUMMARY_PATTERN='^(RESULT |\[(probe|golden|gallery|labels|gates|server|setup|compile)\]|=== |ERROR|Traceback|[A-Za-z]*Error:|[[:space:]]*(Elapsed \(wall|Maximum resident)|  Missing keys|  prefix_k =)'

quote() {   # single-quote one argument for the remote bash; refuse embedded single quotes
  [[ "$1" != *"'"* ]] || { echo "argument contains a single quote: $1" >&2; exit 2; }
  printf "'%s'" "$1"
}

mask() {    # 8+ digit runs (MIMIC subject/study ids) and dicom-style ids never reach the terminal
  sed -E 's/[0-9]{8,}/<num>/g; s/[0-9a-f]{8}(-[0-9a-f]{8}){4}/<id>/g' | cut -c1-300
}

cmd="${1:-}"
shift || true
case "$cmd" in
  sync)
    stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ) $(git -C "$REPO_ROOT" rev-parse HEAD)"
    if [[ -n "$(git -C "$REPO_ROOT" status --porcelain -- app scripts hybrid_xmamba configs tests)" ]]; then
      stamp="$stamp dirty"
    else
      stamp="$stamp clean"
    fi
    printf '%s\n' "$stamp" > "$REPO_ROOT/.sync_stamp"
    ssh "$CLUSTER_HOST" "mkdir -p $(quote "$CLUSTER_REPO/logs")"
    rsync -az --exclude-from="$REPO_ROOT/.rsync-exclude-chat" "$REPO_ROOT/" "$CLUSTER_HOST:$CLUSTER_REPO/"
    echo "[sync] $REPO_ROOT -> $CLUSTER_HOST:$CLUSTER_REPO/ ($stamp)"
    ;;
  submit)
    wrapper="${1:?usage: submit <wrapper> [VAR=value ...] [-- <sbatch args>]}"
    shift
    [[ -f "$REPO_ROOT/$wrapper" ]] || { echo "no such wrapper: $wrapper" >&2; exit 2; }
    envs=() sb=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --) shift; sb=("$@"); break ;;
        *=*) envs+=("$(quote "$1")"); shift ;;
        *) echo "unexpected argument: $1" >&2; exit 2 ;;
      esac
    done
    sbq=()
    for a in "${sb[@]+"${sb[@]}"}"; do sbq+=("$(quote "$a")"); done
    ssh "$CLUSTER_HOST" "cd $(quote "$CLUSTER_REPO") && env ${envs[*]+"${envs[*]}"} sbatch --parsable ${sbq[*]+"${sbq[*]}"} $(quote "$wrapper")"
    ;;
  state)
    ssh "$CLUSTER_HOST" "sacct -j $(quote "${1:?jobid}") --format=JobID,JobName%28,State,Elapsed,MaxRSS,ExitCode,NodeList -P"
    ;;
  summary)
    ssh "$CLUSTER_HOST" "grep -aE $(quote "$SUMMARY_PATTERN") $(quote "$CLUSTER_REPO/${1:?log path}") | tail -n 200" | mask
    ;;
  queue)
    ssh "$CLUSTER_HOST" "squeue --me"
    ;;
  *)
    sed -n '2,10p' "$0"
    exit 2
    ;;
esac
```

`scripts/chat_cluster_setup_h100.sh` (CPU job; re-runnable; additive only):

```bash
#!/bin/bash
# ============================================================================
# CHAT_UI_PLAN.md P0-G — set up the chat UI's own cluster directory. Additive
# only (R8): creates symlinks that do not exist yet, installs the web deps into
# overlay directories (never into the shared venvs), checks the inputs exist.
#   bash scripts/chat_remote.sh submit scripts/chat_cluster_setup_h100.sh MAIN_REPO=<thesis checkout>
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node, x86 venv incompatible; gx13v1: faulty GPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=chat_setup
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"
MAIN_REPO="${MAIN_REPO:?set MAIN_REPO to the thesis checkout}"
DATA="${DATA:-/sc/home/$USER/dataset/mimic_full}"
echo "=== chat setup: node=$(hostname) job=${SLURM_JOB_ID:-?} ==="
for name in outputs results .venv .venv_chexbert; do
  if [ -e "${name}" ] || [ -L "${name}" ]; then
    echo "[setup] ${name}: present, left alone"
  elif [ -e "${MAIN_REPO}/${name}" ]; then
    ln -s "${MAIN_REPO}/${name}" "${name}"
    echo "[setup] ${name}: linked to the thesis checkout"
  else
    echo "[setup] ERROR ${name} not found in MAIN_REPO"
  fi
done
mkdir -p "${HOME}/chat_sessions" && chmod 700 "${HOME}/chat_sessions"
for f in outputs/h100_report_gen_m3_tower13d_s42/checkpoints/last.ckpt \
         outputs/h100_report_gen_full_ext_4gpu_tower13d/checkpoints/last.ckpt \
         outputs/h100_kd_150m_v2_full_data_lr3e6/checkpoints/last.ckpt \
         results/report_gen_m3_test_split_s42/hyps.txt results/report_gen_m3_test_split_s42/refs.txt \
         results/report_gen_m3_test_split_s42/chexbert_labels.json results/retrieval_floor_test_split/hyps.txt \
         "${DATA}/train.parquet" "${DATA}/test.parquet"; do
  if [ -f "${f}" ]; then echo "[setup] ok $(basename "$(dirname "${f}")")/$(basename "${f}")"
  else echo "[setup] ERROR missing $(basename "$(dirname "${f}")")/$(basename "${f}")"; fi
done
[ -d .chat_deps ] || .venv/bin/python -m pip install --quiet --target .chat_deps \
  "fastapi>=0.115" "uvicorn>=0.30" "python-multipart>=0.0.9" "httpx>=0.27"
[ -d .chat_deps_chexbert ] || .venv_chexbert/bin/python -m pip install --quiet --target .chat_deps_chexbert \
  "fastapi>=0.115" "uvicorn>=0.30"
PYTHONPATH=.chat_deps .venv/bin/python -c "import sys, torch, fastapi, uvicorn; print('[setup] main venv: python', sys.version.split()[0], 'torch', torch.__version__, 'fastapi', fastapi.__version__)"
PYTHONPATH=.chat_deps_chexbert .venv_chexbert/bin/python -c "import sys, fastapi, sklearn, transformers; assert int(transformers.__version__.split('.')[0]) < 5; assert tuple(int(x) for x in sklearn.__version__.split('.')[:2]) < (1, 8); print('[setup] chexbert venv: python', sys.version.split()[0], 'transformers', transformers.__version__, 'sklearn', sklearn.__version__, 'fastapi', fastapi.__version__)"
if command -v node >/dev/null 2>&1; then echo "[setup] node $(node --version)"; else echo "[setup] node absent"; fi
echo "=== END chat setup ==="
```

Tests (`tests/test_chat_remote.py`, laptop): `bash -n` passes on both scripts; `chat_remote.sh` contains no `--delete` and no `rm`; `sync` refuses without the env file (`CHAT_CLUSTER_ENV=/nonexistent` → exit 1); `quote` refuses a single quote (`submit x.sh "A=it's"` → exit 2, run with a fake env file and `PATH` where `ssh` is a stub script that records its argv into a temp file); a stubbed `submit scripts/chat_cluster_setup_h100.sh MAIN_REPO=/x -- --time=00:05:00` sends exactly `cd '<repo>' && env 'MAIN_REPO=/x' sbatch --parsable '--time=00:05:00' 'scripts/chat_cluster_setup_h100.sh'`; `mask` turns `s50414267` into `s<num>` and leaves a 7-digit job id alone; `.rsync-exclude-chat` lists `/outputs/`, `/results/`, `/.venv/`, `/venv/`, `/logs/`, `/data/`, `/.git/`, `/ISBI Paper /` and `*.parquet`. Parity test: the setup wrapper is CPU-only on `pot-hpi-aisc-batch` with `--qos=aisc`, excludes ga03, contains no `rm ` and no `ln -sf`, installs only with `--target`.

**As built (4a786cb, d77debd, 69944cb):** the committed `scripts/chat_remote.sh` and `scripts/chat_cluster_setup_h100.sh` are authoritative where they differ from the code above: `SUMMARY_PATTERN` also keeps dotted exception names, `mask` blanks every exception message to `<Name>: <msg>` and masks only digit runs not preceded by `.`; `env` arguments must be `NAME=value`; `CLUSTER_REPO` and `MAIN_REPO` may not overlap; `.sync_stamp` is sent last and alone; the setup job installs with `uv pip install --target` (the shared venvs have no pip), guards on `.setup_ok` sentinels and exits 1 on any `[setup] ERROR`.

Then: `cp scripts/chat_cluster.env.example scripts/chat_cluster.env`; `bash scripts/chat_remote.sh sync`; `bash scripts/chat_remote.sh submit scripts/chat_cluster_setup_h100.sh MAIN_REPO=/sc/home/krishankumar.bhushan/hybrid_mamba_xlstm`; when it ends, `state <id>` and `summary logs/chat_setup_<id>.log`. Gate: every `[setup]` line is `ok`/`linked`/`present`, no `ERROR`, both venv lines print. Tick with the job id and the versions. Commit `"P0-G: cluster workspace (rsync without delete, summary-only logs)"`.

### P1 — Feasibility probes (cluster; measure before building)

Gate: `chat_ui_state.json["decisions"]` records transport, bind policy, serving device, default decode path and whether `compile` is offered, each backed by a job id.

Why first: three assumptions carry the whole design and none is measured. (1) The laptop can reach a long-lived server inside a CPU job, with SSE frames unbuffered. (2) CPU beam-3 decoding of the 184M decoder is fast enough for a demo (spec target: 8 s per turn). (3) CPU decoding is deterministic, and its drift from the GPU dumps is small. P1 needs no app code beyond a stdlib probe, and it can run while P2 proceeds on the laptop.

Predictions (R4, written before submission):

- P1-B: loopback bind with `ssh -J lx01 NODE` works (60%); node bind with `ssh -L 8000:NODE:PORT lx01` works (70%). Frames arrive about 1.0 s apart on whichever works. SQLite on `/sc/home`: 1–20 ms per commit.
- P1-C: cached CPU beam-3 for 100 tokens takes 3–8 s per report at 8 threads; uncached takes 30–120 s. `cached_a` vs `cached_b`: 0/20 differ. Cached vs uncached on CPU: 0/5 differ. CPU vs published GPU: 0–6 of 20 differ. Peak RSS 3–5 GB.

- [x] **P1-A** Stdlib streaming probe `app/tunnel/probe_server.py` + `probe_client.py` and the CPU wrapper `scripts/chat_probe_h100.sh`; local tests green.

**Files:** create `app/__init__.py`, `app/tunnel/__init__.py`, `app/tunnel/probe_server.py`, `app/tunnel/probe_client.py`, `scripts/chat_probe_h100.sh`, `tests/test_app_probe.py`; modify `tests/test_willi_parity.py`.
**Produces:** `serve(bind: str, port: int, n_frames: int = 5, interval_s: float = 1.0) -> ThreadingHTTPServer`; `sqlite_commit_ms(path: str, n: int = 200) -> float`; `measure(url: str, timeout: float = 30.0) -> List[float]` (frame arrival times in seconds since the request).

1. Failing tests:

```python
# tests/test_app_probe.py
"""CHAT_UI_PLAN.md P1-A: the reachability probe streams SSE frames as written (stdlib only)."""
import json
import socket
import threading
import urllib.request

from app.tunnel.probe_client import measure
from app.tunnel.probe_server import serve, sqlite_commit_ms


def _start(**kw):
    server = serve("127.0.0.1", 0, **kw)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, "http://127.0.0.1:{}".format(server.server_address[1])


def test_probe_streams_frames_as_written_not_buffered():
    server, base = _start(n_frames=4, interval_s=0.2)
    try:
        arrivals = measure(base + "/stream")
    finally:
        server.shutdown()
    assert len(arrivals) == 4
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    assert min(gaps) > 0.1, gaps   # a buffering hop would deliver all four at once


def test_probe_healthz_names_the_host():
    server, base = _start()
    try:
        body = urllib.request.urlopen(base + "/healthz").read()
    finally:
        server.shutdown()
    assert json.loads(body) == {"status": "ok", "host": socket.gethostname()}


def test_sqlite_probe_reports_a_positive_commit_cost(tmp_path):
    assert sqlite_commit_ms(str(tmp_path / "probe.db"), n=20) > 0
```

```python
# tests/test_willi_parity.py (append)
def test_chat_probe_wrapper_is_cpu_only_on_the_renamed_partition():
    """CHAT_UI_PLAN.md P1-A. CPU-only: a GPU job would hold an accelerator idle for a port check."""
    src = (REPO_ROOT / "scripts" / "chat_probe_h100.sh").read_text()
    directives = [l for l in src.splitlines() if l.startswith("#SBATCH")]
    assert "#SBATCH --partition=pot-hpi-aisc-batch" in directives
    assert "#SBATCH --account=aisc" in directives and "#SBATCH --qos=aisc" in directives
    assert not [l for l in directives if "--gpus" in l or "--gres" in l]
    assert any(l.startswith("#SBATCH --exclude=ga03") for l in directives)
    assert "app/tunnel/probe_server.py" in src and "--sqlite-probe" in src
```

2. Run `venv/bin/python -m pytest tests/test_app_probe.py -v`. Expected: FAIL, `ModuleNotFoundError: No module named 'app'`.

3. Implement. `app/__init__.py` is a docstring only (`"""CXR report-generation chat app (CHAT_UI_PLAN.md). Imports nothing on purpose: .venv_chexbert imports app.labeler."""`); `app/tunnel/__init__.py` is empty.

```python
# app/tunnel/probe_server.py
"""CHAT_UI_PLAN.md P1-A: stdlib-only streaming probe for the cluster.

Answers one question before any app code exists: can a CPU job serve HTTP that reaches the laptop
through lx01 with SSE frames arriving as written rather than buffered? It also times SQLite commits
on the filesystem the store will use (D22). Standard library only, so it runs in any interpreter.

    python3 app/tunnel/probe_server.py --endpoint-file ~/chat_sessions/probe_endpoint \
        [--bind 127.0.0.1] [--sqlite-probe ~/chat_sessions/probe.db]
"""
import argparse
import json
import socket
import sqlite3
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def make_handler(n_frames: int, interval_s: float):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            print("[probe] " + fmt % args, flush=True)

        def do_GET(self):
            if self.path != "/healthz":
                self.send_error(404)
                return
            body = json.dumps({"status": "ok", "host": socket.gethostname()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/stream":
                self.send_error(404)
                return
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for i in range(n_frames):
                frame = "event: tick\ndata: {}\n\n".format(
                    json.dumps({"i": i, "server_ts": time.time()})).encode()
                self.wfile.write(b"%x\r\n" % len(frame) + frame + b"\r\n")
                self.wfile.flush()
                time.sleep(interval_s)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    return Handler


def serve(bind: str, port: int, n_frames: int = 5, interval_s: float = 1.0) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((bind, port), make_handler(n_frames, interval_s))


def sqlite_commit_ms(path: str, n: int = 200) -> float:
    """Mean ms per single-row commit, with the pragmas app/store.py will use (D22)."""
    con = sqlite3.connect(path)
    con.execute("PRAGMA locking_mode=EXCLUSIVE")
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("CREATE TABLE IF NOT EXISTS t (i INTEGER, s TEXT)")
    t0 = time.perf_counter()
    for i in range(n):
        con.execute("INSERT INTO t VALUES (?, ?)", (i, "x" * 300))
        con.commit()
    ms = (time.perf_counter() - t0) * 1000.0 / n
    con.close()
    return ms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=0, help="0 picks a free port")
    ap.add_argument("--endpoint-file", required=True)
    ap.add_argument("--sqlite-probe", default=None)
    args = ap.parse_args()
    if args.sqlite_probe:
        Path(args.sqlite_probe).expanduser().parent.mkdir(parents=True, exist_ok=True)
        print("[probe] sqlite_commit_ms={:.3f}".format(sqlite_commit_ms(str(Path(args.sqlite_probe).expanduser()))),
              flush=True)
    server = serve(args.bind, args.port)
    endpoint = Path(args.endpoint_file).expanduser()
    endpoint.parent.mkdir(parents=True, exist_ok=True)
    endpoint.write_text("{}:{}\n".format(socket.gethostname(), server.server_address[1]))
    print("[probe] serving on {}:{} (bind {})".format(socket.gethostname(), server.server_address[1], args.bind),
          flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
```

```python
# app/tunnel/probe_client.py
"""CHAT_UI_PLAN.md P1-B: time SSE frames arriving through the tunnel. Run on the laptop.

    venv/bin/python app/tunnel/probe_client.py http://127.0.0.1:8000/stream
Exit 0 iff at least 5 frames arrived spread out (min gap >= 0.5 s), i.e. nothing buffered them.
"""
import sys
import time
import urllib.request
from typing import List


def measure(url: str, timeout: float = 30.0) -> List[float]:
    req = urllib.request.Request(url, data=b"{}", method="POST",
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    arrivals, buf = [], b""
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        while True:
            chunk = resp.read1(65536)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                _, buf = buf.split(b"\n\n", 1)
                arrivals.append(time.monotonic() - t0)
    return arrivals


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/stream"
    arrivals = measure(url)
    for i, t in enumerate(arrivals):
        print("frame {} at {:.2f} s".format(i, t))
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    ok = len(arrivals) >= 5 and min(gaps) >= 0.5
    print("STREAMED" if ok else "BUFFERED or SHORT: {}".format(gaps))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

```bash
#!/bin/bash
# ============================================================================
# CHAT_UI_PLAN.md P1-A — cluster reachability probe for the chat server.
# CPU only. Times SQLite commits on /sc/home (D22), then serves the stdlib
# probe until --time runs out, writing <node>:<port> to ENDPOINT_FILE.
#
#   BIND=127.0.0.1 sbatch scripts/chat_probe_h100.sh   # path (b): ssh -J lx01 <node>
#   BIND=0.0.0.0   sbatch scripts/chat_probe_h100.sh   # path (a): forward via lx01
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node; gx13v1: faulty GPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=01:00:00
#SBATCH --job-name=chat_probe
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
BIND="${BIND:-127.0.0.1}"
ENDPOINT_FILE="${ENDPOINT_FILE:-$HOME/chat_sessions/probe_endpoint}"
echo "=== chat probe: node=$(hostname) bind=${BIND} job=${SLURM_JOB_ID:-?} ==="
python3 app/tunnel/probe_server.py --bind "${BIND}" --endpoint-file "${ENDPOINT_FILE}" \
  --sqlite-probe "$HOME/chat_sessions/probe.db"
```

4. Run `venv/bin/python -m pytest tests/test_app_probe.py tests/test_willi_parity.py -k "probe" -v`. Expected: PASS.
5–7. Task loop. Commit: `"P1-A: stdlib streaming probe + CPU wrapper"`.

- [x] **P1-B** Tunnel drill from the laptop: which path streams, round-trip time, SQLite ms per commit on `/sc/home`; recorded as decisions.

1. `bash scripts/chat_remote.sh sync`, then `bash scripts/chat_remote.sh submit scripts/chat_probe_h100.sh BIND=127.0.0.1`. Wait for RUNNING (`bash scripts/chat_remote.sh queue`).
2. Laptop: `EP=$(ssh hpi-hpc cat chat_sessions/probe_endpoint); NODE=${EP%%:*}; PORT=${EP##*:}`.
3. Path (b), loopback: `ssh -N -o ExitOnForwardFailure=yes -J hpi-hpc -L 8000:127.0.0.1:$PORT krishankumar.bhushan@$NODE &`, then `venv/bin/python app/tunnel/probe_client.py` → expect `STREAMED`; `curl -s localhost:8000/healthz` names the node. Stop the forward (kill the background `ssh`).
4. `bash scripts/chat_remote.sh` has no cancel; use `ssh hpi-hpc scancel <this probe's job id>` (own job, R8). Then `submit scripts/chat_probe_h100.sh BIND=0.0.0.0`. Path (a): `ssh -N -o ExitOnForwardFailure=yes -L 8000:$NODE:$PORT hpi-hpc &` → `probe_client.py`. `scancel` it afterwards.
5. `bash scripts/chat_remote.sh summary logs/chat_probe_<id>.log` → the `[probe] sqlite_commit_ms=` line.
6. Record: `tick P1-B --evidence jobs=<ids> path=<a|b|both> rtt_ms=<…> sqlite_commit_ms=<…>`, and write `decisions.transport` and `decisions.bind` in the state file.

Decision rule: prefer (b), because it never exposes a port on the cluster network. If only (a) works, R6 makes the token mandatory in every mode. If neither works, stop and ask the user: the transport design must change before P7.

- [x] **P1-C** CPU decode probe job `scripts/chat_cpu_decode_probe_h100.sh`: seconds per report (cached, uncached), determinism, drift from the published GPU dump, peak RSS.

**Files:** create `scripts/chat_cpu_decode_probe_h100.sh`; modify `tests/test_willi_parity.py`.

1. Failing parity test:

```python
def test_chat_cpu_decode_probe_is_cpu_only_and_decodes_the_published_protocol():
    """CHAT_UI_PLAN.md P1-C. Same protocol as the published dump, on CPU, compared line by line."""
    src = (REPO_ROOT / "scripts" / "chat_cpu_decode_probe_h100.sh").read_text()
    directives = [l for l in src.splitlines() if l.startswith("#SBATCH")]
    assert "#SBATCH --partition=pot-hpi-aisc-batch" in directives and "#SBATCH --qos=aisc" in directives
    assert not [l for l in directives if "--gpus" in l or "--gres" in l]
    assert any(l.startswith("#SBATCH --exclude=ga03") for l in directives)
    for needle in ("--model-config hybrid_150m_m3_rrg", "--decode beam", "--beam-size 3",
                   "--max-new-tokens 100", "--cached-decode", "report_gen_m3_test_split_s42/hyps.txt",
                   "OMP_NUM_THREADS", "HF_HUB_OFFLINE", "cached_vs_published_gpu_differ"):
        assert needle in src, needle
```

2. Run it: FAIL (file missing).
3. Implement:

```bash
#!/bin/bash
# ============================================================================
# CHAT_UI_PLAN.md P1-C — can the published decoder serve a chat turn on CPU?
# Decodes the first N test studies with the published protocol on CPU (cached
# twice for determinism, uncached on fewer), then compares with the published
# H100 dump line by line. Writes ${OUT}/summary.json.
#
#   sbatch scripts/chat_cpu_decode_probe_h100.sh
# Output (DUA-covered, never commit): results/chat_cpu_probe_<job>/
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node, x86 .venv incompatible; gx13v1: faulty GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=chat_cpu_decode_probe
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
DATA="${DATA:-/sc/home/$USER/dataset/mimic_full}"
CHECKPOINT="${CHECKPOINT:-./outputs/h100_report_gen_m3_tower13d_s42/checkpoints/last.ckpt}"
PUBLISHED="${PUBLISHED:-results/report_gen_m3_test_split_s42/hyps.txt}"
N="${N:-20}"
N_UNCACHED="${N_UNCACHED:-5}"
OUT="${OUT:-results/chat_cpu_probe_${SLURM_JOB_ID:-local}}"
export HF_HOME="${SCRATCH_ROOT}/.hf" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
source "${VENV_ACTIVATE}"
for f in "${CHECKPOINT}" "${PUBLISHED}" "${DATA}/test.parquet"; do
  [ -f "$f" ] || { echo "ERROR: not found: $f"; exit 1; }
done
mkdir -p "${OUT}"
# Node type, for the determinism record. R7: the job log carries only [probe]/RESULT/=== lines;
# the eval script's own output (which prints report text) stays in ${OUT}/<arm>.log on the cluster.
echo "[probe] cpu $(lscpu | grep -m1 '^Model name' | cut -d: -f2 | xargs) ncpu=${SLURM_CPUS_PER_TASK:-?}"
TIME_V=(); [ -x /usr/bin/time ] && TIME_V=(/usr/bin/time -v)

run() {  # $1 = arm name; the rest are extra flags
  local arm="$1"; shift
  "${TIME_V[@]}" python scripts/evaluate_report_generation.py \
    --checkpoint "${CHECKPOINT}" --model-config hybrid_150m_m3_rrg \
    --parquet "${DATA}/test.parquet" --decode beam --beam-size 3 --max-new-tokens 100 \
    --dump-dir "${OUT}/${arm}" "$@" > "${OUT}/${arm}.log" 2>&1 \
    || { echo "ERROR: arm ${arm} failed; traceback frames follow (R7: no data lines)"
         grep -E '^Traceback|^  File |^[A-Za-z]*Error' "${OUT}/${arm}.log" | tail -20 | sed 's/^/[probe] /'; exit 1; }
  grep -E "Elapsed \(wall|Maximum resident|Missing keys|prefix_k =" "${OUT}/${arm}.log" | sed "s/^[[:space:]]*/[probe] ${arm}: /" || true
}
run cached_1 --num-samples 1 --cached-decode        # load cost, to subtract
run cached_a --num-samples "${N}" --cached-decode
run cached_b --num-samples "${N}" --cached-decode   # same node, same input: determinism
run uncached --num-samples "${N_UNCACHED}"          # the published path, on CPU

python - "${OUT}" "${PUBLISHED}" "${N}" "${N_UNCACHED}" <<'EOF'
import json, sys
out, published, n, nu = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
pub = open(published).read().splitlines()
def lines(arm):
    return open("{}/{}/hyps.txt".format(out, arm)).read().splitlines()
a, b, u = lines("cached_a"), lines("cached_b"), lines("uncached")
res = {
    "n": n, "n_uncached": nu,
    "cached_vs_cached_differ": sum(x != y for x, y in zip(a, b)),
    "cached_vs_uncached_cpu_differ": sum(x != y for x, y in zip(a[:nu], u)),
    "cached_vs_published_gpu_differ": sum(x != y for x, y in zip(a, pub[:n])),
    "uncached_vs_published_gpu_differ": sum(x != y for x, y in zip(u, pub[:nu])),
    "differing_rows": [i for i, (x, y) in enumerate(zip(a, pub[:n])) if x != y],
}
print("RESULT " + json.dumps(res))
json.dump(res, open("{}/summary.json".format(out), "w"), indent=2)
EOF
echo "=== seconds per report: (wall(cached_a) - wall(cached_1)) / (N - 1); uncached: (wall - load) / N_UNCACHED ==="
```

4. Parity test passes locally. Task loop steps 5–7, commit `"P1-C: CPU decode probe wrapper"`.
5. `bash scripts/chat_remote.sh sync && bash scripts/chat_remote.sh submit scripts/chat_cpu_decode_probe_h100.sh`. When it ends, `summary logs/chat_cpu_decode_probe_<id>.log` gives the `RESULT` line, wall times, RSS (and `lscpu`'s model line, which the wrapper prints as `[probe] cpu …`). Record: `tick P1-C --evidence job=<id> log=logs/chat_cpu_decode_probe_<id>.log s_per_report_cached=<…> s_per_report_uncached=<…> cpu_vs_gpu_differ=<k>/20 rss_gb=<…>`, and compare with the P1 predictions.

**As built (69944cb, 25419d5):** the committed wrapper runs a throwaway `warm` arm first, requires `/usr/bin/time`, prints peak RSS as `peak_rss_mb=`, masks exception messages in its failure path, emits RESULT as two lines (run health with observed counts, then drift) over the arms that finished, and exits 1 if any arm failed. **Results:** job 2590151 — CPU cached vs published GPU decode 0/20 differ, uncached 0/5, cached run-to-run 0/20; job 2590184 (warm) — cached 3.05 s per report ((177.76 − 119.86)/19), model load on CPU ≈ 117 s, peak RSS 4.5 GB, Xeon Platinum 8480CL, 8 threads.

- [x] **P1-D** Decision record in `chat_ui_state.json["decisions"]`: transport, bind policy, serving device, default `cached_decode`, whether `compile` is offered, and the drift note every card will show.

Pre-registered decision tree (user decision U7: the target is 8 s per turn, and a GPU job takes over automatically when CPU cannot meet it):

- Estimated turn = cached CPU seconds per report + 1.0 s (encode, retrieve, label). **≤ 8 s → serve on CPU** (`scripts/serve_chat_h100.sh`).
- **> 8 s, or `cached_a ≠ cached_b` → serve on GPU** (`scripts/serve_chat_gpu_h100.sh`, P7-B): no further question. On GPU the uncached path reproduces the published dump exactly (P2-E), and the drift note says so.
- `compile` is not offered on CPU (`--allow-compile` off), because E7 verified it on GPU only and Inductor's CPU build cost is unmeasured. On GPU it may be allowed after P2-E's GPU arm.
- If CPU vs GPU differs on k of 20 with k > 0, every card's provenance says: "CPU decode; differs from the published GPU decode on k/20 probe studies". If k = 0, it says "matched the published GPU decode on 20/20 probe studies".

### P2 — Engine (laptop first, then one golden job)

Gate: `pytest tests/test_app_engine.py tests/test_app_imaging.py` green. Golden job: engine == script on the same CPU node (0/20 differ); on GPU, the engine's uncached output == the published dump (0/20 differ).

- [x] **P2-A** `app/requirements.txt`, installed into `venv/`; `app` added to the PEP 604/585 scan roots.

**Files:** create `app/requirements.txt`, `tests/test_app_deps.py`; modify `tests/test_willi_parity.py` (`SCAN_ROOTS`, currently `[REPO_ROOT / "hybrid_xmamba", REPO_ROOT / "scripts"]`).

1. Failing test:

```python
# tests/test_app_deps.py
"""CHAT_UI_PLAN.md P2-A: the app's three runtime deps are importable in this interpreter."""
import importlib.util


def test_app_dependencies_import():
    import fastapi  # noqa: F401
    import uvicorn  # noqa: F401
    from fastapi.testclient import TestClient  # noqa: F401
    assert importlib.util.find_spec("python_multipart") or importlib.util.find_spec("multipart")


def test_app_is_scanned_for_pep604_and_pep585():
    from tests.test_willi_parity import SCAN_ROOTS, REPO_ROOT
    assert REPO_ROOT / "app" in SCAN_ROOTS
```

2. Run `venv/bin/python -m pytest tests/test_app_deps.py -v`. Expected: FAIL, `ModuleNotFoundError: No module named 'fastapi'`.
3. Implement:

```
# app/requirements.txt — the chat app's own deps (CHAT_UI_PLAN.md P2-A). Everything else it uses is
# already in requirements.txt. httpx is what FastAPI's TestClient runs on.
fastapi>=0.115
uvicorn>=0.30
python-multipart>=0.0.9
httpx>=0.27
```

Run `venv/bin/pip install -r app/requirements.txt` and record the installed versions in the evidence. Edit `SCAN_ROOTS = [REPO_ROOT / "hybrid_xmamba", REPO_ROOT / "scripts", REPO_ROOT / "app"]`.
4. `venv/bin/python -m pytest tests/test_app_deps.py tests/test_willi_parity.py -q` → PASS.
5–7. Task loop. Commit `"P2-A: app deps; app/ scanned for PEP 604/585"`.

- [x] **P2-B** `on_step` callback on `beam_search_cached` and `beam_search_decode`, with parity tests; tiny decoder in `app/tiny.py`.

**Files:** create `app/tiny.py`, `tests/test_app_engine.py`; modify `hybrid_xmamba/models/hybrid_lm.py` (`beam_search_cached`, lines 510–570; typing import at line 13), `scripts/evaluate_report_generation.py` (`beam_search_decode`, lines 151–206).
**Produces:** `beam_search_cached(..., on_step: Optional[Callable[[int, List[int]], None]] = None)` and the same keyword on `beam_search_decode`. `app.tiny.TINY_VOCAB` (exactly 97 entries), `tiny_decoder_config() -> HybridConfig`, `tiny_decoder(seed: int = 0) -> HybridLanguageModel`, `TinyTokenizer.decode(ids, skip_special_tokens=True) -> str`.

1. Failing tests:

```python
# tests/test_app_engine.py
"""CHAT_UI_PLAN.md P2-B/P2-D: engine stages and the on_step callback (tiny model, CPU)."""
import pytest
import torch

from app.tiny import TINY_VOCAB, tiny_decoder

EMPTY = torch.zeros((1, 0), dtype=torch.long)   # report generation seeds with no BOS


def _prefix(dim=64, k=4):
    return torch.randn(1, k, dim, generator=torch.Generator().manual_seed(1))


def test_tiny_vocab_matches_the_m6d_config():
    assert len(TINY_VOCAB) == 97 and len(set(TINY_VOCAB)) == 97


def test_on_step_leaves_cached_beam_output_unchanged():
    model = tiny_decoder()
    plain = model.beam_search_cached(EMPTY, prefix_embeds=_prefix(), beam_size=3, max_new_tokens=8)
    seen = []
    hooked = model.beam_search_cached(EMPTY, prefix_embeds=_prefix(), beam_size=3, max_new_tokens=8,
                                      on_step=lambda step, ids: seen.append((step, ids)))
    assert torch.equal(plain, hooked)
    assert [s for s, _ in seen] == list(range(8))
    assert [len(ids) for _, ids in seen] == list(range(1, 9))
    assert seen[-1][1] == hooked[0].tolist()          # the last snapshot is the answer


def test_on_step_leaves_uncached_beam_output_unchanged():
    from scripts.evaluate_report_generation import beam_search_decode
    model = tiny_decoder()
    plain = beam_search_decode(model, EMPTY, prefix_embeds=_prefix(), beam_size=3, max_new_tokens=6)
    seen = []
    hooked = beam_search_decode(model, EMPTY, prefix_embeds=_prefix(), beam_size=3, max_new_tokens=6,
                                on_step=lambda step, ids: seen.append(ids))
    assert torch.equal(plain, hooked) and seen[-1] == hooked[0].tolist()


def test_exception_from_on_step_stops_decoding():   # the chat app's cancel mechanism
    class Stop(Exception):
        pass
    calls = []

    def cb(step, ids):
        calls.append(step)
        if step == 2:
            raise Stop()

    with pytest.raises(Stop):
        tiny_decoder().beam_search_cached(EMPTY, prefix_embeds=_prefix(), beam_size=3,
                                          max_new_tokens=8, on_step=cb)
    assert calls == [0, 1, 2]


def test_cached_beam_of_one_equals_the_published_greedy():   # D11
    from scripts.evaluate_report_generation import greedy_decode
    model = tiny_decoder()
    greedy = greedy_decode(model, EMPTY, prefix_embeds=_prefix(), max_new_tokens=8)
    beam1 = model.beam_search_cached(EMPTY, prefix_embeds=_prefix(), beam_size=1, max_new_tokens=8)
    assert torch.equal(greedy, beam1)
```

2. Run `venv/bin/python -m pytest tests/test_app_engine.py -v`. Expected: FAIL (`app.tiny` missing, then `unexpected keyword argument 'on_step'`).
3. Implement:

```python
# app/tiny.py
"""Tiny random-init stand-ins for laptop work (CHAT_UI_PLAN.md P2-B, P2-D). No weights, no data.

The decoder config is the one tests/test_mamba3_numerics.py::_cached_lm proves token-identical
between the cached and uncached beam search (M6-D), so the tiny engine exercises the real cached
path. Ids decode to radiology words so the UI shows something report-shaped.
"""
from typing import List, Sequence

import torch
import torch.nn as nn

TINY_VOCAB = [
    "Findings:", "Impression:", "The", "the", "heart", "size", "is", "normal", "mediastinal", "contours",
    "are", "within", "limits", "lungs", "clear", "no", "focal", "consolidation", "pleural", "effusion",
    "or", "pneumothorax", "seen", "there", "mild", "moderate", "small", "large", "left", "right",
    "bilateral", "basilar", "atelectasis", "opacity", "opacities", "pulmonary", "edema", "vascular",
    "congestion", "cardiomegaly", "enlarged", "cardiac", "silhouette", "stable", "unchanged", "compared",
    "to", "prior", "study", "of", "and", "with", "without", "evidence", "acute", "cardiopulmonary",
    "process", "abnormality", "lobe", "upper", "lower", "middle", "chest", "tube", "endotracheal",
    "nasogastric", "line", "catheter", "tip", "terminates", "in", "pacemaker", "leads", "sternotomy",
    "wires", "fracture", "rib", "degenerative", "changes", "spine", "hilar", "aortic", "knob",
    "calcified", "granuloma", "nodule", "lesion", "pneumonia", "infection", "may", "be", "present",
    "likely", "low", "volumes", ".", ",",
]


def tiny_decoder_config():
    from hybrid_xmamba.models.configuration_hybrid import HybridConfig
    return HybridConfig(vocab_size=len(TINY_VOCAB), dim=64, num_layers=4, layer_pattern=["mamba3", "mlstm"],
                        head_dim=16, num_heads=4, max_position_embeddings=512, tfla_impl="exact",
                        mamba3_d_state=32, mamba3_head_dim=16, mamba3_chunk_size=8)


def tiny_decoder(seed: int = 0):
    from hybrid_xmamba.models.hybrid_lm import HybridLanguageModel
    torch.manual_seed(seed)
    return HybridLanguageModel(tiny_decoder_config()).eval()


class TinyTokenizer:
    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return " ".join(TINY_VOCAB[i] for i in ids).replace(" .", ".").replace(" ,", ",")
```

`hybrid_lm.py`: extend the typing import to `from typing import Callable, List, Optional, Tuple, Union`. Add `on_step: Optional[Callable[[int, List[int]], None]] = None` as the last parameter of `beam_search_cached`. Rename the loop variable to `step`, and right after `caches = self.reorder_cache(caches, beam_idx)` and before `logits = self.step_logits(...)` add:

```python
                if on_step is not None:
                    # CHAT_UI_PLAN.md P2-B: observe only. All beams have the same length here, so
                    # argmax(scores) is also the length-penalised best. Raising stops decoding.
                    on_step(step, tokens[int(torch.argmax(scores))].tolist())
```

Add to its docstring: "`on_step(step, best_ids)` is called once per step with the current best beam's ids, before that step's forward. It only observes. Raising from it stops decoding, which is how the chat app cancels a turn. The default `None` leaves this method unchanged."

`evaluate_report_generation.py`: add `on_step=None` to `beam_search_decode`, rename `for _ in range(max_new_tokens):` to `for step in range(max_new_tokens):`, and after `beams = candidates[:beam_size]` add:

```python
        if on_step is not None:   # CHAT_UI_PLAN.md P2-B: observe only; beams[0] is the current best
            on_step(step, beams[0][1][0].tolist())
```

4. `venv/bin/python -m pytest tests/test_app_engine.py tests/test_mamba3_numerics.py -k "beam or on_step or tiny" -v` → PASS, including the existing `test_cached_beam_search_is_token_identical_to_the_uncached_one`.
5–7. Task loop. Commit `"P2-B: on_step callback on both beam searches (parity-tested); tiny decoder"`.

- [x] **P2-C** `app/imaging.py`: upload sniffing and decoding (16-bit, EXIF, first frame, bounds), the published transform, the model-input image, thumbnails.

**Files:** create `app/imaging.py`, `tests/test_app_imaging.py`, `tests/app_helpers.py`.
**Produces:** `UploadError(ValueError)` (its message is shown to the user verbatim); `sniff_format(data: bytes) -> str` (`"PNG"|"JPEG"|"WEBP"`); `load_upload(data: bytes) -> Tuple[Image.Image, Dict[str, Any]]`; `model_transform() -> Callable`; `model_input_image(img) -> Image.Image`; `thumbnail_jpeg(img, max_side: int = 512) -> bytes`; constants `CLIP_MEAN`, `CLIP_STD`, `MAX_UPLOAD_BYTES`, `MIN_SIDE`, `MAX_PIXELS`.

1. Failing tests:

```python
# tests/app_helpers.py
"""Shared helpers for the chat-app tests (CHAT_UI_PLAN.md §8)."""
import io
import json
from typing import Dict, Iterable, Iterator

import numpy as np
from PIL import Image


def png_bytes(w: int = 320, h: int = 320, mode: str = "L") -> bytes:
    arr = (np.random.default_rng(0).random((h, w)) * 255).astype(np.uint8)
    img = Image.fromarray(arr).convert(mode)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def jpeg_bytes(w: int = 320, h: int = 320) -> bytes:
    buf = io.BytesIO()
    Image.open(io.BytesIO(png_bytes(w, h))).convert("RGB").save(buf, "JPEG", quality=92)
    return buf.getvalue()


def iter_sse(chunks: Iterable[str]) -> Iterator[Dict]:
    """The Python twin of app/static/api.js parseSSE: yields {"event", "data"}; skips comments."""
    buf = ""
    for chunk in chunks:
        buf = (buf + chunk).replace("\r\n", "\n")
        while "\n\n" in buf:
            frame, buf = buf.split("\n\n", 1)
            event, data = "message", []
            for line in frame.split("\n"):
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:][1:] if line[5:].startswith(" ") else line[5:])
            if data:
                yield {"event": event, "data": json.loads("\n".join(data))}
```

```python
# tests/test_app_imaging.py
"""CHAT_UI_PLAN.md P2-C: uploads become exactly what the published transform expects."""
import io

import numpy as np
import pytest
import torch
import torchvision.transforms as T
from PIL import Image

from app.imaging import (CLIP_MEAN, CLIP_STD, UploadError, load_upload, model_input_image,
                         model_transform, thumbnail_jpeg)
from tests.app_helpers import png_bytes

PUBLISHED = T.Compose([   # verbatim from scripts/evaluate_report_generation.py run_checkpoint_inspection
    T.Resize((224, 224)), T.Grayscale(num_output_channels=3), T.ToTensor(),
    T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])


def _gray_rgb(w=320, h=320):
    return Image.open(io.BytesIO(png_bytes(w, h))).convert("RGB")


def test_16bit_png_is_rescaled_not_saturated():   # Review Focus 1
    ramp = np.linspace(0, 4095, 256 * 256).reshape(256, 256).astype(np.uint16)   # 12-bit data
    buf = io.BytesIO()
    Image.fromarray(ramp).save(buf, "PNG")
    img, facts = load_upload(buf.getvalue())
    arr = np.asarray(img.convert("L"))
    assert arr.min() <= 5 and arr.max() >= 250, (arr.min(), arr.max())
    assert facts["mode"].startswith("I")


def test_exif_rotation_is_applied_before_the_model_sees_it():   # Review Focus 5
    base = Image.new("RGB", (200, 100), (0, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6                                   # display rotated 90° clockwise
    buf = io.BytesIO()
    base.save(buf, "JPEG", exif=exif.tobytes())
    img, facts = load_upload(buf.getvalue())
    assert img.size == (100, 200) and facts["exif_transposed"] is True


@pytest.mark.parametrize("payload, needle", [
    (b"GIF89a" + b"\x00" * 200, "PNG, JPEG or WEBP"),
    (b"\x00" * 128 + b"DICM" + b"\x00" * 200, "DICOM"),
    (b"just some text", "PNG, JPEG or WEBP"),
])
def test_unsupported_bytes_get_a_readable_message(payload, needle):
    with pytest.raises(UploadError, match=needle):
        load_upload(payload)


def test_too_small_and_too_large_are_refused():
    with pytest.raises(UploadError, match="64"):
        load_upload(png_bytes(32, 32))
    with pytest.raises(UploadError, match="20 MB"):
        load_upload(b"\x89PNG\r\n\x1a\n" + b"\x00" * (20 * 1024 * 1024 + 1))


def test_model_transform_is_the_published_one():
    img = _gray_rgb()
    assert torch.equal(model_transform()(img), PUBLISHED(img))
    assert list(CLIP_MEAN) == [0.48145466, 0.4578275, 0.40821073]


def test_retrieval_transform_gives_identical_tensors_for_grayscale_origin_images():   # D4
    from scripts.evaluate_cxr_retrieval import _img_transform
    img = _gray_rgb(300, 340)
    assert torch.equal(model_transform()(img), _img_transform()(img))


def test_model_input_image_is_what_the_tower_sees():
    img = _gray_rgb()
    shown = model_input_image(img)
    assert shown.size == (224, 224)
    rebuilt = T.Normalize(mean=list(CLIP_MEAN), std=list(CLIP_STD))(T.ToTensor()(shown))
    assert torch.equal(rebuilt, model_transform()(img))


def test_thumbnail_is_a_bounded_jpeg():
    data = thumbnail_jpeg(_gray_rgb(1200, 900))
    thumb = Image.open(io.BytesIO(data))
    assert thumb.format == "JPEG" and max(thumb.size) == 512
```

2. Run `venv/bin/python -m pytest tests/test_app_imaging.py -v`. Expected: FAIL (module missing).
3. Implement:

```python
# app/imaging.py
"""Upload intake and the published image transform (CHAT_UI_PLAN.md P2-C).

The model must see exactly what every published number saw: Resize((224,224)) -> Grayscale(3) ->
ToTensor -> Normalize(CLIP mean/std), the transform in run_checkpoint_inspection. Uploads are not
MIMIC JPEGs, so two things happen first that MIMIC never needed: EXIF orientation is applied (the
model must see what the user sees) and 16-bit grayscale is rescaled to 8-bit (PIL's convert would
clip it to white). For an 8-bit, EXIF-free grayscale JPEG, both are no-ops.
"""
import io
from typing import Any, Callable, Dict, Tuple

import numpy as np
from PIL import Image, ImageOps

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MIN_SIDE = 64
MAX_PIXELS = 50_000_000
FORMATS_MSG = "Use a PNG, JPEG or WEBP image."


class UploadError(ValueError):
    """Shown to the user verbatim."""


def sniff_format(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if data[:3] == b"\xff\xd8\xff":
        return "JPEG"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    if data[128:132] == b"DICM":
        raise UploadError("DICOM is not supported. Export the image as PNG or JPEG first.")
    raise UploadError(FORMATS_MSG)


def _to_8bit(img: Image.Image) -> Image.Image:
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.asarray(img, dtype=np.float64)
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / max(hi - lo, 1.0) * 255.0
        return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))
    return img


def load_upload(data: bytes) -> Tuple[Image.Image, Dict[str, Any]]:
    """bytes -> the 8-bit RGB image the model and the viewer both use, plus facts for the stage detail."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError("Image is larger than 20 MB.")
    fmt = sniff_format(data)
    try:
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > MAX_PIXELS:
            raise UploadError("Image is larger than 50 megapixels.")
        img.seek(0)
        img.load()
    except UploadError:
        raise
    except Exception as exc:   # truncated or corrupt file, decompression bomb
        raise UploadError("Could not read the image ({}).".format(type(exc).__name__))
    mode_in, size_in = img.mode, img.size
    transposed = ImageOps.exif_transpose(img)
    exif_transposed = transposed.size != size_in or transposed is not img
    img = _to_8bit(transposed).convert("RGB")
    if min(img.size) < MIN_SIDE:
        raise UploadError("Image is smaller than 64×64 pixels.")
    return img, {"format": fmt, "mode": mode_in, "input_px": list(img.size), "exif_transposed": bool(exif_transposed)}


def model_transform() -> Callable:
    import torchvision.transforms as T
    return T.Compose([T.Resize((224, 224)), T.Grayscale(num_output_channels=3), T.ToTensor(),
                      T.Normalize(mean=list(CLIP_MEAN), std=list(CLIP_STD))])


def model_input_image(img: Image.Image) -> Image.Image:
    """The 224×224 image the tower sees, before ToTensor/Normalize (lossless as PNG)."""
    import torchvision.transforms as T
    return T.Grayscale(num_output_channels=3)(T.Resize((224, 224))(img))


def thumbnail_jpeg(img: Image.Image, max_side: int = 512) -> bytes:
    thumb = img.copy()
    thumb.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    thumb.convert("RGB").save(buf, "JPEG", quality=88)
    return buf.getvalue()
```

Check during implementation: `ImageOps.exif_transpose` returns a copy whenever EXIF data exists, even without rotation. If so, set `exif_transposed` from the orientation tag (`img.getexif().get(0x0112, 1) != 1`) instead, so an un-rotated JPEG reports `False`.

4. `venv/bin/python -m pytest tests/test_app_imaging.py -v` → PASS.
5–7. Task loop. Commit `"P2-C: upload intake + published transform (16-bit, EXIF handled)"`.

- [x] **P2-D** `app/engine.py`: `Engine` stages (preprocess, encode, generate) and card; `RealEngine`; `TinyEngine`; one tower pass; streamed checkpoint hash.

**Files:** create `app/engine.py`, `app/schemas.py` (only `Options` from §6.1 for now; P3-A adds the rest); extend `app/tiny.py` (`TinyTower`) and `tests/test_app_engine.py`.
**Consumes:** `app.imaging.*`, `app.tiny.*`, `scripts.evaluate_report_generation.load_report_generation_module`, `beam_search_decode`, `scripts.repair_generations.repair_report`.
**Produces:** §6.6 engine interfaces; `file_sha256(path: Path, cache_file: Optional[Path] = None) -> str`; `tensor_sha256(module: nn.Module) -> str`.

1. Failing tests (append to `tests/test_app_engine.py`):

```python
import hashlib
import threading

from app.engine import Cancelled, build_engine, file_sha256
from app.schemas import Options
from tests.app_helpers import png_bytes


def _noop(step, text):
    pass


def _encoded(eng):
    _, prep = eng.preprocess(png_bytes(320, 320))
    _, enc = eng.encode(prep)
    return enc


def test_tiny_engine_runs_three_stages_and_streams_snapshots():
    eng = build_engine("tiny")
    res_p, prep = eng.preprocess(png_bytes(320, 320))
    assert prep.pixel_values.shape == (1, 3, 224, 224) and res_p.detail["resized_to"] == [224, 224]
    res_e, enc = eng.encode(prep)
    assert enc.prefix.shape == (1, 4, 64) and abs(float(enc.pooled.norm()) - 1.0) < 1e-5
    assert res_e.detail["one_pass"] is True
    snaps = []
    res_g, gen = eng.generate(enc, Options(max_new_tokens=16), lambda s, t: snaps.append((s, t)),
                              threading.Event())
    assert [s for s, _ in snaps] == list(range(16))
    assert snaps[-1][1] == gen.report and len(gen.token_ids) == 16
    assert res_g.detail["stopped"] == "budget" and res_g.detail["cached_decode"] is True
    assert gen.report == " ".join(gen.report.split())          # sanitised like the dumps


def test_cancel_stops_generation_within_one_step():
    eng = build_engine("tiny")
    enc = _encoded(eng)
    cancel, seen = threading.Event(), []

    def snap(step, text):
        seen.append(step)
        if step == 3:
            cancel.set()

    with pytest.raises(Cancelled):
        eng.generate(enc, Options(max_new_tokens=50), snap, cancel)
    assert seen == [0, 1, 2, 3]


def test_uncached_and_greedy_paths_equal_the_published_functions():
    from scripts.evaluate_report_generation import beam_search_decode, greedy_decode
    eng = build_engine("tiny")
    enc = _encoded(eng)
    _, unc = eng.generate(enc, Options(max_new_tokens=16, cached_decode=False), _noop, threading.Event())
    ref = beam_search_decode(eng.decoder, EMPTY, prefix_embeds=enc.prefix, beam_size=3, max_new_tokens=16)
    assert unc.token_ids == ref[0].tolist()
    _, grd = eng.generate(enc, Options(max_new_tokens=16, decode="greedy"), _noop, threading.Event())
    assert grd.token_ids == greedy_decode(eng.decoder, EMPTY, prefix_embeds=enc.prefix, max_new_tokens=16)[0].tolist()


def test_display_repair_only_changes_the_display_copy():
    eng = build_engine("tiny")
    enc = _encoded(eng)
    _, raw = eng.generate(enc, Options(max_new_tokens=16), _noop, threading.Event())
    _, rep = eng.generate(enc, Options(max_new_tokens=16, display_repair=True), _noop, threading.Event())
    assert raw.report == rep.report and raw.display_report == raw.report


def test_streamed_sha256_matches_hashlib_and_is_cached(tmp_path):
    blob = tmp_path / "w.bin"
    blob.write_bytes(b"x" * (3 * 1024 * 1024 + 7))
    cache = tmp_path / "sha256.json"
    assert file_sha256(blob, cache) == hashlib.sha256(blob.read_bytes()).hexdigest()
    assert str(blob.resolve()) in cache.read_text()


def test_one_tower_pass_gives_the_published_patch_grid_and_pooled_vector():
    open_clip = pytest.importorskip("open_clip")
    try:
        model, _ = open_clip.create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    except Exception as exc:   # no network and no local cache
        pytest.skip("BiomedCLIP unavailable: {}".format(exc))
    v = model.visual.eval()
    x = torch.randn(1, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        feats = v.trunk.forward_features(x)                    # == ReportGenerationLightningModule._patch_grid
        assert feats.shape == (1, 197, 768)
        assert torch.equal(v.head(v.trunk.forward_head(feats)), v(x))
```

2. Run: FAIL (`app.engine` missing).
3. Implement `app/engine.py`:

```python
"""Inference engine for the chat app (CHAT_UI_PLAN.md P2-D).

A thin layer over code the thesis already trusts: load_report_generation_module (prefix_k from
run_metadata.json, the operator flags, the missing-key guard), the published transform, and the two
beam searches with the observe-only on_step callback (P2-B). Nothing here re-implements decoding.
"""
import hashlib
import json
import os
import platform
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from app.imaging import load_upload, model_input_image, model_transform

DISCLAIMER = "Research prototype; not for clinical use."
# StageResult, Prepared, Encoded, Generated, Cancelled exactly as in CHAT_UI_PLAN.md §6.6.


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 1)


def file_sha256(path: Path, cache_file: Optional[Path] = None, chunk: int = 1 << 20) -> str:
    """Streamed SHA-256 (D12), cached by (path, size, mtime) so a 2.4 GB checkpoint is hashed once."""
    path = Path(path).resolve()
    st = path.stat()
    key = "{}|{}|{}".format(path, st.st_size, int(st.st_mtime))
    cache = {}
    if cache_file is not None and Path(cache_file).exists():
        cache = json.loads(Path(cache_file).read_text())
        if key in cache:
            return cache[key]
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    digest = h.hexdigest()
    if cache_file is not None:
        cache[key] = digest
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_file).write_text(json.dumps(cache, indent=1))
    return digest


def tensor_sha256(module: torch.nn.Module) -> str:
    """Fingerprint of a module's weights: fp32 bytes of every state-dict tensor, sorted by key."""
    h = hashlib.sha256()
    sd = module.state_dict()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().float().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


class Engine:
    """Shared stage logic. Subclasses set name, device, tower, prefix_mapper, decoder, tokenizer."""

    name = "engine"
    device = torch.device("cpu")
    drift_note = ""

    def preprocess(self, data: bytes):
        t0 = time.perf_counter()
        img, facts = load_upload(data)
        return self._prepared(img, dict(facts, image_sha256=hashlib.sha256(data).hexdigest()), t0)

    def _prepared(self, img, facts, t0):
        pixel_values = model_transform()(img).unsqueeze(0)
        detail = dict(facts, resized_to=[224, 224], grayscale_to_3ch=True, normalize="biomedclip_clip_mean_std")
        prepared = Prepared(img, model_input_image(img), pixel_values, facts["image_sha256"], facts)
        return StageResult(detail, _ms(t0)), prepared

    def encode(self, prepared):
        t0 = time.perf_counter()
        with torch.no_grad():
            px = prepared.pixel_values.to(self.device)
            feats = self.tower.trunk.forward_features(px)          # == module._patch_grid(px)
            pooled = F.normalize(self.tower.head(self.tower.trunk.forward_head(feats)).float(), dim=-1)[0]
            prefix = self.prefix_mapper(feats)
        detail = {"patch_grid": list(feats.shape[1:]), "pooled_dim": int(pooled.shape[-1]),
                  "prefix_tokens": int(prefix.shape[1]), "device": str(self.device), "one_pass": True}
        return StageResult(detail, _ms(t0)), Encoded(feats, pooled.cpu(), prefix)

    def _decode_text(self, ids: List[int]) -> str:
        return " ".join(self.tokenizer.decode(ids, skip_special_tokens=True).split())

    def generate(self, enc, opts, on_snapshot, cancel):
        from scripts.evaluate_report_generation import beam_search_decode
        from scripts.repair_generations import repair_report

        if opts.cached_decode and not self.decoder.supports_cached_decode():
            raise ValueError("{} has no O(1) decode cache; set cached_decode=false.".format(self.name))
        t0 = time.perf_counter()
        first = []

        def cb(step: int, ids: List[int]) -> None:
            if cancel.is_set():
                raise Cancelled()
            if not first:
                first.append(time.perf_counter())
            on_snapshot(step, self._decode_text(ids))

        beam = 1 if opts.decode == "greedy" else opts.beam_size
        empty = torch.zeros((1, 0), dtype=torch.long, device=self.device)   # no BOS, as in training
        with torch.no_grad():
            if opts.cached_decode:
                out = self.decoder.beam_search_cached(empty, prefix_embeds=enc.prefix, beam_size=beam,
                                                      max_new_tokens=opts.max_new_tokens, on_step=cb)
            else:
                out = beam_search_decode(self.decoder, empty, prefix_embeds=enc.prefix, beam_size=beam,
                                         max_new_tokens=opts.max_new_tokens, on_step=cb)
        ids = out[0].tolist()
        report = self._decode_text(ids)
        repaired, stats = repair_report(report, dedup="none", truncate=True)
        total = time.perf_counter() - t0
        start = first[0] if first else t0
        detail = {"decode": opts.decode, "beam_size": beam, "tokens": len(ids), "stopped": "budget",
                  "cached_decode": bool(opts.cached_decode), "compiled": False,
                  "prefill_ms": round((start - t0) * 1000.0, 1),
                  "per_token_ms": round((time.perf_counter() - start) * 1000.0 / max(len(ids) - 1, 1), 2),
                  "device": str(self.device), "threads": torch.get_num_threads(), "drift_note": self.drift_note}
        gen = Generated(ids, report, repaired if opts.display_repair else report,
                        stats.get("sentences_truncated", 0) > 0)
        return StageResult(detail, round(total * 1000.0, 1)), gen
```

`TinyEngine(Engine)`: `name="tiny"`; `tower=TinyTower()` (seeded; see below); `prefix_mapper=ImagePrefixMapper(patch_dim=32, decoder_dim=64, k=4)` (seeded); `decoder=tiny_decoder()`; `tokenizer=TinyTokenizer()`; optional `step_delay_s` that sleeps inside `on_snapshot` (for the API tests); `card()` returns `{"name": "tiny", "checkpoint": None, "cached_decode_available": True, "prefix_k": 4, ...}`; `drift_note = "tiny random-init model"`.

```python
# app/tiny.py (append)
class _TinyTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 32, kernel_size=16, stride=16)
        self.cls = nn.Parameter(torch.zeros(1, 1, 32))

    def forward_features(self, x):
        p = self.patch(x).flatten(2).transpose(1, 2)                       # (B, 196, 32)
        return torch.cat([self.cls.expand(x.shape[0], -1, -1), p], dim=1)  # (B, 197, 32)

    def forward_head(self, feats):
        return feats[:, 0]


class TinyTower(nn.Module):
    """Stand-in for open_clip's TimmModel: same trunk/head surface, 32-d patches, 16-d pooled output."""

    def __init__(self, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.trunk = _TinyTrunk()
        self.head = nn.Linear(32, 16)

    def forward(self, x):
        return self.head(self.trunk.forward_head(self.trunk.forward_features(x)))
```

`RealEngine(Engine)`:

```python
class RealEngine(Engine):
    def __init__(self, checkpoint: str, model_config: str, device: str = "cpu", threads: int = 8,
                 cache_dir: Optional[Path] = None, drift_note: str = ""):
        from scripts.evaluate_report_generation import load_report_generation_module
        from transformers import AutoTokenizer
        torch.set_num_threads(threads)
        self.name, self.device, self.drift_note = model_config, torch.device(device), drift_note
        self.module = load_report_generation_module(checkpoint, model_config, device=device)
        self.tower, self.prefix_mapper, self.decoder = (self.module.image_encoder, self.module.prefix_mapper,
                                                        self.module.decoder)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self._card = self._provenance(Path(checkpoint), cache_dir)

    def _provenance(self, ckpt: Path, cache_dir: Optional[Path]) -> Dict[str, Any]:
        cfg = self.decoder.config
        meta = ckpt.resolve().parent.parent / "run_metadata.json"
        git = lambda *a: subprocess.run(["git"] + list(a), capture_output=True, text=True).stdout.strip()
        return {"name": self.name, "checkpoint": str(ckpt),
                "checkpoint_sha256": file_sha256(ckpt, (cache_dir / "sha256.json") if cache_dir else None),
                "prefix_k": int(self.module.prefix_k), "scan_impl": cfg.scan_impl, "tfla_impl": cfg.tfla_impl,
                "mamba3_chunk_size": getattr(cfg, "mamba3_chunk_size", None),
                "layer_pattern": list(cfg.layer_pattern),
                "cached_decode_available": bool(self.decoder.supports_cached_decode()),
                "train_experiment": json.loads(meta.read_text()).get("experiment_name") if meta.exists() else None,
                "torch": torch.__version__, "device": str(self.device), "threads": torch.get_num_threads(),
                "cpu": platform.processor() or platform.machine(),
                "git_sha": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
                "drift_note": self.drift_note}

    def card(self) -> Dict[str, Any]:
        return dict(self._card)

    def tower_sha256(self) -> str:
        return tensor_sha256(self.tower)
```

Check during implementation: `module.prefix_k` is the attribute name `ReportGenerationLightningModule` uses (`grep -n "self.prefix_k" hybrid_xmamba/training/lightning_module.py`). Check also the key name in `run_metadata.json` for the experiment (`resolved_config` holds the Hydra config). Adjust both to what the code says.

On the cluster, `CLUSTER_REPO` has no `.git` (P0-G keeps it out of the rsync). When `git rev-parse HEAD` fails, `git_sha` and `git_dirty` come from the repo-root `.sync_stamp` file (`<UTC time> <sha> <clean|dirty>`, written by `chat_remote.sh sync`), and the card's `git_source` says `git` or `sync_stamp`. Test both with a temporary directory.

`build_engine(kind, **kw)`: `"tiny"` → `TinyEngine(**kw)`, `"real"` → `RealEngine(**kw)`, otherwise `ValueError`.

4. `venv/bin/python -m pytest tests/test_app_engine.py -v` → PASS (the BiomedCLIP test passes from the local HF cache, or skips).
5–7. Task loop. Commit `"P2-D: engine stages over the published loaders; tiny engine"`.

**As built (d54895a):** the committed `app/engine.py` and `app/tiny.py` are authoritative where they differ from the code above: `TinyTower` mean-pools its patch tokens (the CLS slot is a zero parameter, so `feats[:, 0]` gave every image the same vector); the experiment name is read from `run_metadata.json["resolved_config"]["experiment_name"]`; `truncated_mid_sentence` also counts `repair_report`'s no-complete-sentence fallback; provenance comes from `git_provenance(root)` (git at the repo toplevel only, else `.sync_stamp`, unknown stays `None`) and the card carries `git_source`; `generate` raises `Cancelled` before the prefill when the event is already set; tiny models are built inside `torch.random.fork_rng(devices=[])` so the caller's RNG is untouched.

- [x] **P2-E** Golden job `scripts/chat_engine_golden.py` with CPU and GPU wrappers: engine vs script (same node) and vs the published GPU dump.

**Files:** create `scripts/chat_engine_golden.py`, `scripts/chat_engine_golden_h100.sh` (CPU), `scripts/chat_engine_golden_gpu_h100.sh` (GPU); modify `tests/test_willi_parity.py`.

Prediction (R4): engine vs script on the same CPU node 0/20 differ; GPU engine-uncached vs published 0/20; GPU engine-cached vs published 0/20 (M6-D at scale, first real-checkpoint measurement); CPU encode 0.3–1 s.

1. Failing parity test:

```python
def test_chat_engine_golden_wrappers_compare_on_the_same_node():
    """CHAT_UI_PLAN.md P2-E. R2: byte identity is only meaningful on the same node and device."""
    cpu = (REPO_ROOT / "scripts" / "chat_engine_golden_h100.sh").read_text()
    gpu = (REPO_ROOT / "scripts" / "chat_engine_golden_gpu_h100.sh").read_text()
    for src in (cpu, gpu):
        assert "#SBATCH --partition=pot-hpi-aisc-batch" in src and "--exclude=ga03" in src
        assert "scripts/chat_engine_golden.py" in src
    assert "scripts/evaluate_report_generation.py" in cpu   # the GPU arm compares with the published dump instead
    assert not [l for l in cpu.splitlines() if l.startswith("#SBATCH") and "--gpus" in l]
    assert "#SBATCH --gpus=1" in gpu and "--uncached" in gpu
```

2. Run: FAIL.
3. Implement the driver:

```python
# scripts/chat_engine_golden.py
"""CHAT_UI_PLAN.md P2-E: decode the first N test studies through app.engine.RealEngine.

    python scripts/chat_engine_golden.py --checkpoint … --model-config hybrid_150m_m3_rrg \
        --parquet …/test.parquet --n 20 --out results/chat_golden_<job>/engine_cached [--uncached]

Writes <out>/hyps.txt (one report per line, sanitised like write_hyps_refs) and <out>/timings.json.
The images go through Engine.preprocess from their file BYTES, the same path an upload takes.
Output is MIMIC-derived: never commit it.
"""
import argparse
import json
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-config", default="hybrid_150m_m3_rrg")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--uncached", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    import pandas as pd
    import torch
    from app.engine import build_engine
    from app.schemas import Options

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eng = build_engine("real", checkpoint=args.checkpoint, model_config=args.model_config,
                       device=device, threads=args.threads)
    df = pd.read_parquet(args.parquet).iloc[: args.n]
    opts = Options(cached_decode=not args.uncached)
    hyps, timings = [], []
    for i in range(len(df)):
        _, prep = eng.preprocess(Path(df.iloc[i]["image"]).read_bytes())
        enc_res, enc = eng.encode(prep)
        gen_res, gen = eng.generate(enc, opts, lambda s, t: None, threading.Event())
        hyps.append(gen.report)
        timings.append({"row": i, "encode_ms": enc_res.ms, **{k: gen_res.detail[k] for k in ("prefill_ms", "per_token_ms")},
                        "generate_ms": gen_res.ms})
        print("[golden] row {} {:.0f} ms".format(i, enc_res.ms + gen_res.ms), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "hyps.txt").write_text("\n".join(" ".join(h.split()) for h in hyps) + "\n")
    (out / "timings.json").write_text(json.dumps({"device": device, "card": eng.card(), "rows": timings}, indent=2))


if __name__ == "__main__":
    main()
```

```bash
#!/bin/bash
# ============================================================================
# CHAT_UI_PLAN.md P2-E (CPU) — the engine must reproduce the eval script byte
# for byte on the SAME node (R2). Also reports drift from the published GPU dump.
#   sbatch scripts/chat_engine_golden_h100.sh
# Output (DUA-covered, never commit): results/chat_golden_<job>/
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node, x86 .venv incompatible; gx13v1: faulty GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=chat_engine_golden
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
DATA="${DATA:-/sc/home/$USER/dataset/mimic_full}"
CHECKPOINT="${CHECKPOINT:-./outputs/h100_report_gen_m3_tower13d_s42/checkpoints/last.ckpt}"
PUBLISHED="${PUBLISHED:-results/report_gen_m3_test_split_s42/hyps.txt}"
N="${N:-20}"
OUT="${OUT:-results/chat_golden_${SLURM_JOB_ID:-local}}"
export HF_HOME="${SCRATCH_ROOT}/.hf" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
source "${VENV_ACTIVATE}"
for f in "${CHECKPOINT}" "${PUBLISHED}" "${DATA}/test.parquet"; do
  [ -f "$f" ] || { echo "ERROR: not found: $f"; exit 1; }
done
mkdir -p "${OUT}"
lscpu | grep -E "^Model name" || true

python scripts/evaluate_report_generation.py --checkpoint "${CHECKPOINT}" --model-config hybrid_150m_m3_rrg \
  --parquet "${DATA}/test.parquet" --num-samples "${N}" --decode beam --beam-size 3 --max-new-tokens 100 \
  --cached-decode --dump-dir "${OUT}/script_cached" > "${OUT}/script_cached.log" 2>&1
python scripts/chat_engine_golden.py --checkpoint "${CHECKPOINT}" --model-config hybrid_150m_m3_rrg \
  --parquet "${DATA}/test.parquet" --n "${N}" --threads "${SLURM_CPUS_PER_TASK:-8}" --out "${OUT}/engine_cached"

python - "${OUT}" "${PUBLISHED}" "${N}" <<'EOF'
import json, sys
out, published, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
read = lambda p: open(p).read().splitlines()
eng, scr, pub = read(out + "/engine_cached/hyps.txt"), read(out + "/script_cached/hyps.txt"), read(published)[:n]
res = {"n": n,
       "engine_vs_script_differ": sum(a != b for a, b in zip(eng, scr)),
       "engine_vs_published_gpu_differ": sum(a != b for a, b in zip(eng, pub)),
       "lengths": [len(eng), len(scr), len(pub)]}
print("RESULT " + json.dumps(res))
json.dump(res, open(out + "/golden.json", "w"), indent=2)
sys.exit(1 if res["engine_vs_script_differ"] or len(set(res["lengths"])) != 1 else 0)
EOF
```

`scripts/chat_engine_golden_gpu_h100.sh` is the same file with these differences: `#SBATCH --gpus=1` and no `--qos` line (as every GPU wrapper in the repo), `--time=01:00:00`, job name `chat_engine_golden_gpu`; it runs `chat_engine_golden.py --uncached --out "${OUT}/engine_uncached"` and `chat_engine_golden.py --out "${OUT}/engine_cached"` (no script arm), and its comparison writes `engine_uncached_vs_published_differ` and `engine_cached_vs_published_differ`, exiting 1 if the uncached count is non-zero.

4. Parity test passes. Task loop 5–7. Commit `"P2-E: engine golden driver + CPU/GPU wrappers"`.
5. On lx01: `sbatch scripts/chat_engine_golden_h100.sh` and `sbatch scripts/chat_engine_golden_gpu_h100.sh`. Tick with both job ids, the `golden.json` numbers and the per-stage timings.

### P3 — Store and streaming API (laptop, tiny engine)

Gate: `pytest tests/test_app_store.py tests/test_app_redact.py tests/test_app_api.py` green: stream framing and `seq` continuity, replay identity, resume, cancel, dropped stream, overload, auth, bind refusal, export, delete, public scoping.

- [x] **P3-A** `app/schemas.py` (`Options`, `error_body`) and `app/ids.py`; bounds tests.

**Files:** extend `app/schemas.py`; create `app/ids.py`, `tests/test_app_schemas.py`.
**Produces:** `error_body(kind: str, message: str) -> Dict` → `{"type": "error", "error": {"type": kind, "message": message}}`; `new_id(prefix: str) -> str` → `"<prefix>_" + 12 hex chars of ms time + 10 random hex chars` (sortable by creation time).

1. Failing tests:

```python
# tests/test_app_schemas.py
import pytest
from pydantic import ValidationError

from app.ids import new_id
from app.schemas import Options, error_body


def test_defaults_are_the_published_protocol():
    o = Options()
    assert (o.decode, o.beam_size, o.max_new_tokens, o.cached_decode) == ("beam", 3, 100, True)
    assert (o.k_images, o.k_reports, o.label, o.display_repair, o.compile) == (4, 3, True, False, False)


@pytest.mark.parametrize("bad", [
    {"beam_size": 0}, {"beam_size": 9}, {"max_new_tokens": 15}, {"max_new_tokens": 201},
    {"k_images": 13}, {"k_reports": 11}, {"retrieval_k": 3}, {"decode": "sample"}, {"test_row": -1},
])
def test_out_of_bounds_or_unknown_options_are_rejected(bad):
    with pytest.raises(ValidationError):
        Options(**bad)


def test_ids_sort_by_creation_and_do_not_collide():
    ids = [new_id("m") for _ in range(500)]
    assert len(set(ids)) == 500 and ids == sorted(ids) and ids[0].startswith("m_")


def test_error_envelope_shape():
    assert error_body("overloaded_error", "busy") == {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
```

2. Run: FAIL. 3. Implement the §6.1 class (with `from typing import Optional` and `from typing_extensions import Annotated, Literal` only if `typing` lacks them; Python ≥ 3.9 has both in `typing`). `new_id`: `"{}_{:012x}{}".format(prefix, int(time.time() * 1000), secrets.token_hex(5))`. Within the same millisecond the random suffix decides the order, so the sort test makes the id monotonic with a module-level `(last_ms, counter)` guard under a lock. 4. PASS. 5–7. Commit `"P3-A: options schema, error envelope, sortable ids"`.

- [x] **P3-B** `app/store.py`: schema §6.4, one transaction per event, restart recovery, uploads, client scoping, export, retention sweep.

**Files:** create `app/store.py`, `tests/test_app_store.py`.
**Produces:**

```python
class Store:
    def __init__(self, home: Path): ...                     # creates home/chat.db and home/uploads/
    def create_session(self, mode: str, client_id: Optional[str] = None, title: str = "") -> Dict[str, Any]: ...
    def list_sessions(self, client_id: Optional[str], limit: int = 50,
                      cursor: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]: ...
    def get_session(self, session_id: str, client_id: Optional[str]) -> Optional[Dict[str, Any]]: ...
    def delete_session(self, session_id: str, client_id: Optional[str]) -> bool: ...
    def last_image(self, session_id: str) -> Optional[Dict[str, Any]]: ...   # {"sha256","filename","test_row"}
    def start_turn(self, session_id: str, text: str, mode: str, options: Dict[str, Any],
                   image_sha256: Optional[str] = None, image_filename: Optional[str] = None,
                   test_row: Optional[int] = None) -> Tuple[str, str]: ...   # (user_message_id, message_id)
    def save_upload(self, session_id: str, sha256: str, original: bytes, ext: str,
                    thumb_jpeg: bytes, model_input_png: bytes) -> None: ...
    def upload_path(self, session_id: str, sha256: str, variant: str) -> Optional[Path]: ...
    def append_event(self, message_id: str, event: str, data: Dict[str, Any]) -> Dict[str, Any]: ...  # data + seq
    def events_after(self, message_id: str, after: int = 0) -> List[Dict[str, Any]]: ...   # [{"seq","event","data"}]
    def get_message(self, message_id: str, client_id: Optional[str]) -> Optional[Dict[str, Any]]: ...
    def finish_turn(self, message_id: str, status: str, report: Optional[str] = None,
                    display_report: Optional[str] = None, provenance: Optional[Dict[str, Any]] = None,
                    total_ms: Optional[float] = None) -> None: ...
    def recover_after_restart(self) -> int: ...
    def export(self, session_id: str, fmt: str, client_id: Optional[str]) -> Tuple[str, str, bytes]: ...
    def sweep(self, older_than_days: int) -> int: ...
    def close(self) -> None: ...                              # releases the EXCLUSIVE lock (D22)
```

The connection is `self._con` (tests use it to back-date rows). With `locking_mode=EXCLUSIVE`, a second `Store` on the same file can only open after the first is closed; the server's lifespan closes it at shutdown.

Scoping rule: `client_id=None` (a private server) sees every session; otherwise only rows whose `client_id` matches. `get_message` resolves the session first, so the same rule applies to messages and images.

1. Failing tests:

```python
# tests/test_app_store.py
"""CHAT_UI_PLAN.md P3-B: the event log is the source of truth and survives restarts."""
import json

from app.store import Store


def _turn(s, mode="private", client=None):
    sess = s.create_session(mode, client)
    user_id, mid = s.start_turn(sess["id"], "hi", mode, {"beam_size": 3})
    return sess, user_id, mid


def test_events_get_contiguous_seq_and_survive_reopen(tmp_path):
    s = Store(tmp_path)
    _, _, mid = _turn(s)
    assert [s.append_event(mid, "stage_start", {"stage": "x"})["seq"] for _ in range(3)] == [1, 2, 3]
    s.close()   # locking_mode=EXCLUSIVE (D22): one open connection owns the file
    reopened = Store(tmp_path)
    assert [e["seq"] for e in reopened.events_after(mid, 0)] == [1, 2, 3]
    assert [e["seq"] for e in reopened.events_after(mid, 2)] == [3]
    assert reopened.events_after(mid, 0)[0]["data"] == {"stage": "x", "seq": 1}


def test_restart_marks_running_turns_as_error(tmp_path):
    s = Store(tmp_path)
    sess, _, mid = _turn(s)
    s.close()
    restarted = Store(tmp_path)
    assert restarted.recover_after_restart() == 1
    assert restarted.get_message(mid, None)["status"] == "error"
    restarted.close()


def test_public_sessions_are_scoped_to_their_client(tmp_path):   # Review Focus 3
    s = Store(tmp_path)
    a = s.create_session("public", "client-a")
    s.create_session("public", "client-b")
    assert [x["id"] for x in s.list_sessions("client-a")[0]] == [a["id"]]
    assert s.get_session(a["id"], "client-b") is None
    assert s.delete_session(a["id"], "client-b") is False


def test_delete_removes_uploads_and_hides_the_session(tmp_path):
    s = Store(tmp_path)
    sess, _, _ = _turn(s)
    s.save_upload(sess["id"], "ab" * 32, b"png-bytes", "png", b"jpg", b"png")
    assert s.upload_path(sess["id"], "ab" * 32, "original").exists()
    assert s.delete_session(sess["id"], None) is True
    assert s.upload_path(sess["id"], "ab" * 32, "original") is None
    assert s.get_session(sess["id"], None) is None


def test_export_json_is_the_event_log_and_md_has_the_report(tmp_path):
    s = Store(tmp_path)
    sess, _, mid = _turn(s)
    s.append_event(mid, "message_stop", {"status": "done", "report": "Findings: clear."})
    s.finish_turn(mid, "done", report="Findings: clear.", display_report="Findings: clear.", total_ms=12.0)
    name, media, body = s.export(sess["id"], "json", None)
    assert media == "application/json" and json.loads(body)["messages"][1]["events"][0]["event"] == "message_stop"
    name, media, body = s.export(sess["id"], "md", None)
    assert media.startswith("text/markdown") and b"Findings: clear." in body


def test_sweep_deletes_only_old_sessions(tmp_path):
    s = Store(tmp_path)
    old = s.create_session("private")
    new = s.create_session("private")
    s._con.execute("UPDATE sessions SET created_at='2000-01-01T00:00:00+00:00' WHERE id=?", (old["id"],))
    s._con.commit()
    assert s.sweep(older_than_days=30) == 1
    assert s.get_session(new["id"], None) is not None
```

2. Run: FAIL. 3. Implement: one `sqlite3.connect(home / "chat.db", check_same_thread=False, isolation_level=None)` guarded by a `threading.Lock`; the §6.4 pragmas and DDL at start-up; `append_event` does `BEGIN IMMEDIATE` → `SELECT COALESCE(MAX(seq), 0) + 1` → `INSERT` → `COMMIT` and returns `dict(data, seq=seq)` (the stored `data_json` includes `seq`); timestamps are UTC ISO-8601; `delete_session` soft-deletes the row and removes `uploads/<sid>` with `shutil.rmtree(ignore_errors=True)`; `export("md")` renders per turn a heading (time, filename or `test row N`), the report, a 14-row label table if a `label` stage exists, neighbours only if the session's mode is private, and one provenance line. 4. PASS. 5–7. Commit `"P3-B: SQLite store (NFS-safe pragmas), uploads, export, sweep"`.

- [x] **P3-C** `app/redact.py`: the public field policy (R1), applied before an event is stored and sent; field-by-field and catch-all tests.

**Files:** create `app/redact.py`, `tests/test_app_redact.py`.
**Produces:** `redact_event(event: str, data: Dict[str, Any], mode: str) -> Optional[Dict[str, Any]]` (`None` drops the event); `PUBLIC_DROP: Dict[str, List[str]]`.

```python
PUBLIC_DROP = {   # dotted paths; "[]" walks a list; private mode returns data unchanged
    "message_start": ["options.reference", "options.test_row", "image.urls.original"],
    "stage_end:preprocess": ["detail.test_row", "detail.identical_to"],
    # U2 (user, 2026-10-01): public mode shows similarity scores only.
    "stage_end:retrieve": ["detail.image_neighbors[].image_url", "detail.image_neighbors[].study_id",
                           "detail.image_neighbors[].gallery_row", "detail.image_neighbors[].txt_row",
                           "detail.image_neighbors[].labels", "detail.report_matches[].report",
                           "detail.report_matches[].group", "detail.report_matches[].group_size",
                           "detail.report_matches[].txt_row", "detail.report_matches[].labels",
                           "detail.true_report_rank", "detail.gallery.build_id"],
    "stage_end:label": ["detail.neighbor_agreement"],
    "stage_end:score": "*",   # the whole event: there is no reference in public mode
}
```

The generated report's own `chexbert_14` labels stay in public mode: they describe model output, not a MIMIC record.

1. Failing tests:

```python
# tests/test_app_redact.py
"""CHAT_UI_PLAN.md P3-C: nothing MIMIC-derived leaves the cluster in public mode (R1)."""
import copy
import json

from app.redact import redact_event

PRIVATE_RETRIEVE = {"stage": "retrieve", "ms": 4.0, "detail": {
    "image_neighbors": [{"rank": 1, "similarity": 0.91, "gallery_row": 1843, "image_url": "/v1/gallery/images/1843",
                         "study_id": "50414267", "labels": {"Edema": 1}}],
    "report_matches": [{"rank": 1, "similarity": 0.41, "group": 88213, "group_size": 4,
                        "report": "Findings: SECRET MIMIC TEXT", "labels": {"Edema": 1}}],
    "true_report_rank": {"rank": 3, "of": 2663}, "gallery": {"build_id": "g1", "images": 191462}}}


def test_public_retrieve_keeps_similarity_scores_only():   # U2
    out = redact_event("stage_end", copy.deepcopy(PRIVATE_RETRIEVE), "public")
    n, m = out["detail"]["image_neighbors"][0], out["detail"]["report_matches"][0]
    assert set(n) == {"rank", "similarity"}
    assert set(m) == {"rank", "similarity"}
    assert "true_report_rank" not in out["detail"] and "build_id" not in out["detail"]["gallery"]


def test_public_label_stage_drops_neighbour_agreement_but_keeps_the_reports_own_labels():
    data = {"stage": "label", "ms": 3.0, "detail": {"chexbert_14": {"Edema": 1}, "positives": ["Edema"],
                                                    "neighbor_agreement": [{"rank": 1, "agree": 13, "of": 14}]}}
    out = redact_event("stage_end", copy.deepcopy(data), "public")
    assert "neighbor_agreement" not in out["detail"] and out["detail"]["chexbert_14"] == {"Edema": 1}


def test_no_private_string_survives_into_a_public_payload():
    out = json.dumps(redact_event("stage_end", copy.deepcopy(PRIVATE_RETRIEVE), "public"))
    for secret in ("SECRET MIMIC TEXT", "50414267", "/v1/gallery/images/1843"):
        assert secret not in out


def test_score_is_dropped_in_public_and_kept_in_private():
    data = {"stage": "score", "ms": 1.0, "detail": {"rouge_l": 0.2}}
    assert redact_event("stage_end", dict(data), "public") is None
    assert redact_event("stage_end", dict(data), "private") == data


def test_private_mode_is_a_no_op():
    assert redact_event("stage_end", copy.deepcopy(PRIVATE_RETRIEVE), "private") == PRIVATE_RETRIEVE


def test_generated_report_is_not_redacted():   # model output is not MIMIC data
    stop = {"status": "done", "report": "Findings: heart normal.", "display_report": "Findings: heart normal."}
    assert redact_event("message_stop", dict(stop), "public") == stop
```

2. Run: FAIL. 3. Implement a small dotted-path deleter (`_drop(obj, path)` that handles `a.b`, `a[].b`); key the policy by `event` or `event:stage`. 4. PASS. 5–7. Commit `"P3-C: public-mode redaction policy (R1)"`.

- [x] **P3-D** `app/pipeline.py`, `app/server.py`, `app/commands.py`: routes, auth, bind refusal, single-worker runner, SSE bridge, polling, cancel, overload, text commands.

**Files:** create `app/pipeline.py`, `app/server.py`, `app/commands.py`, `tests/test_app_api.py`, `tests/test_app_commands.py`; extend `tests/app_helpers.py`.
**Consumes:** `Engine`, `Store`, `redact_event`, `Options`, `error_body`, `new_id`.
**Produces:** `create_app(engine: str = "tiny", mode: str = "private", home: Optional[str] = None, host: str = "127.0.0.1", token: Optional[str] = None, queue_cap: int = 4, gallery_dir: Optional[str] = None, labeler_url: Optional[str] = None, models: Sequence[str] = ("hybrid_150m_m3_rrg",), allow_compile: bool = False, cors_origins: Sequence[str] = (), published_dirs: Optional[Dict[str, str]] = None, tiny_step_delay_s: float = 0.0, drift_note: str = "") -> FastAPI`; `parse_command(text: str) -> Optional[Dict[str, Any]]`; `NOT_A_QA_BOT: str`.

Routes in this task: `GET /` (index.html), `/static/*`, `GET /healthz`, `GET /v1/models`, `POST /v1/sessions`, `GET /v1/sessions`, `GET /v1/sessions/{id}`, `DELETE /v1/sessions/{id}`, `POST /v1/sessions/{id}/messages` (stream), `GET /v1/messages/{id}?after=`, `POST /v1/messages/{id}/cancel`, `GET /v1/sessions/{id}/export?format=`. P5 and P6 add the rest.

Behaviour that the tests pin:

- R6/D9: `create_app` raises `RuntimeError("...token...")` when `host` is not loopback and `token` is empty, or when `mode == "public"` and `token` is empty.
- D23: with a token set, every `/v1/*` route needs `Authorization: Bearer <token>` (401 otherwise); `/`, `/static/*` and `/healthz` stay open.
- D8: in public mode the `X-Client-Id` header (a random id the page keeps in `localStorage`) scopes sessions and messages; a missing header gets 400.
- Validation and overload are refused **before** the stream opens: 422 / 429 with `error_body`. After it opens, failures are `error` events.
- The pipeline runs on `ThreadPoolExecutor(max_workers=1)`. At most `queue_cap` turns are accepted and unfinished at once. Each event is redacted, stored (`append_event`), then pushed into the request's `asyncio.Queue` with `loop.call_soon_threadsafe`. A `RuntimeError` from a closed loop is swallowed: the event is already stored.
- D7: a client disconnect only stops forwarding. `POST /v1/messages/{id}/cancel` sets the turn's `threading.Event`; the engine raises `Cancelled` at the next step; the turn ends `aborted`.
- Text-only turns: no image and no `test_row` → reuse `store.last_image(session)`; none → 422 `"Attach an X-ray first."`. With text, `parse_command` overrides the drawer options; a non-command text with no image produces a `warning` event `not_a_command` carrying `NOT_A_QA_BOT`, and the turn ends `done` with no report. Text sent with an image is stored as a note (and applied if it is a command).
- `message_start.image.urls` point at `/v1/messages/<user_message_id>/image?variant=…` (served from P6-B).
- The pipeline stores every upload at `preprocess` with `store.save_upload` (original bytes, `thumbnail_jpeg`, the model-input PNG), once per session and hash. A text-only turn re-reads `original` through `store.upload_path`. (P6-B only adds the image endpoint and the UI.)
- Until P5-E, the last three stages end skipped with fixed reasons: `retrieve` → `gallery_unavailable` (no gallery), `label` → `label_off` when `options.label` is false, else `labeler_unavailable` (no labeller), `score` → `no_reference`. P5-E keeps these reasons when it adds the real stages.

Core of the bridge:

```python
def sse(event: str, data: Dict[str, Any]) -> str:
    return "event: {}\ndata: {}\n\n".format(event, json.dumps(data, separators=(",", ":"), ensure_ascii=False))


async def _frames(queue: "asyncio.Queue[Optional[str]]"):
    while True:
        try:
            frame = await asyncio.wait_for(queue.get(), timeout=15.0)
        except asyncio.TimeoutError:
            yield ": ping\n\n"
            continue
        if frame is None:          # the worker finished the turn
            return
        yield frame
    # A client that disconnects only stops this generator. The worker keeps running and storing
    # events; GET /v1/messages/{id}?after=<seq> picks them up (D7).
```

`app/commands.py`:

```python
"""Text-only turns (spec §4): a small command set; anything else gets one fixed answer."""
import re
from typing import Any, Dict, Optional

COMMAND_HELP = "Commands: beam N, greedy, tokens N, retrieve N, reference: <text>, repair on|off."
NOT_A_QA_BOT = ("I generate chest X-ray reports and can't answer questions. Attach an X-ray, "
                "or send a command. " + COMMAND_HELP)
_RULES = [
    (r"beam (\d+)", lambda m: {"decode": "beam", "beam_size": int(m.group(1))}),
    (r"greedy", lambda m: {"decode": "greedy"}),
    (r"tokens (\d+)", lambda m: {"max_new_tokens": int(m.group(1))}),
    (r"retrieve (\d+)", lambda m: {"k_images": int(m.group(1)), "k_reports": int(m.group(1))}),
    (r"reference:\s*(.+)", lambda m: {"reference": m.group(1)}),
    (r"repair (on|off)", lambda m: {"display_repair": m.group(1).lower() == "on"}),
]


def parse_command(text: str) -> Optional[Dict[str, Any]]:
    """Option overrides for a text-only turn, or None if the text is not a command."""
    t = " ".join((text or "").split())
    for pattern, build in _RULES:
        m = re.fullmatch(pattern, t, flags=re.IGNORECASE)
        if m:
            return build(m)
    return None
```

1. Failing tests (abridged list; every name below is a test to write in full):

```python
# tests/test_app_commands.py
from app.commands import parse_command


def test_commands_map_to_options():
    assert parse_command("beam 5") == {"decode": "beam", "beam_size": 5}
    assert parse_command("  GREEDY ") == {"decode": "greedy"}
    assert parse_command("tokens 150") == {"max_new_tokens": 150}
    assert parse_command("retrieve 6") == {"k_images": 6, "k_reports": 6}
    assert parse_command("reference: Findings: clear lungs.") == {"reference": "Findings: clear lungs."}
    assert parse_command("repair on") == {"display_repair": True}
    assert parse_command("what does this mean?") is None
```

```python
# tests/test_app_api.py
"""CHAT_UI_PLAN.md P3-D: the streaming API on the tiny engine."""
import json
import socket
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from app.server import create_app
from tests.app_helpers import iter_sse, png_bytes

STAGES = ["preprocess", "encode", "retrieve", "generate", "label", "score"]


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path))) as c:
        yield c


@pytest.fixture
def live(tmp_path):
    """A real uvicorn on an ephemeral port: TestClient buffers whole responses, so streaming and
    disconnect behaviour are tested over real sockets."""
    import uvicorn
    app = create_app(engine="tiny", home=str(tmp_path), tiny_step_delay_s=0.02, queue_cap=2)
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    yield "http://127.0.0.1:{}".format(port)
    server.should_exit = True
    t.join(5)


def _turn(c, sid, options=None, image=True, text=""):
    files = {"image": ("x.png", png_bytes(), "image/png")} if image else None
    data = {"text": text, "options": json.dumps(options or {"max_new_tokens": 16})}
    with c.stream("POST", "/v1/sessions/{}/messages".format(sid), files=files, data=data) as r:
        assert r.status_code == 200, r.read()
        return list(iter_sse(r.iter_text()))


def test_turn_streams_contiguous_events_in_contract_order(client):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    frames = _turn(client, sid)
    assert [f["data"]["seq"] for f in frames] == list(range(1, len(frames) + 1))
    assert frames[0]["event"] == "message_start" and frames[-1]["event"] == "message_stop"
    assert [f["data"]["stage"] for f in frames if f["event"] == "stage_end"] == STAGES
    assert frames[-1]["data"]["status"] == "done"
    assert frames[-1]["data"]["disclaimer"] == "Research prototype; not for clinical use."


def test_stored_events_replay_exactly_what_was_sent(client):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    frames = _turn(client, sid)
    mid = frames[0]["data"]["message_id"]
    stored = client.get("/v1/messages/{}".format(mid)).json()["events"]
    assert [(e["event"], e["data"]) for e in stored] == [(f["event"], f["data"]) for f in frames]
    assert [e["seq"] for e in client.get("/v1/messages/{}?after=5".format(mid)).json()["events"]][0] == 6


def test_dropped_stream_does_not_cancel_the_turn(live):   # Review Focus 4
    sid = httpx.post(live + "/v1/sessions", json={}).json()["id"]
    with httpx.stream("POST", live + "/v1/sessions/{}/messages".format(sid),
                      files={"image": ("x.png", png_bytes(), "image/png")},
                      data={"text": "", "options": json.dumps({"max_new_tokens": 60})}, timeout=30) as r:
        frames = []
        for frame in iter_sse(r.iter_text()):
            frames.append(frame)
            if frame["event"] == "content_block_delta":
                break                                       # the tunnel drops here
    mid = frames[0]["data"]["message_id"]
    for _ in range(400):
        msg = httpx.get(live + "/v1/messages/{}".format(mid)).json()
        if msg["status"] != "running":
            break
        time.sleep(0.05)
    assert msg["status"] == "done"
    assert [e["seq"] for e in msg["events"]] == list(range(1, len(msg["events"]) + 1))


def test_cancel_endpoint_aborts_the_turn(live):
    sid = httpx.post(live + "/v1/sessions", json={}).json()["id"]
    frames, cancel_status = [], []
    with httpx.stream("POST", live + "/v1/sessions/{}/messages".format(sid),
                      files={"image": ("x.png", png_bytes(), "image/png")},
                      data={"text": "", "options": json.dumps({"max_new_tokens": 150})}, timeout=30) as r:
        for frame in iter_sse(r.iter_text()):
            frames.append(frame)
            if frame["event"] == "content_block_delta" and not cancel_status:
                mid = frames[0]["data"]["message_id"]
                cancel_status.append(httpx.post(live + "/v1/messages/{}/cancel".format(mid)).status_code)
    assert cancel_status == [200]
    assert frames[-1]["event"] == "message_stop" and frames[-1]["data"]["status"] == "aborted"
    assert len([f for f in frames if f["event"] == "content_block_delta"]) < 150
    mid = frames[0]["data"]["message_id"]
    assert httpx.get(live + "/v1/messages/{}".format(mid)).json()["status"] == "aborted"


def test_overload_is_refused_before_the_stream_opens(live):   # the live fixture has queue_cap=2
    sid = httpx.post(live + "/v1/sessions", json={}).json()["id"]
    url = live + "/v1/sessions/{}/messages".format(sid)
    body = {"text": "", "options": json.dumps({"max_new_tokens": 150})}
    opened = []

    def hold():
        with httpx.stream("POST", url, files={"image": ("x.png", png_bytes(), "image/png")},
                          data=body, timeout=60) as r:
            opened.append(r.status_code)
            for _ in r.iter_text():
                pass

    threads = [threading.Thread(target=hold) for _ in range(2)]
    for t in threads:
        t.start()
    deadline = time.time() + 10
    while len(opened) < 2 and time.time() < deadline:
        time.sleep(0.02)
    r = httpx.post(url, files={"image": ("x.png", png_bytes(), "image/png")}, data=body, timeout=30)
    assert r.status_code == 429 and r.json()["error"]["type"] == "overloaded_error"
    for t in threads:
        t.join(60)
    assert opened == [200, 200]


def test_refuses_non_loopback_bind_without_token(tmp_path):   # Review Focus 2, R6
    with pytest.raises(RuntimeError, match="token"):
        create_app(engine="tiny", home=str(tmp_path), host="0.0.0.0", token=None)
    with pytest.raises(RuntimeError, match="token"):
        create_app(engine="tiny", home=str(tmp_path), mode="public", token=None)


def test_token_guards_every_v1_route_but_not_health(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path), token="t0k")) as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/v1/sessions").status_code == 401
        assert c.post("/v1/sessions", json={}).status_code == 401
        assert c.post("/v1/sessions", json={}, headers={"Authorization": "Bearer t0k"}).status_code == 200


def test_public_client_cannot_read_another_clients_session(tmp_path):   # Review Focus 3
    with TestClient(create_app(engine="tiny", home=str(tmp_path), mode="public", token="t")) as c:
        auth = {"Authorization": "Bearer t"}
        sid = c.post("/v1/sessions", json={}, headers=dict(auth, **{"X-Client-Id": "a"})).json()["id"]
        assert c.get("/v1/sessions/{}".format(sid), headers=dict(auth, **{"X-Client-Id": "b"})).status_code == 404
        assert c.get("/v1/sessions", headers=dict(auth, **{"X-Client-Id": "b"})).json()["sessions"] == []


def test_text_only_turn_reuses_the_previous_image_and_applies_the_command(client):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    _turn(client, sid)
    frames = _turn(client, sid, image=False, text="beam 2")
    start = frames[0]["data"]
    assert start["image"]["source"] == "previous" and start["options"]["beam_size"] == 2


def test_question_without_image_gets_the_fixed_answer(client):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    _turn(client, sid)
    frames = _turn(client, sid, image=False, text="is this pneumonia?")
    assert any(f["event"] == "warning" and f["data"]["code"] == "not_a_command" for f in frames)


def test_first_turn_without_image_is_a_422(client):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    r = client.post("/v1/sessions/{}/messages".format(sid), data={"text": "beam 2", "options": "{}"})
    assert r.status_code == 422 and "Attach an X-ray" in r.json()["error"]["message"]


def test_bad_options_are_a_422_before_streaming(client):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    r = client.post("/v1/sessions/{}/messages".format(sid), files={"image": ("x.png", png_bytes(), "image/png")},
                    data={"text": "", "options": json.dumps({"retrieval_k": 3})})
    assert r.status_code == 422 and r.json()["error"]["type"] == "validation_error"


def test_export_and_delete(client):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    _turn(client, sid)
    js = client.get("/v1/sessions/{}/export?format=json".format(sid))
    assert js.headers["content-type"].startswith("application/json")
    assert js.json()["messages"][1]["events"][-1]["event"] == "message_stop"   # [0] user, [1] assistant
    md = client.get("/v1/sessions/{}/export?format=md".format(sid))
    assert md.headers["content-type"].startswith("text/markdown") and "## Turn 1" in md.text
    assert client.delete("/v1/sessions/{}".format(sid)).status_code == 204
    assert client.get("/v1/sessions/{}".format(sid)).status_code == 404
```

The Markdown export starts with `# Session <id>` and has one `## Turn <n> — <UTC time>` heading per turn.

2. Run: FAIL. 3. Implement. 4. `venv/bin/python -m pytest tests/test_app_api.py tests/test_app_commands.py -v` → PASS. 5–7. Commit `"P3-D: streaming API, runner, cancel, auth, scoping, commands"`.

- [x] **P3-E** OpenAPI summaries and examples for every route; `app/README.md`'s curl walkthrough reproduced by a test.

**Files:** modify `app/server.py`; create `tests/test_app_openapi.py` (with its own `client` fixture, identical to the one in `tests/test_app_api.py`).

1. Failing tests:

```python
def test_openapi_documents_every_v1_route(client):
    spec = client.get("/openapi.json").json()
    routes = {(m.upper(), p) for p, ops in spec["paths"].items() for m in ops}
    for want in [("POST", "/v1/sessions"), ("GET", "/v1/sessions"), ("POST", "/v1/sessions/{session_id}/messages"),
                 ("GET", "/v1/messages/{message_id}"), ("POST", "/v1/messages/{message_id}/cancel")]:
        assert want in routes
    for path, ops in spec["paths"].items():
        for op in ops.values():
            assert op.get("summary"), path


def test_curl_walkthrough_options_validate():
    from app.schemas import Options
    Options(**{"decode": "beam", "beam_size": 3, "max_new_tokens": 100, "cached_decode": True, "compile": False,
               "k_images": 4, "k_reports": 3, "label": True, "reference": None, "display_repair": False})
```

2–4. Fail, add `summary=`/`description=` and an `openapi_examples` entry on the options form field, pass. 5–7. Commit `"P3-E: OpenAPI reference"`.

### P4 — Frontend shell (laptop, tiny engine)

Gate: in a browser on `--engine tiny`, a full turn streams; reload replays an identical card; JSON and Markdown exports open; `node --test` green; layout usable at 375 px.

- [x] **P4-A** `index.html` and `styles.css`: layout grid, banner, light/dark tokens, 375 px; served at `/` and `/static/*`.

**Files:** create `app/static/index.html`, `app/static/styles.css`; modify `app/server.py`; create `tests/test_app_static.py`.

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CXR Report Chat</title>
  <link rel="stylesheet" href="/static/styles.css">
  <script type="module" src="/static/app.js"></script>
</head>
<body>
  <header class="banner" role="note">Research prototype — not for clinical use.
    <span id="mode-badge"></span> <span id="health" aria-live="polite"></span></header>
  <div class="layout">
    <nav id="sidebar" aria-label="Sessions">
      <button id="new-session" type="button">New chat</button>
      <ol id="session-list"></ol>
    </nav>
    <main id="conversation" aria-live="polite"></main>
    <aside id="drawer" hidden aria-label="Settings"></aside>
  </div>
  <form id="composer" aria-label="New turn">
    <div id="image-well" tabindex="0" role="button" aria-label="Attach an X-ray: click, drop or paste">
      Drop, paste or click to attach an X-ray (PNG, JPEG, WEBP)</div>
    <div id="preview" hidden></div>
    <textarea id="prompt" rows="2" placeholder="Optional note, or a command: beam 5, greedy, tokens 150, reference: …"></textarea>
    <div id="chips" aria-label="Current settings"></div>
    <button id="settings" type="button" aria-controls="drawer">Settings</button>
    <button id="send" type="submit">Send</button>
    <button id="stop" type="button" hidden>Stop</button>
    <input id="file" type="file" accept="image/png,image/jpeg,image/webp" hidden>
  </form>
  <div id="viewer" hidden role="dialog" aria-modal="true" aria-label="Image viewer"></div>
</body>
</html>
```

`styles.css`: colour tokens on `:root` (`--bg`, `--fg`, `--muted`, `--accent`, `--card`, `--border`, `--ok`, `--warn`, `--bad`), redefined in `@media (prefers-color-scheme: dark)`; `body` sets an explicit background; bars and icons are CSS or Unicode, never SVG (the no-external-requests test forbids any `http://`/`https://` string in `app/static/`, including SVG namespaces); grid `sidebar 260px | conversation 1fr | drawer 320px`, collapsing below 800 px (sidebar becomes a toggle) and to one column at 375 px with a 16 px gutter and no horizontal scroll; the banner is sticky.

Tests:

```python
def test_index_is_served_and_has_the_banner(client):
    r = client.get("/")
    assert r.status_code == 200 and "Research prototype — not for clinical use." in r.text


def test_page_makes_no_external_requests():
    import re
    from pathlib import Path
    static = Path("app/static")
    for f in static.glob("*"):
        text = f.read_text()
        assert not re.search(r"https?://", text), f
        assert "EventSource" not in text, f
```

Commit `"P4-A: page shell + styles"`.

*As built (P4-A, commits d98f4cb, 8fa8ecc; the code is authoritative where it differs from the HTML above):*
- **Banner.** It is `<header class="banner">` with no `role`. The disclaimer sits in `<p class="disclaimer">`, and `#sidebar-toggle` (☰) comes first.
- **Accessibility markup.**
  - `#chips` has `role="group"`.
  - The textarea has `aria-label="Note or command"`.
  - `#settings` has `aria-expanded`.
  - `#image-well` is named by its visible text.
  - A `data:` favicon is set.
- **Layout.**
  - The narrow layout applies at ≤ 800 px. There the drawer is confined to the conversation row and the sidebar is off-canvas (`body.sidebar-open`).
  - Chips sit on their own row, and the action row wraps.
  - At ≤ 480 px height the document scrolls.
- **Caching.** `/` and `/static/*` send `Cache-Control: no-cache`.
- **Geometry check.** `scripts/chat_ui_layout_check.py` is a CDP geometry check, run locally and not by `validate.sh`. It had 88 cases; P4-E brings it to 94.

- [x] **P4-B** `api.js` (SSE parser, stream, poll, cancel, image loader) and `state.js` (reducers, replay); `node --test` with a recorded fixture; `validate.sh` gains the node gate.

**Files:** create `app/static/api.js`, `app/static/state.js`, `tests/frontend/parsers.test.mjs`, `tests/frontend/fixtures/turn_tiny.json`, `tests/test_app_frontend_fixture.py`; modify `scripts/validate.sh`.
**Produces (JS):** `parseSSE(buffer) -> {events, rest}`, `streamTurn({base, sessionId, form, token, clientId, signal})` (async generator), `pollMessage({base, messageId, after, token, clientId, signal})` (async generator, 500 ms), `cancelMessage(...)`, `loadImage(url, auth) -> Promise<objectURL>` (fetch with headers, cached per URL; D23), `authHeaders(token, clientId)`; `initialView(messageId)`, `applyEvent(view, {event, data})`, `replay(events)`.

```javascript
// app/static/api.js (core)
export function parseSSE(buffer) {
  // Pure: feed text, get the complete frames and the unconsumed tail. ": ping" comments are skipped.
  buffer = buffer.replace(/\r\n/g, '\n');
  const events = [];
  let i;
  while ((i = buffer.indexOf('\n\n')) >= 0) {
    const frame = buffer.slice(0, i);
    buffer = buffer.slice(i + 2);
    let event = 'message';
    const data = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith(':')) continue;
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    }
    if (data.length) events.push({ event, data: JSON.parse(data.join('\n')) });
  }
  return { events, rest: buffer };
}

export function authHeaders(token, clientId) {
  const h = {};
  if (token) h.Authorization = `Bearer ${token}`;
  if (clientId) h['X-Client-Id'] = clientId;
  return h;
}

export async function* streamTurn({ base = '', sessionId, form, token, clientId, signal }) {
  const res = await fetch(`${base}/v1/sessions/${sessionId}/messages`,
                          { method: 'POST', body: form, signal, headers: authHeaders(token, clientId) });
  if (!res.ok) throw Object.assign(new Error('turn refused'), { status: res.status, body: await res.json().catch(() => null) });
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    const { events, rest } = parseSSE(buf + dec.decode(value, { stream: true }));
    buf = rest;
    yield* events;
  }
}
```

```javascript
// app/static/state.js — pure reducers: the stored event log rebuilds exactly the live view.
export function initialView(messageId) {
  return { id: messageId, status: 'running', lastSeq: 0, mode: null, provenance: null, options: null,
           image: null, stages: {}, report: '', displayReport: '', provisional: false, truncated: false,
           neighbors: [], matches: [], trueRank: null, labels: null, agreement: null, score: null,
           notices: [], error: null, totalMs: null };
}

export function applyEvent(view, { event, data }) {
  if (data.seq <= view.lastSeq) return view;            // polling can resend what the stream delivered
  const v = { ...view, lastSeq: data.seq };
  switch (event) {
    case 'message_start':
      return { ...v, id: data.message_id, mode: data.mode, provenance: data.model, options: data.options, image: data.image };
    case 'stage_start':
      return { ...v, stages: { ...v.stages, [data.stage]: { state: 'running' } } };
    case 'stage_end': {
      const st = data.skipped ? { state: 'skipped', skipped: data.skipped }
                              : { state: 'done', ms: data.ms, detail: data.detail };
      const next = { ...v, stages: { ...v.stages, [data.stage]: st } };
      const d = data.detail || {};
      if (data.stage === 'retrieve') return { ...next, neighbors: d.image_neighbors || [], matches: d.report_matches || [], trueRank: d.true_report_rank || null };
      if (data.stage === 'label') return { ...next, labels: d.chexbert_14 || null, agreement: d.neighbor_agreement || null };
      if (data.stage === 'score') return { ...next, score: d };
      return next;
    }
    case 'content_block_delta':
      return { ...v, report: data.delta.text, provisional: true };
    case 'content_block_stop':
      return { ...v, provisional: false };
    case 'warning':
      return { ...v, notices: [...v.notices, data] };
    case 'error':
      return { ...v, error: data.error };
    case 'message_stop':
      return { ...v, status: data.status, report: data.report ?? v.report, displayReport: data.display_report ?? v.report,
               truncated: !!data.truncated_mid_sentence, provisional: false, totalMs: data.total_ms };
    default:
      return v;
  }
}

export const replay = (events) => events.reduce(applyEvent, initialView(events.length ? events[0].data.message_id : null));
```

Stage keys are inserted in arrival order, so `Object.keys(view.stages)` follows the contract order.

```javascript
// tests/frontend/parsers.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseSSE } from '../../app/static/api.js';
import { applyEvent, initialView, replay } from '../../app/static/state.js';

test('a frame split across chunks reassembles', () => {
  const whole = 'event: stage_start\ndata: {"seq":1,"stage":"encode"}\n\n';
  let { events, rest } = parseSSE(whole.slice(0, 17));
  assert.equal(events.length, 0);
  ({ events, rest } = parseSSE(rest + whole.slice(17)));
  assert.deepEqual(events, [{ event: 'stage_start', data: { seq: 1, stage: 'encode' } }]);
  assert.equal(rest, '');
});

test('CRLF, a CR/LF split across chunks, multi-line data and ping comments', () => {
  let { events, rest } = parseSSE(': ping\r\n\r\nevent: x\r\ndata: {"a":\r');
  ({ events, rest } = parseSSE(rest + '\ndata: 1, "seq": 1}\r\n\r\n'));
  assert.deepEqual(events, [{ event: 'x', data: { a: 1, seq: 1 } }]);
});

test('replaying the stored log rebuilds the live view exactly', () => {
  const log = JSON.parse(readFileSync(new URL('./fixtures/turn_tiny.json', import.meta.url)));
  let live = initialView(log[0].data.message_id);
  for (const e of log) live = applyEvent(live, e);
  assert.deepEqual(replay(log), live);
  assert.equal(live.status, 'done');
  assert.deepEqual(Object.keys(live.stages), ['preprocess', 'encode', 'retrieve', 'generate', 'label', 'score']);
});

test('an event repeated by polling is ignored', () => {
  const log = JSON.parse(readFileSync(new URL('./fixtures/turn_tiny.json', import.meta.url)));
  const once = replay(log);
  assert.deepEqual(replay([...log, ...log.slice(3, 9)]), once);
});
```

`tests/test_app_frontend_fixture.py` regenerates `turn_tiny.json` from a fresh tiny turn when `UPDATE_FIXTURES=1`; otherwise it asserts that the fixture's event-name sequence equals a fresh turn's (so the fixture cannot silently drift from the server).

`scripts/validate.sh`, new gate after Gate 2:

```bash
# ── Gate 2b: frontend parsers (node --test) — CHAT_UI_PLAN.md P4-B ──────────
echo ""
echo "── Gate 2b: node --test tests/frontend ──"
if command -v node >/dev/null 2>&1 && compgen -G "${REPO_ROOT}/tests/frontend/*.test.mjs" >/dev/null; then
  if node --test "${REPO_ROOT}"/tests/frontend/*.test.mjs 2>&1; then gate_pass "node: frontend tests passed"
  else gate_fail "node: frontend tests failed"; fi
else
  echo -e "${WARN_TAG} node not found or no tests/frontend/*.test.mjs — skipped"
fi
```

Run `node --test tests/frontend/*.test.mjs` (FAIL first, then PASS). Commit `"P4-B: SSE parser, reducers, node tests in validate.sh"`.

*As built (P4-B, commits 1a1beca, 50ed940; the code is authoritative where it differs from the listings above):*
- **Selectors.** `state.js` adds the pure selectors `stageState(view, name)` (keyed on `view.status`), `labelsPending(view)` and `STAGES`.
  - When the turn is aborted, stages left running or pending show as skipped/`stopped`.
  - When the turn errored, a running stage shows as `error` and a pending one as skipped/`not_run`.
  - A skipped stage leaves its derived fields at `null`/`[]`.
- **Polling.** `pollMessage` retries a network `TypeError` and 429/502/503/504 with backoff from 500 ms to 5 s, and gives up after 20 consecutive failures. Any other non-2xx is terminal.
- **Streaming.** `streamTurn` takes `onMessageId` and cancels its reader in a `finally`.
- **Parsing.** `parseSSE` skips malformed frames.
- **Images.** `loadImage` accepts same-origin relative URLs only.
- **Node gate.** Gate 2b runs with `--test-timeout=30000`, behind a version guard (node ≥ 22.7, or 20.19+). When skipped, it still adds a SUMMARY line. `tests/test_validate_node_gate.py` tests the gate itself.

- [x] **P4-C** `render.js`: stage timeline, report card, label chips, provenance footer; every update re-renders from the view.

**Files:** create `app/static/render.js`.
**Produces:** `renderUserTurn(msg, ctx)`, `renderAssistantCard(view, ctx) -> HTMLElement` composed of `renderTimeline`, `renderReport`, `renderLabels`, `renderProvenance` (P6 adds `renderImages`, `renderNeighbors`, `renderMatches`). `ctx` carries `{loadImage, openViewer, copy}`.

- User turn: the thumbnail (P6-B serves it; until then the local preview URL), the text, and the resolved options as small chips (`beam 3 · 100 tok · cached · k 4/3`).
- Timeline: `<ol class="timeline">` with one `<li data-stage data-state="pending|running|done|skipped|error">` per stage, text `encode · 612 ms`, `aria-label="encode, done, 612 milliseconds"`; a click toggles a `<table>` of the stage `detail` (keys and values; arrays of objects as nested rows; long text clipped with a "show more").

```javascript
// app/static/render.js (pattern every builder follows)
export const STAGES = ['preprocess', 'encode', 'retrieve', 'generate', 'label', 'score'];

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) node.setAttribute(k, v === true ? '' : v);
  }
  node.append(...children.filter((c) => c != null));
  return node;
}

export function renderTimeline(view) {
  return el('ol', { class: 'timeline', 'aria-label': 'Pipeline stages' }, ...STAGES.map((s) => {
    const st = view.stages[s] || { state: 'pending' };
    const label = st.state === 'done' ? `${s} · ${Math.round(st.ms)} ms` : st.state === 'skipped' ? `${s} · skipped` : s;
    const spoken = st.state === 'done' ? `${s}, done, ${Math.round(st.ms)} milliseconds`
                 : st.state === 'skipped' ? `${s}, skipped: ${st.skipped}` : `${s}, ${st.state}`;
    const li = el('li', { 'data-stage': s, 'data-state': st.state, 'aria-label': spoken, tabindex: 0 }, label);
    if (st.detail) li.addEventListener('click', () => li.classList.toggle('open'));
    if (st.detail) li.append(detailTable(st.detail));
    return li;
  }));
}
```

`detailTable(detail)` renders keys and values (arrays of objects as nested rows; strings over 200 characters clipped with "show more").
- Report: split on the literal `Findings:` and `Impression:` headers into two sections; a `provisional` class while `view.provisional`; buttons "Copy", "Show raw" (toggles raw report vs the display copy), and a note "Report stopped at the token budget mid-sentence" when `view.truncated`.
- Labels: 14 chips in `CHEXBERT_14` order (fetched once from `/v1/models`); positives filled, negatives outlined; with `view.score`, each chip also shows agree/disagree with the reference, and the four score numbers sit beside them. Before the `label` stage ends, the chips show a "labelling…" placeholder; if it was skipped, "labels unavailable (<reason>)".
- Provenance footer: `model · ckpt <first 8 of sha256> · scan <…>/tfla <…> · prefix_k <…> · beam <n> · <tokens> tok · <total> ms · <device>` plus the `drift_note`; a link opens `/v1/models` in the drawer.
- Rendering is throttled with `requestAnimationFrame`; the card is replaced as a whole (`old.replaceWith(renderAssistantCard(view, ctx))`).

Verification is the P4-E browser checklist. Commit `"P4-C: card rendering"`.

*As built (P4-C, commits 45aca96 and 0157ed1; the code is authoritative where it differs from the text above):*
- **Builders:**
  - `renderUserTurn(msg, ctx)` takes `msg = {text, image: {url, filename}, options}`, where `url` is a `blob:`/`data:` URL or a `/v1` path that is fetched through `ctx.loadImage`.
  - `renderAssistantCard(view, ctx)`, `statusText(view)`, `scheduleRender`.
- **`ctx`:** `{loadImage, openViewer, copy, showModels, labelNames, ui, turn}`.
  - `ctx.ui` is a Map that keeps each card's open details and raw-report toggle across whole-card replaces.
  - `ctx.turn` names the card and its controls.
- **Accessibility:**
  - A stage that has a detail is a `<button aria-expanded aria-controls>` inside its `li`. Every other stage is plain text.
  - Chip state is spoken through `.visually-hidden` text.
  - Controls carry `data-action` focus hooks, and `focusKey`/`restoreFocus` keep focus across a re-render.
- **Security:**
  - Same-origin paths are checked by one predicate exported from `api.js`.
  - A recursive test rejects HTML-sink APIs anywhere in `app/static/`.
- **Stage settling:** a `done` turn with at least one recorded stage settles its pending stages as `skipped/not_run`, so a public turn has no pending score.
- **Labels:** with labels off, "labels off" shows immediately.

- [x] **P4-D** `app.js`: routing, sessions sidebar, minimal file picker, Stop (cancel then abort), settings drawer, exports, health strip, polling watchdog.

**Files:** create `app/static/app.js`.

- Routing: `#/s/<session_id>`; no hash → newest session or an empty "New chat".
- Sidebar: `GET /v1/sessions`, newest first, title (first user text or upload filename), date, turn count; Delete asks once.
- Composer (minimal here; P6-A replaces it): the file input and Send; Enter sends, Shift+Enter adds a newline; Send is disabled while a turn runs.
- Stop: `POST /v1/messages/{id}/cancel`, then `controller.abort()`.
- Watchdog: when no bytes arrive for 3 s and the view is still `running`, switch to `pollMessage` from `view.lastSeq` until the status leaves `running`.
- Drawer: model (from `/v1/models`; `cached_decode` disabled when the model has no cache), decode, beam size, token budget, cached decode, compile (hidden unless the server allows it), `k_images`, `k_reports`, label on/off, display repair, mode (read-only), API base URL and token. Stored in `localStorage` under `cxrchat.settings` inside `try/catch`; defaults apply when storage is unavailable. `X-Client-Id`: a random id under `cxrchat.client`.
- Health strip: `GET /healthz` every 10 s; "server restarting…" while it fails.
- Exports: buttons fetch `/v1/sessions/{id}/export?format=json|md` with auth headers and save via a Blob link.

Commit `"P4-D: app wiring, sessions, drawer, stop, watchdog"`.

*As built (P4-D, commits 64a4728, fed1fce and 38f4b76; the code is authoritative where it differs from the text above):*
- **Structure.** `app.js` exports pure helpers and `createApp(env)`, and it starts only when `#composer` exists. Tests: `tests/frontend/app.test.mjs`, 118 tests.
- **Replay.** `GET /v1/sessions/{id}` carries no events, so replay fetches `/v1/messages/{id}` for each assistant message. Each card fails on its own and gets its own Retry.
- **Watchdog.**
  - It counts events, not bytes; the server pings only after 15 s.
  - It arms at accept, once `X-Message-Id` has arrived, so a slow upload waits instead of aborting.
  - Stop does a cancel and then an abort. It is disabled until the turn is accepted.
- **Drawer.**
  - No API-base field, because the app is same-origin.
  - The token is stored locally, with a hint saying so.
  - `cached_decode` is disabled for cache-less models.
  - `compile` is shown only when the server allows it.
- **Accessibility.**
  - `aria-busy` is set on `#conversation` while a turn streams and while a session opens.
  - A `.visually-hidden` status region carries `statusText`.
  - Esc returns focus to whatever opened the panel.
  - After Delete, focus moves to the next session.
- **Health.** `/healthz` has a 5 s timeout.

- [x] **P4-E** Browser checklist on the tiny engine, with screenshots in the evidence.

Run `venv/bin/python -m app.server --engine tiny --home /tmp/cxrchat_dev` (CLI from P7-B; until then `venv/bin/uvicorn --factory app.server:create_app`), open `http://127.0.0.1:8000`, and check: a turn streams with all six stages; the report fills step by step, then settles; reload shows the identical card; a second session switches cleanly; JSON and Markdown exports download and open; Stop aborts within a step; keyboard-only use works (Tab order, Enter, Esc); 375 px wide works with no horizontal scroll; screen-reader labels on timeline steps and chips. Record screenshots and the checklist in the evidence. Commit `"P4-E: browser checklist (tiny engine)"`.

*As built (P4-E, commits fca9f6f, cdde87a, c542ff2 and cf7f12a; the code is authoritative where it differs from the text above):*
- **Script.** `scripts/chat_ui_browser_check.py` starts `create_app(engine="tiny")` in-process on a free loopback port (there is no CLI until P7-B), or takes `--url`.
  - It prints one `CHECK <name> PASS|FAIL` line per item, then `RESULT {"checks": 9, "failures": []}`.
  - It exits 1 when a check fails, and 2 on a start failure or overrun.
  - It is run locally, not by `validate.sh`.
- **The nine checks:**
  - **stream:** six stages; the report grows over ≥ 3 snapshots; the card ends `done`.
  - **reload:** the card is identical after reload.
  - **switch:** switching sessions shows each session's own card, with no leaked stream.
  - **exports:** the JSON export parses; the Markdown export has `## Turn 1`.
  - **stop:** the turn ends `aborted` with fewer than 150 deltas.
  - **keyboard:** Tab order (including the chat links), Enter, and Esc, which returns focus to Settings.
  - **375 px:** no horizontal overflow.
  - **AX tree:** stage names start with their visible text; chip state is exposed; the status region reads "Report ready"; no name starts with a CSS glyph.
  - **error:** a real 422 from the note command `tokens 500` shows the notice, and the composer keeps its contents.
- **Check 8's chips** come from one synthetic labelled turn, seeded through `Store`'s public methods in a second tiny app, because the tiny pipeline skips `label` and `score` until P5-E.
  - The turn is marked SYNTHETIC in its text, its card and its file name.
  - It is replaced by a live turn at P5-E.
- **Harness.** `scripts/chat_ui_cdp.py` holds the shared CDP client, the Chrome launcher and the app runner.
  - Stop is graceful: CDP `Browser.close`, then SIGTERM to the group, then SIGKILL as the fallback, with one 3 s close deadline. No signal goes to a reaped process.
  - Both checks share one SIGTERM handler.
  - The layout check now has 94 cases. They assert that `#stop` and the open `#drawer` are rendered, and pin the 120 px report floor under a tall composer.
- **Evidence** (`docs/chat_ui/evidence/p4e/`): 8 PNGs plus `checklist.json`. The PNGs cover streaming, settled (light and dark), the drawer, 375×812, the stopped card, the error notice and the labelled chips. They come from the tiny engine and a synthetic image, so they hold no MIMIC data.
- **Page fixes.** The checks found no new UI defect. Review fixes:
  - `accept()` can no longer lock the page.
  - The composer is spent before `accept()`'s DOM work.
  - The route change clears the notice.
  - Retry buttons are named per turn.
  - `pollTurn` ignores a poll failure once the turn has settled.
  - `aria-current` is dropped from a session that failed to open.
  - The `aria-busy` clear waits one frame.
  - The CSS stage glyphs carry empty alt text.
- **Tests.** Node: 265. Harness: Chrome-free pytest in `tests/test_chat_ui_{browser,layout}_check.py`.

- [x] **P4-F** Clean report display and settings that visibly apply. Added 2026-10-08 after the user's own test on the tiny dev server.

The user reported two problems.

**Every report runs to the token budget and repeats itself.**
- The cause is the published protocol: `beam_search_decode` has no stop condition, and the model never learned an end-of-report token (V5-D). The real model behaves the same way: at 200 tokens, 17.9% of its sentences are repeats.
- The display repair used `dedup="none"`, so turning repair on never removed a repeat.
- The user's decision was "Clean display": decoding stays exactly the published protocol, and only the display changes.

**Changed settings seemed not to apply.**
- Settings did reach the server. What was missing was feedback.
- The composer chips changed only when a field was committed.
- Nothing said when a change takes effect.
- Nothing said that Send with no new image re-runs the last X-ray.
- k_images, k_reports and the CheXbert labels toggle did nothing, because those stages are skipped until P5-E, and nothing said so.

Commit `"P4-F: clean report display (dedup + stream view)"` and `"P4-F: settings feedback, re-run hint, server features"`.

*As built (P4-F, commits 86e62f8 and cbd5ac2; the code is authoritative):*
- **Display copy.** `display_report = repair_report(report, dedup="all", truncate=True)`. `report` and `truncated_mid_sentence` are unchanged (R2).
- **Stream view.** With `display_repair` on, each `content_block_delta.text` is `stream_view(raw)` (`app/engine.py`):
  - sentences already shown are dropped;
  - the start of a repeat is held back;
  - a new fragment is kept;
  - it is built on `split_sentences` from `scripts/repair_generations.py`.
- **User's case** (tiny engine, 200 tokens): the card falls from 173 words to 23, and repeated sentences from 13 to 0. No streamed snapshot repeats.
- **UI defaults and chips.**
  - Display repair is on by default in the UI only. `Options.display_repair` stays `False` for API clients.
  - "Show raw" shows the protocol text.
  - The chip reads `raw text` when repair is off.
  - The composer chips follow a number field as it is typed: whole in-bounds values on `input`; `change` still clamps.
- **Hints.**
  - The drawer says "Changes apply from your next Send.", and, while a turn runs, "The running turn keeps the settings it started with."
  - With no new image in a chat that has one, the composer says "No new image: Send re-runs <file> with these settings."
- **Server features.**
  - `GET /v1/models` gains `features: {retrieval, labels}`, which are true only when the pipeline has a gallery or a labeller (both false until P5-E).
  - The drawer disables k_images, k_reports and CheXbert labels when the feature is off, each with a one-line reason.
  - The composer drops the `k` chip when retrieval is off.
- **Browser check.** A 10th check, `settings`, uses real typing. It checks:
  - the chips update before the field loses focus;
  - the turn's chips and the card's token count show the new value;
  - a re-run with no image uses a new budget;
  - the hint appeared.
- **Test counts.** Layout: 94 cases. Node: 275.
- **Known limits, carried:**
  - On a stopped or streaming card with repair on, "Show raw" shows the stream view.
  - A header glued to a sentence ("Findings: X.") does not dedup a later "X.".
  - A browser that stored `display_repair: false` keeps it.

- [x] **P4-G** Stop condition: stop decoding once the report starts repeating; a Settings drawer with Save. Added 2026-10-09 after the user's second test.

**Why.** After P4-F the card was clean, but every turn still decoded the whole budget: 200 tokens took 16 s for a 23-word report, and the card still said it stopped at the budget. The published decoder has no stop condition, and the model never learned an end-of-report token (V5-D). So the app adds a stop condition at decode time.

**The stop condition.**
- **The check.** `Options.stop_on_repeat`. After every step the engine looks at the best beam's text. When its last *completed* sentence repeats an earlier one (the same key as `stream_view`), decoding stops at that step and keeps that beam.
- **Thesis code untouched.** The stop leaves the decoder through the step callback, as Stop does, so the thesis decoders stay untouched (R3).
- **Defaults.**
  - The server default is `False`, so API clients and every parity test keep the published protocol (R2).
  - The UI default is `True`, behind a Settings switch: "Stop when the report starts repeating".
  - Switched off, the turn decodes the whole budget. Its chip then reads `full budget`.
- **What the card shows.**
  - `generate.detail.stopped` is `"repeat"` or `"budget"`.
  - `truncated_mid_sentence` is true only for a budget stop.
  - A repeat stop says "Stopped when the model began repeating itself.".
  - A budget stop with repair on says "Reached the N-token budget; the unfinished last sentence is hidden (Show raw shows it).".
- **Measured on the tiny engine** (200 tokens, uncached): before, 200 tokens and 17.9 s; after, 21 tokens, 2.3 s, `stopped: "repeat"`.
- **Real decoders** (cached Mamba-3 and uncached 13D): they share the same callback, but are first run on the cluster at P7-D.

**The Settings drawer.**
- A Save button, primary; Enter in any field also saves. It confirms with "Settings saved. They apply from your next Send.".
- Each number field shows its range in the label, e.g. "Token budget (16–200)". The 16–200 bound is the spec's (`docs/chat_ui/CHAT_UI_SPEC.md:115`).
- Focusing a field selects its whole value, so typing replaces it.
- A value outside the range shows an error under the field and keeps the previous value. It never snaps to the bound silently: before, typing after "200" gave "200150", and Enter turned that into 200.

Commits:
- `"P4-G: stop decoding when the report repeats"`;
- `"P4-G: settings drawer with Save, ranges, no silent clamp"`;
- the review fix.

*As built (P4-G, commits c0b51ce, d839a35 and 60e9217; the code is authoritative):*
- **The stop.**
  - `repeat_started` sits beside `stream_view` in `app/engine.py`.
  - A private `_RepeatStop` leaves the decoder from the step callback. `Cancelled` is checked first.
  - The stopped result equals the published decoder cut at the stop step, on cached, uncached, beam and greedy decoding (pinned by tests).
  - `truncated_mid_sentence = stopped == "budget" and …`: with beam > 1, the best beam at the first repeat can already be into the next sentence, and that fragment is not a budget cut.
  - `OPTIONS_DOC` documents all of this.
- **Typing a number.**
  - A valid value applies live, as in P4-F.
  - When the text turns invalid, the setting goes back to the value it had when the field was focused, e.g. "300" over 120: 120 → 30 → 120.
  - Enter or Save on invalid text keeps that value and re-selects the bad text.
  - The select-on-focus guard is armed only by a pointer press, so a click after Tab places the caret.
- **Save.** It closes the drawer and confirms in `#saved` (cleared when the drawer reopens). If browser storage refuses, it says so instead of "Settings saved".
- **Card note.** With repair on, a budget stop that hides nothing (no complete sentence) uses the plain note.
- **Checks.**
  - The browser check's `settings` case types real keys: replace-not-append, Tab-then-click, "300" over 120 ends at 120, and the Save path.
  - The settled screenshots are a default (stop-on) turn.
  - Layout: 138 cases (the Save button in every drawer viewport). Node: 299.

- [x] **P4-H** (laptop) An end-to-end Playwright test of the UI, driven like a real user, with fixes for what it finds. The user's request, 2026-10-09: "do a complete testing through the playwright through the ui testing by opening the complete steps in the ui and checking and also perfomring the operation and users behaviour and checking how its processing and fixing any issues that are arriving".
  - **The trigger.** The user also hit "Error: Internal error (ImportError)" with no output on localhost.
    - Root cause, reproduced: a stale dev server. It started at 14:24, before the P9-G2 decoder commits. `Engine.generate` imported the new `scripts/evaluate_report_generation.py` lazily on the first turn, and that file imports `StopDecoding` from the old, already-loaded `hybrid_lm`.
    - Fixed operationally by restarting the server on a8e7efe. A real turn then succeeded.
  - **Requirements** (the brief is `.superpowers/sdd/CHAT_UI_PLAN/task-P4-H-brief.md`):
    - **A1.** Bind the decoders when the engine starts, so a running server never mixes old and new code.
    - **A2.** `/healthz` gains `code_version` and `started_at`.
    - **A3.** An internal error tells the user to restart the server after a code update. It shows the class name only, never the exception text (R1/R7).
    - **B.** The suite is `tests/e2e/test_ui_playwright.py`, using Python Playwright and the installed Chrome via `channel="chrome"`. Its tests are marked `e2e` and `slow`, so `validate.sh` skips them, and the dependency lives in `requirements-e2e.txt`. It has 14 user journeys:
      - first load;
      - upload by file chooser and by drag-and-drop;
      - send with and without a note;
      - re-run;
      - the settings drawer;
      - stop;
      - Show raw;
      - sessions and replay;
      - export;
      - refusals and a server kill;
      - keyboard only;
      - viewports and dark mode;
      - two tabs.

      Every test fails on a console error or an unexpected 4xx/5xx.
    - **C.** Explore first, then fix every defect a user would notice, each with a failing test first. Report the findings as a table.
    - **D.** At most 8 PNGs in `docs/chat_ui/evidence/p4h/`.
  - **Gates:**
    - `validate.sh` prints "All gates passed.";
    - the e2e suite passes;
    - the browser check passes 10/10;
    - the layout check stays green.

  *As built (5afbf8b..24fda5a, 2748f40, 8c19b37; reviewed clean after fix round 1):*
  - **A1.** `app/engine.py` imports `scripts.evaluate_report_generation as erg` when the module loads. A test with `sys.modules[...] = None` proves `generate` imports nothing.
  - **A2.** `/healthz` gains two fields:
    - `started_at`;
    - `code_version`: the short sha from the card's `git_sha`, with a `-dirty` suffix for an uncommitted tree, and `null` in public mode.
  - **A3.** The restart hint is a separate `.note.error-hint`, shown on private `Internal error (…)` messages only.
  - **B.** `tests/e2e/test_ui_playwright.py` holds 14 journeys plus a harness meta-test, 15 tests in all. Every test ends in `assert_clean`, and teardown checks again after a 250 ms settle.
    - Run it with `<venv>/bin/python -m pytest tests/e2e -m e2e -q`. Set `CHAT_UI_E2E_EVIDENCE=docs/chat_ui/evidence/p4h` to regenerate the evidence.
    - The `e2e` marker lives in `tests/conftest.py`.
  - **C. Defects fixed:**
    - F1: the stale-server ImportError;
    - F2: keyboard focus lost when Send or Stop disables itself. The handoff happens only on `:focus-visible`, so a tap does not open the phone keyboard;
    - F3/F8: the re-run hint and the user chips describe what really runs. `isCommand` and the server parser share `tests/frontend/fixtures/commands.json`, and the server is ASCII-only, with drift guards on both sides;
    - F4: a chat is listed when its turn starts;
    - F5: only transport failures go unlogged, marked by `transportFailure` at the source;
    - F6: a deleted chat's running turn stops at the next step, through `Store.message_visible`;
    - F7/F7b: no empty chat is left after a refusal, and a race in that cleanup is fixed.
  - **Deferred:**
    - D1, label stage states: P5-E;
    - D2, the image viewer after a reload: P6.
  - **D.** 7 PNGs in `docs/chat_ui/evidence/p4h/`.
  - **Parked for P6-A and the final review:**
    - `clearNotice()` still focuses the prompt after a tapped Retry or Dismiss, which opens the keyboard on a phone;
    - the server and the page disagree on some whitespace characters (U+001C–1F, U+0085, U+FEFF);
    - the poll's `res.json()` transport marking is untested;
    - a test title is stale (`app.test.mjs:3274`);
    - `pytest.ini`'s `[tool:pytest]` header means pytest ignores the file.

### P5 — Labels and retrieval backend

Gate: laptop — `pytest tests/test_app_labels.py tests/test_app_gallery.py tests/test_app_retrieval_stages.py` green on the tiny gallery. Cluster — gallery built; towers hash-checked; test-split R@1/5/10 equal to `evaluate_cxr_retrieval.py` in the same job; labels cross-check 0 mismatches; 50/50 self-retrieval.

Predictions (R4): P5-B towers identical and no `img_proj` (85%); R@k equal to every digit (90%); 45–90 min on one H100. P5-C 0 mismatches against `y_true` (85%, the group-broadcast argument); f1chexbert stays on CPU (55%), so 8 CPU shards of about 1 h, else 15–45 min on one H100. P5-F 50/50 self-retrieval at rank 1; the live CPU own-rank equals the build's GPU rank on ≥ 48/50 test studies; the labeller service equals the published labels on 50 + 50.

- [x] **P5-A** `app/labels.py` (names, client, rule labeller, agreement) and the `app/labeler.py` service.

**Files:** create `app/labels.py`, `app/labeler.py`, `tests/test_app_labels.py`.

```python
# app/labels.py
"""CheXbert-14 labels for the chat app (CHAT_UI_PLAN.md P5-A).

CHEXBERT_14 is the label order F1CheXbert.target_names reports; P5-F checks it against the live
service rather than trusting this list.
"""
import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Sequence

CHEXBERT_14 = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
               "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
               "Pleural Other", "Fracture", "Support Devices", "No Finding"]


class LabelerUnavailable(RuntimeError):
    pass


class LabelerClient:
    def __init__(self, url: str, timeout: float = 10.0):
        self.url, self.timeout = url.rstrip("/"), timeout

    def label(self, texts: List[str]) -> List[List[int]]:
        req = urllib.request.Request(self.url + "/label", data=json.dumps({"texts": texts}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read())
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise LabelerUnavailable(str(exc))
        if body.get("label_names") != CHEXBERT_14:
            raise LabelerUnavailable("label order mismatch: {}".format(body.get("label_names")))
        return body["labels"]

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/healthz", timeout=2.0) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False


class RuleLabeler:
    """Laptop stand-in: keyword rules, so every label-dependent code path runs without CheXbert."""
    RULES = {"Cardiomegaly": ("cardiomegaly", "enlarged"), "Edema": ("edema",),
             "Pleural Effusion": ("effusion",), "Pneumothorax": ("pneumothorax",),
             "Atelectasis": ("atelectasis",), "Consolidation": ("consolidation",),
             "Pneumonia": ("pneumonia",), "Lung Opacity": ("opacity", "opacities"),
             "Fracture": ("fracture",), "Support Devices": ("tube", "line", "catheter", "pacemaker", "wires"),
             "Lung Lesion": ("nodule", "lesion")}

    def label(self, texts: List[str]) -> List[List[int]]:
        out = []
        for t in texts:
            low = t.lower()
            row = [int(any(k in low for k in self.RULES.get(n, ()))) for n in CHEXBERT_14]
            row[CHEXBERT_14.index("No Finding")] = int(not any(row))
            out.append(row)
        return out

    def healthy(self) -> bool:
        return True


def label_agreement(generated: Sequence[int], neighbor: Sequence[int]) -> Dict[str, Any]:
    """"n/14 labels agree" (the user's wording) plus the positives behind any disagreement (D14)."""
    pairs = list(zip(CHEXBERT_14, generated, neighbor))
    return {"agree": sum(int(a == b) for _, a, b in pairs), "of": len(pairs),
            "both_positive": [n for n, a, b in pairs if a and b],
            "neighbor_only": [n for n, a, b in pairs if b and not a],
            "generated_only": [n for n, a, b in pairs if a and not b]}
```

```python
# app/labeler.py
"""CheXbert-14 microservice (CHAT_UI_PLAN.md P5-A). Runs in .venv_chexbert, which pins
transformers<5 and scikit-learn<1.8 for f1chexbert; like score_chexbert_standalone.py it imports
nothing from hybrid_xmamba.

    .venv_chexbert/bin/uvicorn app.labeler:app --host 127.0.0.1 --port 8001
"""
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="CheXbert labeller")
_labeler = None


def _get():
    global _labeler
    if _labeler is None:
        from f1chexbert import F1CheXbert
        _labeler = F1CheXbert()
    return _labeler


class LabelRequest(BaseModel):
    texts: List[str]


@app.get("/healthz")
def healthz():
    _get()
    return {"status": "ok"}


@app.post("/label")
def label(req: LabelRequest):
    lab = _get()
    return {"label_names": [str(x) for x in lab.target_names],
            # whitespace-normalised, exactly like the hyps.txt/refs.txt lines the published labels came from
            "labels": [[int(v) for v in lab.get_label(" ".join(t.split()))] for t in req.texts]}
```

Tests (the service is tested with a fake `f1chexbert` module injected into `sys.modules`):

```python
# tests/test_app_labels.py
import sys
import types

from app.labels import CHEXBERT_14, RuleLabeler, label_agreement


def test_agreement_counts_and_names_the_differing_positives():
    gen = [0] * 14
    nb = [0] * 14
    gen[CHEXBERT_14.index("Cardiomegaly")] = 1
    nb[CHEXBERT_14.index("Edema")] = 1
    a = label_agreement(gen, nb)
    assert (a["agree"], a["of"]) == (12, 14)
    assert a["generated_only"] == ["Cardiomegaly"] and a["neighbor_only"] == ["Edema"]


def test_rule_labeler_sets_no_finding_only_when_nothing_else_fires():
    rows = RuleLabeler().label(["Findings: lungs clear.", "Small left pleural effusion."])
    assert rows[0][CHEXBERT_14.index("No Finding")] == 1
    assert rows[1][CHEXBERT_14.index("Pleural Effusion")] == 1 and rows[1][CHEXBERT_14.index("No Finding")] == 0


def test_labeler_service_normalises_whitespace_and_reports_names(monkeypatch):
    seen = []

    class FakeF1:
        target_names = CHEXBERT_14

        def get_label(self, text):
            seen.append(text)
            return [0] * 13 + [1]

    monkeypatch.setitem(sys.modules, "f1chexbert", types.SimpleNamespace(F1CheXbert=FakeF1))
    import importlib
    import app.labeler as svc
    importlib.reload(svc)
    from fastapi.testclient import TestClient
    body = TestClient(svc.app).post("/label", json={"texts": ["a  b\nc"]}).json()
    assert body["label_names"] == CHEXBERT_14 and body["labels"] == [[0] * 13 + [1]] and seen == ["a b c"]
```

Commit `"P5-A: labels client, rule labeller, agreement, CheXbert service"`.

*Rulings (controller, 2026-10-09; the task runs in parallel, in its own worktree):*
- **Code and scope.** Transcribe the code above, keeping `CHEXBERT_14` exactly. Do not touch `app/pipeline.py`, `app/server.py` or `app/static/`; P5-E wires these modules in.
- **`LabelerClient`.**
  - It validates the response shape: one row of 14 values in {0, 1} per text. Anything else raises `LabelerUnavailable`, with messages that carry counts only and never text (R7). That includes the order mismatch.
  - An empty input returns `[]` without making a request.
  - `healthy()` also catches `ValueError` and `HTTPException`.
- **`label_agreement`** raises `ValueError` unless both inputs have length 14.
- **The service.**
  - It accepts at most 64 texts, each at most 20,000 characters; more gives 422.
  - `/healthz` returns 503 with a fixed message when f1chexbert fails to load.
- **Tests.**
  - The client runs against a local fake HTTP server, covering good, mismatch, wrong-shape, non-JSON and refused responses.
  - The service limits are covered, plus the 503.
  - A parity test checks that `app/labeler.py` imports nothing from `hybrid_xmamba`.

*As built (761def2, cherry-picked to 6bac709; reviewed clean, no fix round):*
- **`app/labels.py`** (standard library only).
  - The client validates the response shape. Its errors carry an HTTP status or an exception class name, raised `from None`.
- **`app/labeler.py`.**
  - It returns 422 without echoing the input, because FastAPI's default 422 body echoes it (R7).
  - It returns a fixed 503 on `/healthz` and `/label` when the model fails to load.
  - A double-checked lock guards the lazy load, and a failed load is logged server-side.
- **Tests.** `tests/test_app_labels.py` has 67 tests, including the client against the real service served by uvicorn. `CHEXBERT_14` is checked against `CHEXPERT_14_LABELS`.
- **Carried forward:**
  - The client does not split large requests. More than 64 texts, or more than 20,000 characters in one text, gives `LabelerUnavailable` (HTTP 422); P5-F's 50 per call fit.
  - The serving wrapper (P5-E/P7) must run `python -m uvicorn app.labeler:app`, since web deps live in a `PYTHONPATH` overlay.
  - A persistent load failure has no back-off, so pollers must poll more slowly than the load takes.
  - Polish for the final review: a reorder-only mismatch message, overlapping client tests.

- [ ] **P5-B** `scripts/build_retrieval_gallery.py` (with `--tiny`) and `_h100.sh`: the §6.5 files through `evaluate_cxr_retrieval`'s own loaders; tower hash; R@k gate inside the job.

**Files:** create `scripts/build_retrieval_gallery.py`, `scripts/build_retrieval_gallery_h100.sh`, `tests/test_build_retrieval_gallery.py`; modify `tests/test_willi_parity.py`.

Method (why it reproduces the chapter by construction): the builder imports `load_models`, `build_dataloader`, `encode_dataset`, `group_ids_from_texts` and `compute_retrieval_metrics` from `scripts/evaluate_cxr_retrieval.py`, and encodes **both** splits with `build_dataloader("mimic", …, local_parquet_dir=DATA, mimic_split=split)` and `encode_dataset(...)` at batch size 32, the reference script's default. The image transform there (`_img_transform`) gives identical tensors to the decoder's transform for grayscale-origin images (proved in P2-C), so the gallery's image vectors are the decoder tower's whenever the towers are identical.

Core of `main()` (the numbered steps below say what each block is for):

```python
def build(args) -> None:
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoTokenizer
    from app.engine import file_sha256, tensor_sha256
    from scripts.evaluate_cxr_retrieval import (build_dataloader, compute_retrieval_metrics, encode_dataset,
                                                group_ids_from_texts, load_models)
    from scripts.evaluate_report_generation import load_report_generation_module

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text_enc, img_proj, tower13d = load_models(args.checkpoint_13d, device)
    module = load_report_generation_module(args.decoder_checkpoint, args.decoder_config, device=device)
    hashes = {"tower_sha256": tensor_sha256(tower13d), "decoder_tower_sha256": tensor_sha256(module.image_encoder)}
    del module
    towers_identical = hashes["tower_sha256"] == hashes["decoder_tower_sha256"] and img_proj is None
    print("[gallery] towers_identical={} img_proj={}".format(towers_identical, img_proj is not None), flush=True)

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "right"
    emb = {}
    for split in ("test", "train"):
        loader, n = build_dataloader("mimic", "", tok, max_length=256, batch_size=args.batch_size,
                                     num_workers=args.workers, local_parquet_dir=args.data, mimic_split=split)
        emb[split] = encode_dataset(loader, text_enc, img_proj, tower13d, device)   # (img, txt), fp32, unit norm
        print("[gallery] {}: {} rows".format(split, n), flush=True)
    np.save(out / "test_img_emb.npy", emb["test"][0].astype(np.float32))
    np.save(out / "txt_emb_test.npy", emb["test"][1].astype(np.float32))
    np.save(out / "img_emb.npy", emb["train"][0].astype(np.float16))
    np.save(out / "txt_emb.npy", np.concatenate([emb["train"][1], emb["test"][1]]).astype(np.float16))
    gate = {"app": compute_retrieval_metrics(emb["test"][0], emb["test"][1])}

    frames = {s: pd.read_parquet(Path(args.data) / "{}.parquet".format(s)) for s in ("train", "test")}
    # Exactly the published reference string (run_checkpoint_inspection's f-string, no `or ""`: a None
    # must print as the published refs.txt printed it), then write_hyps_refs's whitespace collapse.
    report = lambda r: " ".join("Findings: {} Impression: {}".format(r.get("findings", ""),
                                                                     r.get("impression", "")).strip().split())
    texts = [report(r) for _, r in frames["train"].iterrows()] + [report(r) for _, r in frames["test"].iterrows()]
    (out / "report_texts.txt").write_text("\n".join(texts) + "\n")
    groups = group_ids_from_texts(texts)
    order = np.argsort(groups, kind="stable")
    starts = np.flatnonzero(np.r_[True, groups[order][1:] != groups[order][:-1]])
    n_train = len(frames["train"])
    np.save(out / "txt_groups.npy", groups)
    np.save(out / "group_order.npy", order)
    np.save(out / "group_starts.npy", starts)
    np.save(out / "txt_test_groups.npy", group_ids_from_texts(texts[n_train:]))
    np.save(out / "txt_split.npy", np.r_[np.zeros(n_train, np.int8), np.ones(len(frames["test"]), np.int8)])
    np.save(out / "txt_split_row.npy", np.r_[np.arange(n_train), np.arange(len(frames["test"]))])
    np.save(out / "img_txt_row.npy", np.arange(n_train))            # one image and one report per train study
    for split, name in (("train", "img_meta.parquet"), ("test", "test_meta.parquet")):
        meta = frames[split][["study_id", "subject_id", "dicom_id", "view", "image"]].copy()
        meta["file_sha256"] = [file_sha256(Path(p)) for p in meta["image"]]
        meta.to_parquet(out / name, index=False)
    manifest = dict(hashes, towers_identical=towers_identical, img_proj_present=img_proj is not None,
                    counts={"images": n_train, "report_rows": len(texts), "report_groups": int(len(starts)),
                            "test": len(frames["test"])}, labels_status="pending", gate_rk=gate, **provenance(args))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "gate_rk.json").write_text(json.dumps(gate, indent=2))
```

`report` collapses whitespace exactly like `write_hyps_refs`, so `report_texts.txt` lines equal the published `refs.txt` lines for test rows (asserted in P5-C). `provenance(args)` returns build id, UTC time, git SHA, job id, checkpoint paths and their `file_sha256`. If `encode_dataset` or `build_dataloader` signatures differ from the reading above (`scripts/evaluate_cxr_retrieval.py:354` and `:405`), follow the code.

Steps inside `main()`:

1. `text_enc, img_proj, tower13d = load_models(ckpt_13d, device)`.
2. `module = load_report_generation_module(decoder_ckpt, "hybrid_150m_m3_rrg", device)`; `decoder_tower_sha256 = tensor_sha256(module.image_encoder)`, `tower_sha256 = tensor_sha256(tower13d)`; `towers_identical = equal and img_proj is None`; print both; `del module`. If not identical, keep going and record it; P5-D then loads the 13D tower as a second module for the image→report query.
3. Test split first (2,663 rows): `test_img, test_txt = encode_dataset(...)` → `test_img_emb.npy`, `txt_emb_test.npy` (fp32). `gate_rk.app = compute_retrieval_metrics(test_img, test_txt)`.
4. Train split (191,462 rows): `img, txt = encode_dataset(...)` → `img_emb.npy` (fp16), and `txt_emb.npy = concat(train_txt, test_txt)` (fp16).
5. Report texts: `"Findings: {} Impression: {}".format(findings, impression).strip()`, sanitised, train rows then test rows → `report_texts.txt`; `txt_groups = group_ids_from_texts(texts)`; `group_order = argsort(txt_groups, kind="stable")`; `group_starts` from the sorted ids; `txt_test_groups = group_ids_from_texts(test_texts)`; `txt_split`, `txt_split_row`, `img_txt_row`.
6. Metadata parquets with `file_sha256` of every image file (needed for "identical to a gallery image").
7. `manifest.json` per §6.5, `labels_status: "pending"`.
8. Optional cross-check: if `${SCRATCH_ROOT}/isbi_gallery_adapted.pt` exists (decoder-tower fp16 train embeddings from the ISBI figure job), record `max |img_emb − isbi|` in the manifest. A value near fp16 rounding confirms D4 independently.

`--tiny OUT`: writes a synthetic gallery in the same layout with no torch model and no data — 200 random unit 16-d image vectors, 300 report rows from `TINY_VOCAB` with deliberate duplicates, 40 test rows, random labels, and 320×320 gray JPEGs under `OUT/images/` referenced by the meta parquets. Tests use it through a `tiny_gallery` fixture (`tmp_path`).

`--compare-rk OUT`: reads `OUT/gate_rk.json` (`app`) and the newest `OUT/reference_rk/phase6_mimic_*.json` (`metrics`), writes `reference` and `equal` (every `i2t_R@k` and `t2i_R@k` equal), exits 1 if unequal.

Wrapper `scripts/build_retrieval_gallery_h100.sh`: `#SBATCH --gpus=1`, `--mem=64G`, `--cpus-per-task=16`, `--time=04:00:00`, job name `chat_gallery`, no `--qos`. Body:

```bash
BUILD_ID="${BUILD_ID:-$(date +%Y%m%d)_${SLURM_JOB_ID:-local}}"
OUT="${OUT:-$HOME/chat_sessions/gallery/${BUILD_ID}}"
python scripts/build_retrieval_gallery.py --checkpoint-13d "${CKPT_13D}" --decoder-checkpoint "${CHECKPOINT}" \
  --data "${DATA}" --out "${OUT}" --workers "${SLURM_CPUS_PER_TASK:-8}"
python scripts/evaluate_cxr_retrieval.py --checkpoint "${CKPT_13D}" --dataset mimic \
  --local-parquet-dir "${DATA}" --mimic-split test --output-dir "${OUT}/reference_rk"
python scripts/build_retrieval_gallery.py --compare-rk "${OUT}"
```

Tests (laptop): `--tiny` writes every §6.5 file with consistent shapes; `group_starts` partitions `group_order`; `--compare-rk` returns 1 on a doctored reference file and 0 on an equal one; the parity test checks the wrapper's directives, `HF_HUB_OFFLINE`, `--mimic-split test` and `--compare-rk`.

Commit `"P5-B: gallery builder (+tiny) reusing the retrieval chapter's loaders"`. Then on lx01: `sbatch scripts/build_retrieval_gallery_h100.sh`; tick with job id, `towers_identical`, `gate_rk.equal`, counts and wall time.

*Rulings (controller, 2026-10-09; the task runs in parallel, in its own worktree):* mirror P9-G3's wrapper hygiene.
- **The job log (R7).** It carries only `===`, `[gallery]`, `RESULT` and `ERROR` lines. The raw output of both Python steps goes to `OUT/build.log` and `OUT/reference_rk.log`. A failure prints `ERROR <step> exit=<code>`.
- **Provenance.** The first line is the `.sync_stamp` sync line, and the manifest records the sha and the clean/dirty flag.
- **Requeue and R8.**
  - `--requeue` and `--open-mode=append`.
  - The job refuses when `OUT/manifest.json` exists, since a finished build is never overwritten.
  - `OUT` sits under `CHAT_HOME/gallery/<build_id>`, never under `/outputs/` or the thesis checkout.
  - `BUILD_ID` defaults to `<date>_<job id>`, so a requeue resumes in the same directory.
- **Reference and compare.**
  - `evaluate_cxr_retrieval.py` runs unchanged (R3); follow its real CLI.
  - `--compare-rk` prints `RESULT {"gate_rk_equal": …}`.
- **The `tiny_gallery` test fixture** is shared, for reuse by P5-C, P5-D and P5-E.
- **Tests.**
  - A sandbox rehearsal of the wrapper.
  - Unit tests of the pure pieces: grouping, report text, manifest and provenance.

- [ ] **P5-C** `scripts/label_gallery_reports.py` and `_h100.sh` (`.venv_chexbert`): `labels.npy` via group representatives; canary and projection; a 0-mismatch cross-check on the 2,663 test references.

**Files:** create `scripts/label_gallery_reports.py`, `scripts/label_gallery_reports_h100.sh`, `tests/test_label_gallery_reports.py`; modify `tests/test_willi_parity.py`.

Speed is the constraint. The only measurement is indirect: CheXbert scoring is a CPU job (`scripts/score_chexbert_h100.sh`: 4 CPUs, no GPU) and job 2525606 labelled 2 × 2,663 reports in about 15 minutes, so **about 0.16 s per report on CPU**. With roughly 150–190k unique groups that is 7–8 hours on one CPU job. Whether `F1CheXbert` uses a GPU when one is present is unknown. Hence a GPU canary first, and a sharded CPU path as the fallback.

Prediction (R4): f1chexbert does not move to CUDA on its own (55%); if it does, 15–45 min on one H100; the CPU path is 8 shards of about 1 h.

The script imports only stdlib, numpy and f1chexbert (it runs in `.venv_chexbert`, like `score_chexbert_standalone.py`):

```python
# scripts/label_gallery_reports.py (core)
def representatives(groups: "np.ndarray") -> "np.ndarray":
    """First row of every duplicate group, in group-id order."""
    _, first = np.unique(groups, return_index=True)
    return first


def label_rows(labeler, texts: List[str], rows: Sequence[int], budget_s: float, canary: int = 1000) -> "np.ndarray":
    out = np.zeros((len(rows), 14), dtype=np.uint8)
    t0 = time.perf_counter()
    for i, r in enumerate(rows):
        out[i] = [int(v) for v in labeler.get_label(texts[r])]
        if i + 1 == canary:
            projected = (time.perf_counter() - t0) / canary * len(rows)
            print("[labels] canary: {:.3f} s/report, projected {:.0f} s for {} rows".format(
                (time.perf_counter() - t0) / canary, projected, len(rows)), flush=True)
            if projected > budget_s:
                raise SystemExit(2)          # wrapper reads exit 2 as "use the sharded CPU path"
    return out
```

Modes:

- Default (one job): read `report_texts.txt` and `txt_groups.npy`; `reps = representatives(groups)`; `labeler = F1CheXbert()`; print the device of its model's first parameter (and `inspect.signature(F1CheXbert.__init__)`, so a `device` argument can be passed if it exists); `label_rows(...)` over `reps`; broadcast with `labels = rep_labels[np.searchsorted(group_ids_of_reps, groups)]`; write `labels.npy` (uint8), `label_names.json` (`labeler.target_names`), set `manifest.labels_status = "done"`.
- `--shard i --of N`: label `reps[i::N]` only and write `labels_shard_<i>.npy` plus its row indices.
- `--merge --of N`: combine the shards, broadcast, write as above.
- Always last: the cross-check. For the 2,663 test rows (in `txt_split_row` order), (a) `report_texts` lines must equal `results/report_gen_m3_test_split_s42/refs.txt` line for line, and (b) `labels` must equal `y_true` of that dump's `chexbert_labels.json`. Write `labels_check.json` with both mismatch counts; exit 1 if either is non-zero.

Wrapper `scripts/label_gallery_reports_h100.sh`: `#SBATCH --gpus=1`, `--time=02:00:00`, the ga03 exclusion (it sources a venv), `VENV_ACTIVATE=.venv_chexbert/bin/activate`, and the CheXbert environment exactly as `score_chexbert_h100.sh` sets it: **no `HF_HOME` override** (its weights live in the default cache) and `HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"`. On exit 2 it prints the fallback command: `sbatch --array=0-7 scripts/label_gallery_reports_cpu_h100.sh` (CPU, `--qos=aisc`, 4 CPUs, 16G, `--time=02:00:00`, `--shard ${SLURM_ARRAY_TASK_ID} --of 8`), then `MERGE=1 sbatch scripts/label_gallery_reports_cpu_h100.sh`.

Laptop tests use a fake `f1chexbert` module and a tiny gallery: representatives and broadcasting; the canary's exit 2 with a tiny budget; shard + merge equals the single-job result; a doctored `y_true` or `refs.txt` gives exit 1.

Commit `"P5-C: CheXbert labels for the gallery, cross-checked"`. Then `sbatch scripts/label_gallery_reports_h100.sh`; tick with job id, mismatches (expected 0), groups labelled, wall time.

- [ ] **P5-D** `app/gallery.py`: load with the tower check, image→image, image→report over groups, own rank (chapter protocol), identical-image lookup, test studies.

*Carried (controller, 2026-10-10):*
- **From the P5-B review.** `Gallery.open` refuses a gallery whose `manifest.json` lacks `gate_rk.equal == true` (a build whose gate failed or never ran), as well as one whose tower hash differs.
- **Tiny sizes.** P5-B's tiny gallery has 200 images, 40 test rows and 240 report rows. Read every size from `manifest["counts"]`, never from this plan's numbers.

**Files:** create `app/gallery.py`, `tests/test_app_gallery.py`.

```python
def _topk(sims: np.ndarray, k: int) -> np.ndarray:
    k = min(k, sims.shape[0])
    idx = np.argpartition(-sims, k - 1)[:k]
    return idx[np.argsort(-sims[idx], kind="stable")]


class Gallery:
    # open(): reads manifest.json; raises GalleryMismatch when expect_tower_sha256 is given and differs
    # from manifest["tower_sha256"]; loads img_emb/txt_emb as float32 RAM copies (~0.8 GB at full size),
    # txt_emb_test as float32, the int arrays, labels (or None while labels_status is "pending"), the meta
    # parquets and report_texts.txt.

    def report_matches(self, query, k):
        sims = self.txt_emb @ query.astype(np.float32)
        group_max = np.maximum.reduceat(sims[self.group_order], self.group_starts)   # best member per group
        out = []
        for rank, g in enumerate(_topk(group_max, k), start=1):
            end = self.group_starts[g + 1] if g + 1 < len(self.group_starts) else len(self.group_order)
            members = self.group_order[self.group_starts[g]:end]
            best = int(members[np.argmax(sims[members])])
            out.append({"rank": rank, "similarity": float(group_max[g]), "group": int(g),
                        "group_size": int(len(members)), "txt_row": best,
                        "report": self.report_texts[best], "labels": self._labels(best)})
        return out

    def own_report_rank(self, query, test_row):
        """The chapter's protocol (D5): strict pairing inside the 2,663 test reports."""
        sims = self.txt_emb_test @ query.astype(np.float32)
        own = sims[test_row]
        same = self.txt_test_groups == self.txt_test_groups[test_row]
        rank = 1 + int((sims > own).sum())
        return {"rank": rank, "of": int(sims.shape[0]), "rank_dedup": 1 + int((sims > sims[same].max()).sum()),
                "hit_at_10": rank <= 10,
                "protocol": "i2t, official test split, strict pairing (compute_retrieval_metrics, groups=None)"}
```

`image_neighbors` returns `rank, similarity, gallery_row, study_id, txt_row, image_url` (`/v1/gallery/images/<row>`) and `labels` (from `img_txt_row`). `find_identical(sha)` checks the train and test `file_sha256` maps. `test_study(row)` returns the image path, study id and the reference (`report_texts[n_train + row]`). `list_test_studies(query, limit)` filters on the study-id prefix.

Tests on the tiny gallery: every one of 20 gallery images retrieves itself first; `report_matches` returns distinct groups in non-increasing similarity, each scored by its best member (checked by brute force); `own_report_rank` equals `1 + #(sims > own)` and `rank_dedup ≤ rank`; ranks on the tiny test split reproduce `compute_retrieval_metrics(test_img, test_txt)` R@1/5/10 when recomputed from per-row ranks; `open` raises `GalleryMismatch` for a wrong tower hash; `find_identical` finds a train image by file hash.

Commit `"P5-D: gallery queries (image→image, image→report, own rank)"`.

- [ ] **P5-E** Pipeline stages `retrieve`, `label`, `score` (and `app/scoring.py`); `POST /v1/retrieve`, `POST /v1/label`, `GET /v1/test-studies`; skipped semantics; redaction of the new fields.

**Files:** modify `app/pipeline.py`, `app/server.py`; create `app/scoring.py`, `tests/test_app_retrieval_stages.py`.

```python
# app/scoring.py
"""Per-turn scores with the thesis tables' own functions (CHAT_UI_PLAN.md P5-E)."""
from typing import Any, Dict, List, Optional

from scripts.bootstrap_compare import chexbert_f1
from scripts.evaluate_report_generation import corpus_bleu, rouge_l_score


def score_pair(hyp: str, ref: str, y_hyp: Optional[List[int]] = None,
               y_ref: Optional[List[int]] = None) -> Dict[str, Any]:
    h, r = hyp.split(), ref.split()
    out = {"rouge_l": rouge_l_score(h, r), "bleu_1": corpus_bleu([h], [r], 1), "bleu_4": corpus_bleu([h], [r], 4)}
    if y_hyp is not None and y_ref is not None:
        out["chexbert_14_micro_f1"] = chexbert_f1([y_ref], [y_hyp], "micro")
        out["exact_match_14"] = list(y_ref) == list(y_hyp)
    return out
```

Stage behaviour:

- `retrieve`: skipped `gallery_unavailable` when no gallery; skipped `k_zero` when both k are 0. Queries with `Encoded.pooled` (D4). The own rank is added when the turn has a `test_row` (picker) or the upload's file hash matches a test image; `identical_to` goes into the preprocess detail on a hash match.
- `label`: skipped `label_off` or `labeler_unavailable` (the turn still ends `done`). Labels the generated report; adds `neighbor_agreement` for every image neighbour that has labels.
- `score`: runs only with a reference — `options.reference` (private only; public sends a `warning` `reference_ignored_public` and skips) or the test row's own reference. CheXbert parts need the reference labelled too (one more labeller call). For test rows in private mode, `published` carries the published model line and the floor line from the dumps, plus `live_equals_published`.
- `PublishedDumps` (in `app/pipeline.py`): `@dataclass class PublishedDumps: model_hyps: List[str]; floor_hyps: List[str]` with `@classmethod load(cls, model_dir: Path, floor_dir: Path)` reading each `hyps.txt` (lines aligned with `test.parquet` rows) and `line(kind: str, row: int) -> Optional[str]`. `create_app(published_dirs={"model": …, "floor": …})` builds it; `None` (laptop) means no `published` key. Only the default engine (`hybrid_150m_m3_rrg`) gets a `model_report` line, since the dump is that model's.
- `POST /v1/retrieve` (multipart image, `k_images`, `k_reports`): runs preprocess, encode and retrieve on the same single worker; returns the redacted detail. `POST /v1/label {"text"}` → `{"chexbert_14": {...}}`; 503 if the labeller is down. `GET /v1/test-studies?q=&limit=` (private only; 403 public).

Tests (tiny engine + tiny gallery + `RuleLabeler`): a turn's `retrieve` detail has `k_images` neighbours and `k_reports` groups with labels; `label` has one agreement per neighbour whose counts match `label_agreement`; a `test_row` turn has `true_report_rank` equal to `Gallery.own_report_rank` and a `score` stage with `reference_source: "test_split"`; `score_pair` equals `bootstrap_compare.per_sample_rouge_l` on the same pair; labeller down → `label` skipped `labeler_unavailable` and status `done`; no gallery → `retrieve` skipped `gallery_unavailable`; public mode: `report_matches[].report` absent, `test_row` option → 403, `reference` → warning and no `score` event.

Commit `"P5-E: retrieve/label/score stages + endpoints"`.

- [ ] **P5-F** Live-path gates job `scripts/chat_retrieval_gates.py` with `_h100.sh` (CPU): self-retrieval, live own-rank vs the build's rank, labeller service vs the published labels on 50 + 50.

**Files:** create `scripts/chat_retrieval_gates.py`, `scripts/chat_retrieval_gates_h100.sh`; modify `tests/test_willi_parity.py`.

The job starts the labeller in the CheXbert environment of `score_chexbert_h100.sh`, with its web overlay and on a free loopback port — `env -u HF_HOME HF_HUB_OFFLINE="${CHEXBERT_HF_HUB_OFFLINE:-0}" PYTHONPATH=.chat_deps_chexbert .venv_chexbert/bin/python -m uvicorn app.labeler:app --host 127.0.0.1 --port "${LABELER_PORT}" &` — waits for `/healthz`, then runs the driver in `.venv` with `PYTHONPATH=.chat_deps` (CPU, 8 threads; the driver takes `--labeler-url http://127.0.0.1:${LABELER_PORT}`):

```python
# scripts/chat_retrieval_gates.py (core)
def run(args) -> Dict[str, Any]:
    eng = build_engine("real", checkpoint=args.checkpoint, model_config=args.model_config, device="cpu", threads=8)
    gal = Gallery.open(Path(args.gallery), expect_tower_sha256=None)
    train = pd.read_parquet(Path(args.data) / "train.parquet")
    test_img = np.load(Path(args.gallery) / "test_img_emb.npy")
    rows = np.linspace(0, len(train) - 1, 50).astype(int)
    self_hits, misses = 0, []
    for r in rows:                                   # 1. self-retrieval through the live upload path
        _, prep = eng.preprocess(Path(train.iloc[r]["image"]).read_bytes())
        _, enc = eng.encode(prep)
        top = gal.image_neighbors(enc.pooled.numpy(), 2)
        self_hits += int(top[0]["gallery_row"] == r)
        if top[0]["gallery_row"] != r:
            misses.append({"row": int(r), "got": top[0]["gallery_row"], "gap": top[0]["similarity"] - top[1]["similarity"]})
    test = pd.read_parquet(Path(args.data) / "test.parquet")
    rank_equal = 0
    for t in range(50):                              # 2. live own-rank vs the build's GPU embedding
        _, prep = eng.preprocess(Path(test.iloc[t]["image"]).read_bytes())
        _, enc = eng.encode(prep)
        rank_equal += int(gal.own_report_rank(enc.pooled.numpy(), t)["rank"] == gal.own_report_rank(test_img[t], t)["rank"])
    pub = json.loads(Path(args.published_labels).read_text())   # 3. the labeller vs the published labels
    client = LabelerClient(args.labeler_url, timeout=120)
    hyps = Path(args.published_hyps).read_text().splitlines()[:50]
    refs = Path(args.published_refs).read_text().splitlines()[:50]
    label_equal = sum(a == b for a, b in zip(client.label(hyps), pub["y_pred"][:50])) + \
                  sum(a == b for a, b in zip(client.label(refs), pub["y_true"][:50]))
    return {"self_retrieval": self_hits, "misses": misses, "own_rank_equal": rank_equal,
            "labeller_equal": label_equal, "label_names_ok": pub["label_names"] == CHEXBERT_14}
```

Writes `results/chat_retrieval_gates_<job>/gates.json`; exits 1 unless `self_retrieval == 50`, `labeller_equal == 100` and `label_names_ok`. `own_rank_equal` is recorded against the ≥ 48/50 prediction, not gated (CPU vs GPU embeddings can swap near-ties, D6).

Commit `"P5-F: live retrieval + labeller gates job"`. Then `sbatch scripts/chat_retrieval_gates_h100.sh`; tick with the job id and the three numbers.

### P6 — Image experience (the user's five additions)

Gate: each of the five features passes its tests and the browser checklist on the tiny engine and tiny gallery; P7-D repeats the checklist on a real test study in private mode.

- [ ] **P6-A** Taking an image: click, drag-drop and paste into the image well; a preview (thumbnail, name, pixel size, MB, remove) before sending; client-side checks.

**Files:** create `app/static/composer.js`, `tests/frontend/composer.test.mjs`; modify `app/static/app.js` (replace the P4-D picker).

```javascript
// app/static/composer.js
// Image intake for the composer: click, drag-drop, paste; a preview before sending (CHAT_UI_PLAN.md P6-A).
// The server re-validates everything; these checks only save a round trip.
export const ACCEPTED = ['image/png', 'image/jpeg', 'image/webp'];
export const MAX_BYTES = 20 * 1024 * 1024;

export function checkFile(file) {   // -> null, or the message to show
  if (!file) return 'No file.';
  if (/\.dcm$/i.test(file.name || '') || file.type === 'application/dicom')
    return 'DICOM is not supported. Export the image as PNG or JPEG first.';
  if (!ACCEPTED.includes(file.type)) return 'Use a PNG, JPEG or WEBP image.';
  if (file.size > MAX_BYTES) return 'Image is larger than 20 MB.';
  return null;
}

export function pickImageFromClipboard(items) {   // DataTransferItemList-like -> File | null
  for (const it of items || []) if (it.kind === 'file' && ACCEPTED.includes(it.type)) return it.getAsFile();
  return null;
}

export function formatBytes(n) {
  return n < 1024 * 1024 ? `${Math.max(1, Math.round(n / 1024))} KB` : `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function attachComposer({ well, input, preview, onChange, onError }) {
  let current = null;
  let url = null;
  const set = (file) => {
    const err = checkFile(file);
    if (err) { onError(err); return; }
    if (url) URL.revokeObjectURL(url);
    current = file;
    url = URL.createObjectURL(file);
    const img = new Image();
    img.alt = `Preview of ${file.name || 'pasted image'}`;
    img.onload = () => { meta.textContent = `${file.name || 'pasted image'} · ${img.naturalWidth}×${img.naturalHeight} px · ${formatBytes(file.size)}`; };
    img.src = url;
    const meta = document.createElement('span');
    const remove = Object.assign(document.createElement('button'), { type: 'button', textContent: '×' });
    remove.setAttribute('aria-label', 'Remove image');
    remove.onclick = () => clear();
    preview.replaceChildren(img, meta, remove);
    preview.hidden = false;
    onChange(current);
  };
  const clear = () => {
    if (url) URL.revokeObjectURL(url);
    current = url = null;
    preview.replaceChildren();
    preview.hidden = true;
    input.value = '';
    onChange(null);
  };
  well.addEventListener('click', () => input.click());
  well.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
  input.addEventListener('change', () => input.files[0] && set(input.files[0]));
  for (const t of ['dragenter', 'dragover']) well.addEventListener(t, (e) => { e.preventDefault(); well.classList.add('dragging'); });
  for (const t of ['dragleave', 'drop']) well.addEventListener(t, () => well.classList.remove('dragging'));
  well.addEventListener('drop', (e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) set(f); });
  document.addEventListener('paste', (e) => { const f = pickImageFromClipboard(e.clipboardData && e.clipboardData.items); if (f) { e.preventDefault(); set(f); } });
  return { get: () => current, clear };
}
```

Tests (`node --test`): `checkFile` accepts the three types, refuses `.dcm`, a GIF, and 20 MB + 1 byte with the right messages; `pickImageFromClipboard` returns the first image item and ignores text items; `formatBytes` formats KB and MB.

The send path clears the preview only after the server accepts the turn (HTTP 200), so a refused upload keeps the preview and shows the server's message.

Commit `"P6-A: image intake (click/drag/paste) with preview"`.

- [ ] **P6-B** Your image in the chat: upload variants stored; `GET /v1/messages/{id}/image`; the thumbnail survives reload and reopen; the 224×224 model input beside the original.

**Files:** modify `app/pipeline.py` (store variants during preprocess), `app/server.py`, `app/static/render.js`, `app/static/state.js`; create `tests/test_app_images.py`.

- Uploads are already stored at `preprocess` since P3-D: `original.<ext>` (the bytes as uploaded), `thumb.jpg` (`thumbnail_jpeg`, 512 px), `model_input.png` (`Prepared.model_input`). This task serves them.
- `GET /v1/messages/{id}/image?variant=original|thumb|model_input` accepts the user or the assistant message id of a turn. Uploads: `Cache-Control: private, max-age=3600`. Test-split turns (private only; 403 in public): the original is the 320 px dataset JPEG, the thumbnail is that same file, the model input is generated in memory; all with `Cache-Control: no-store`. Unknown variant → 422; another client's message (public) → 404.
- Render: the user turn shows the thumbnail (through `loadImage`, D23). The assistant card gets an "Images" row: "Your X-ray" (thumbnail, labelled with the original pixel size from the preprocess detail) and "What the model saw (224×224)". Each opens the viewer (P6-E).

Tests: after a turn, each variant returns 200 with the right content type, and `model_input` is 224×224; after `GET /v1/sessions/{id}` (a reload), the user message carries the three URLs and they still resolve; a test-row image is 403 in public mode and served with `no-store` in private mode; after `DELETE` the image is 404.

Commit `"P6-B: persisted upload variants + images in the card"`.

- [ ] **P6-C** Similar X-rays: a neighbour grid with similarity and "n/14 labels agree" (with the differing positives); `GET /v1/gallery/images/{row}` private-only with `no-store`.

**Files:** modify `app/server.py`, `app/static/render.js`, `app/static/styles.css`; extend `tests/test_app_images.py`.

- Endpoint: `GET /v1/gallery/images/{row}` → the stored 320 px JPEG for that gallery row; 403 in public mode; 404 out of range; `Cache-Control: no-store`. The row is an integer index, never a path.
- Card section "Similar X-rays (13D tower)": one figure per neighbour with the image (private) or a neutral placeholder (public), `#rank · similarity` and a similarity bar, then (private only) "n/14 labels agree" once the `label` stage has ended ("labels…" before; "labels unavailable" if skipped). Public mode shows rank and similarity only (U2). When positives differ, chips name them ("neighbour: Edema", "report: Cardiomegaly"). A badge marks "identical to your upload" when the neighbour's file hash equals the upload's. A click opens the viewer side by side: the query on the left, the neighbour on the right, labelled "320×320 px (stored gallery size)".

Tests: the endpoint's 403/404/200 and headers; the redacted public retrieve detail has no `image_url`, so the page never requests gallery images in public mode.

Commit `"P6-C: similar X-rays grid with label agreement"`.

- [ ] **P6-D** Matching reports: a ranked report list (13D encoders), the own-report rank badge for test studies, and the test-split picker (private).

**Files:** modify `app/static/render.js`, `app/static/app.js`; extend `tests/test_app_retrieval_stages.py`.

- Card section "Matching reports (13D image and text encoders)": rank, similarity, group size (e.g. "×37 identical reports"), the report clamped to three lines with "Show all" (private), and its label chips. Public mode shows rank and similarity only (U2).
- Badge for test studies: "Own report: rank 3 of 2,663 test reports · R@10 hit". The tooltip shows the dedup-aware rank and the protocol line from the detail.
- Picker (private): a "Test-split study" button in the composer opens a searchable list from `GET /v1/test-studies`; choosing one sends a turn with `options.test_row` and no upload. The card then also shows, from the `score` detail, the published (GPU) report and the retrieval-floor report beside the live one, with "identical" / "differs" against the published line.

Tests: a `test_row` turn on the tiny stack carries `true_report_rank` with `of` equal to the tiny test split size and a `score` detail with `published`; `GET /v1/test-studies` is 403 in public mode.

Commit `"P6-D: matching reports, own-rank badge, test-split picker"`.

- [ ] **P6-E** Viewer: full screen, single or side by side (query and the selected match), zoom, pan, brightness, contrast, keyboard, focus trap.

**Files:** create `app/static/viewer.js`, `tests/frontend/viewer.test.mjs`; modify `app/static/styles.css`.

```javascript
// app/static/viewer.js (pure part, node-tested)
export const MIN_SCALE = 0.25, MAX_SCALE = 16;
export function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }

export function zoomAt(view, factor, cx, cy) {
  // Zoom about the cursor: the image point under (cx, cy) stays under it.
  const scale = clamp(view.scale * factor, MIN_SCALE, MAX_SCALE);
  const k = scale / view.scale;
  return { ...view, scale, x: cx - (cx - view.x) * k, y: cy - (cy - view.y) * k };
}

export function clampPan(view, imgW, imgH, boxW, boxH) {
  // Keep at least 10% of the image inside the pane.
  const w = imgW * view.scale, h = imgH * view.scale;
  return { ...view, x: clamp(view.x, -w + 0.1 * w, boxW - 0.1 * w), y: clamp(view.y, -h + 0.1 * h, boxH - 0.1 * h) };
}

export function filterCss({ brightness = 100, contrast = 100 }) {
  return `brightness(${brightness}%) contrast(${contrast}%)`;
}
```

DOM part, `openViewer({left, right})` where each side is `{url, label, note}`:

- A full-screen overlay (`role="dialog"`, `aria-modal="true"`) with one pane, or two panes side by side (stacked below 700 px).
- "Link panes" toggle, on by default: zoom and pan apply to both panes.
- Wheel and pinch (pointer events) zoom at the cursor; drag pans; double-click toggles fit / 1:1; buttons +, −, Fit, 1:1, Reset.
- Brightness and contrast sliders, 0–300%, CSS filter only (display only; the model input is unaffected).
- Keyboard: Esc closes, `+`/`-` zoom, `0` resets, arrow keys pan. Focus is trapped inside and returns to the opener on close.
- A footer per pane shows the label and the pixel size ("2544×3056 px", or "320×320 px (stored gallery size)").

Tests (`node --test`): `zoomAt` keeps the point under the cursor fixed and clamps to [0.25, 16]; `clampPan` keeps 10% visible; `filterCss` output.

Commit `"P6-E: full-screen viewer (side by side, zoom, pan, brightness, contrast)"`.

- [ ] **P6-F** End-to-end check of the five features on the tiny stack, with screenshots; the cluster re-check is part of P7-D.

Run `venv/bin/python -m app.server --engine tiny --gallery <tiny gallery dir> --labeler rule` and check each feature: (1) click, drag-drop and paste each attach with a preview, and refused files show the message; (2) after a reload and after reopening the session, the X-ray and the 224×224 model input are both shown; (3) the similar-X-rays grid shows scores and "n/14 labels agree"; (4) the matching-reports list and, for a picked test study, the own-rank badge; (5) the viewer opens from every image, side by side for a neighbour, with zoom, pan, brightness and contrast working by mouse and keyboard. Record screenshots and the checklist in the evidence. Commit `"P6-F: five image features checked end to end (tiny)"`.

### P7 — Cluster serving

Gate: from the laptop, a real test study chosen in the picker decodes through the tunnel; its report equals the P2-E CPU golden line for that study (same node type; otherwise within the P1-C drift, which the card states); the five image features work on real data in private mode; the preemption drill passes.

- [ ] **P7-A** `scripts/chat_app_smoke_h100.sh`: on the cluster, `app.server` imports with the `.chat_deps` overlay and `app.labeler` with `.chat_deps_chexbert` (both installed by P0-G, so the shared venvs stay untouched).

```bash
#!/bin/bash
# ============================================================================
# CHAT_UI_PLAN.md P7-A — the chat app imports on the cluster, each half in its
# own venv plus its overlay (P0-G). Installs nothing into the shared venvs (R8).
#   bash scripts/chat_remote.sh submit scripts/chat_app_smoke_h100.sh
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node, x86 venv incompatible; gx13v1: faulty GPU
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --job-name=chat_app_smoke
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"
PYTHONPATH=.chat_deps .venv/bin/python -c "import fastapi, app.server; print('[setup] app.server imports; fastapi', fastapi.__version__)"
PYTHONPATH=.chat_deps_chexbert .venv_chexbert/bin/python -c "import app.labeler, transformers; print('[setup] app.labeler imports; transformers', transformers.__version__)"
```

Parity test: CPU directives, ga03 exclusion, both `PYTHONPATH=` overlays, and no `pip install` in the file. Commit `"P7-A: cluster import smoke for the app"`; then `bash scripts/chat_remote.sh sync && bash scripts/chat_remote.sh submit scripts/chat_app_smoke_h100.sh`, and tick with the job id and the printed versions.

- [ ] **P7-B** `scripts/serve_chat_h100.sh` and the `python -m app.server` CLI: labeller and API, free port, endpoint file, token file, SIGTERM handling, requeue.

CLI: `python -m app.server --engine tiny|real --mode private|public --home DIR [--device cpu|cuda] [--gallery DIR] [--labeler URL|rule|none] [--host 127.0.0.1] [--port 0] [--endpoint-file PATH] [--token-file PATH] [--models m3[,13d]] [--threads 8] [--drift-note TEXT] [--allow-compile]`. It picks a free port when `--port 0`, writes `<hostname>:<port>` to the endpoint file once uvicorn has started, reads the token from `--token-file` (never from the command line or the environment, so it stays out of `scontrol show job` and shell history), and on shutdown marks running turns `error` with `server_restart`.

```bash
#!/bin/bash
# ============================================================================
# CHAT_UI_PLAN.md P7-B — the chat server: CheXbert labeller + API in one CPU job.
#   sbatch scripts/serve_chat_h100.sh                        # private, loopback
#   MODE=public sbatch scripts/serve_chat_h100.sh            # needs $CHAT_HOME/app_token
# Reach it from the laptop with app/tunnel/tunnel.sh. Data stays in $CHAT_HOME.
# ============================================================================
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM node, x86 .venv incompatible; gx13v1: faulty GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append   # without append, a requeue truncates the log
#SBATCH --job-name=chat_server
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/hybrid_model_mamba_xlstm" 2>/dev/null || cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/hybrid_xmamba_h100}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
CHAT_HOME="${CHAT_HOME:-$HOME/chat_sessions}"
MODE="${MODE:-private}"
BIND="${BIND:-127.0.0.1}"                       # P1-B decides; 0.0.0.0 requires the token file
GALLERY="${GALLERY:-$(ls -d "${CHAT_HOME}"/gallery/*/ 2>/dev/null | sort | tail -1)}"
TOKEN_FILE="${CHAT_HOME}/app_token"
export HF_HOME="${SCRATCH_ROOT}/.hf" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "${CHAT_HOME}" && chmod 700 "${CHAT_HOME}"
if { [ "${MODE}" = "public" ] || [ "${BIND}" != "127.0.0.1" ]; } && [ ! -s "${TOKEN_FILE}" ]; then
  echo "ERROR: ${MODE} mode / bind ${BIND} needs a token in ${TOKEN_FILE} (R6)"; exit 1
fi
echo "=== chat server: node=$(hostname) mode=${MODE} bind=${BIND} gallery=${GALLERY:-none} job=${SLURM_JOB_ID:-?} ==="

# Free loopback port: compute nodes are shared, so a fixed 8001 can already be taken.
LABELER_PORT="${LABELER_PORT:-$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')}"
# The CheXbert environment of score_chexbert_h100.sh: its weights live in the DEFAULT HF cache, not
# ${SCRATCH_ROOT}/.hf, and it runs with HF_HUB_OFFLINE=0. Inheriting this job's settings would hide them.
env -u HF_HOME HF_HUB_OFFLINE="${CHEXBERT_HF_HUB_OFFLINE:-0}" PYTHONPATH=.chat_deps_chexbert \
  .venv_chexbert/bin/python -m uvicorn app.labeler:app --host 127.0.0.1 --port "${LABELER_PORT}" &   # loopback only
LABELER_PID=$!
source "${VENV_ACTIVATE}"
export PYTHONPATH=".chat_deps${PYTHONPATH:+:${PYTHONPATH}}"   # web deps overlay (P0-G); the shared venv is untouched
python -m app.server --engine real --device "${DEVICE:-cpu}" --mode "${MODE}" --home "${CHAT_HOME}" --host "${BIND}" --port 0 \
  --endpoint-file "${CHAT_HOME}/endpoint" --labeler "http://127.0.0.1:${LABELER_PORT}" \
  ${GALLERY:+--gallery "${GALLERY}"} --threads "${SLURM_CPUS_PER_TASK:-8}" \
  --drift-note "${DRIFT_NOTE:-}" $([ -s "${TOKEN_FILE}" ] && echo --token-file "${TOKEN_FILE}") &
API_PID=$!
trap 'kill -TERM ${API_PID} ${LABELER_PID} 2>/dev/null; wait' TERM INT
wait ${API_PID}
kill -TERM ${LABELER_PID} 2>/dev/null || true
```

`DRIFT_NOTE` carries the P1-D sentence. `scripts/serve_chat_gpu_h100.sh` is the same file except: `#SBATCH --gpus=1`, no `--qos` line (as every GPU wrapper in the repo), `--mem=48G`, job name `chat_server_gpu`, and `DEVICE=cuda`. Which of the two serves is P1-D's decision (U7). Tests: CLI argument parsing (a unit test of the parser and of the token-file reader); the endpoint file is written after start-up (tiny engine, real uvicorn); parity: both wrappers use the overlays and `env -u HF_HOME`, the GPU one has `--gpus=1` and `DEVICE=cuda`. Commit `"P7-B: server CLI + CPU/GPU serve jobs"`.

- [ ] **P7-C** `app/tunnel/tunnel.sh` (a forward loop that re-discovers the node) and `app/tunnel/public_demo.sh` (cloudflared, public mode only).

```bash
#!/bin/bash
# CHAT_UI_PLAN.md P7-C — keep http://localhost:${LOCAL_PORT:-8000} pointed at the chat server,
# wherever SLURM has (re)started it. Runs on the laptop. Ctrl-C to stop.
set -u
LOGIN="${LOGIN:-hpi-hpc}"; LOCAL_PORT="${LOCAL_PORT:-8000}"; VIA="${VIA:-jump}"   # jump (loopback bind, P1-B) | login
while true; do
  EP=$(ssh -o BatchMode=yes "${LOGIN}" cat chat_sessions/endpoint 2>/dev/null) || EP=""
  if [ -z "${EP}" ]; then echo "$(date +%T) no endpoint yet; is the job running? (ssh ${LOGIN} squeue --me)"; sleep 15; continue; fi
  NODE="${EP%%:*}"; PORT="${EP##*:}"
  echo "$(date +%T) forwarding localhost:${LOCAL_PORT} -> ${NODE}:${PORT} via ${VIA}"
  if [ "${VIA}" = "jump" ]; then
    # Compute-node host keys are not in ~/.ssh/known_hosts and BatchMode cannot prompt (P1-B: "Host key
    # verification failed"); accept-new into a separate file, reached only through the login node.
    ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new \
        -o UserKnownHostsFile="${HOME}/.ssh/known_hosts_hpi_nodes" -J "${LOGIN}" \
        -L "${LOCAL_PORT}:127.0.0.1:${PORT}" "${CLUSTER_USER:-krishankumar.bhushan}@${NODE}"
  else
    ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L "${LOCAL_PORT}:${NODE}:${PORT}" "${LOGIN}"
  fi
  sleep 15   # the job moved or died: re-read the endpoint
done
```

`public_demo.sh` refuses to run unless `curl -s localhost:8000/healthz` reports `"mode":"public"`, then runs `cloudflared tunnel --url http://localhost:8000` and prints a reminder to stop it after the demo. Parity tests: `tunnel.sh` reads `chat_sessions/endpoint` and supports both paths; `public_demo.sh` checks the mode first. Commit `"P7-C: tunnel loop + public demo wrapper"`.

- [ ] **P7-D** Real-turn gate from the laptop, and the five image features on real data (private); timings against the targets.

1. On lx01: `sbatch scripts/serve_chat_h100.sh`. 2. Laptop: `bash app/tunnel/tunnel.sh` (`VIA` from P1-B). 3. Open `http://localhost:8000`; pick test rows 0–4 with the picker. 4. For each, the live report must equal line *i* of the P2-E CPU golden `hyps.txt` (same node type) — record it; the card's published-vs-live line must match the P1-C result for that row. 5. Repeat the P6-F checklist on real data. 6. Record per-stage ms, first-byte time and total turn time against the targets (first byte ≤ 300 ms; total within the P1-D decision). Tick with job id, screenshots, timings.

*Added at P4-G (2026-10-09):* the UI now stops at the first repeated sentence by default.
- **The step 4 equality check** runs with "Stop when the report starts repeating" switched off, which is the published protocol. The same rule holds for every published-vs-live comparison (P5-E `live_equals_published`).
- **Measuring the stop condition.** A second pass over the same rows, with the switch on, measures it on the real decoders:
  - how often it fires at the 100-token default and at 200 tokens;
  - the tokens and time it saves;
  - that it never cuts a report before its first complete sentence.

  Record these beside the timings.

- [ ] **P7-E** Requeue drill: `scontrol requeue <job>` mid-turn → the turn is stored `error` with `server_restart`; the job comes back; the tunnel re-attaches unattended.

Record: the requeue time, the new node, the time until `tunnel.sh` re-attached, the stored status of the interrupted turn, and that the SQLite file opened cleanly on the new node (D22). If the database is reported locked after the move, record it and switch the DB to `journal_mode=DELETE` (rollback journal) as the documented fallback.

- [ ] **P7-F** Parity tests for the new wrappers: partition, account and qos, ga03 exclusion, no `--gres`, requeue and append, HF offline, loopback labeller, token required for a non-loopback bind or public mode.

```python
def test_serve_chat_wrapper_follows_the_cluster_invariants_and_r6():
    src = (REPO_ROOT / "scripts" / "serve_chat_h100.sh").read_text()
    directives = [l for l in src.splitlines() if l.startswith("#SBATCH")]
    for want in ("#SBATCH --partition=pot-hpi-aisc-batch", "#SBATCH --account=aisc", "#SBATCH --qos=aisc",
                 "#SBATCH --requeue"):
        assert want in directives, want
    assert any(l.startswith("#SBATCH --open-mode=append") for l in directives)
    assert any(l.startswith("#SBATCH --exclude=ga03") for l in directives)
    assert not [l for l in directives if "--gpus" in l or "--gres" in l]
    assert "HF_HUB_OFFLINE=1" in src and "--host 127.0.0.1 --port" in src
    assert "env -u HF_HOME" in src, "the labeller needs score_chexbert_h100.sh's HF cache, not the app's"
    assert "needs a token" in src and "--token-file" in src
```

Commit `"P7-F: parity tests for the chat wrappers"`.

### P8 — Hardening

Gate: the chaos checklist passes; the non-functional targets are measured.

- [ ] **P8-A** Request limits (413 above 21 MB before parsing), structured request-id logs, the error envelope everywhere, a per-client rate limit in public mode.

Middleware rejects `Content-Length` > 21 MB with 413 before the multipart parser runs. Every request logs one JSON line (`ts, request_id, method, path, status, ms, client`), and `X-Request-Id` is echoed on every response (a caller's own id is kept). All errors use `error_body`. Public mode: at most 6 turns per client per 10 minutes (token bucket) → 429 `rate_limited`.

```python
# tests/test_app_hardening.py
def test_oversize_body_is_413_before_parsing(client):
    sid = client.post("/v1/sessions", json={}).json()["id"]
    r = client.post("/v1/sessions/{}/messages".format(sid), content=b"x" * (21 * 1024 * 1024 + 1),
                    headers={"Content-Type": "multipart/form-data; boundary=zzz"})
    assert r.status_code == 413 and r.json()["error"]["type"] == "validation_error"


def test_request_id_is_echoed_and_generated(client):
    assert client.get("/healthz", headers={"X-Request-Id": "abc"}).headers["X-Request-Id"] == "abc"
    assert len(client.get("/healthz").headers["X-Request-Id"]) >= 16


def test_public_rate_limit_is_per_client(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path), mode="public", token="t")) as c:
        h = {"Authorization": "Bearer t", "X-Client-Id": "a"}
        sid = c.post("/v1/sessions", json={}, headers=h).json()["id"]
        codes = [c.post("/v1/sessions/{}/messages".format(sid), headers=h,
                        files={"image": ("x.png", png_bytes(), "image/png")},
                        data={"text": "", "options": json.dumps({"max_new_tokens": 16})}).status_code for _ in range(7)]
        assert codes[:6] == [200] * 6 and codes[6] == 429
        other = dict(h, **{"X-Client-Id": "b"})
        sid_b = c.post("/v1/sessions", json={}, headers=other).json()["id"]
        assert c.post("/v1/sessions/{}/messages".format(sid_b), headers=other,
                      files={"image": ("x.png", png_bytes(), "image/png")},
                      data={"text": "", "options": json.dumps({"max_new_tokens": 16})}).status_code == 200
```

Commit `"P8-A: limits, request ids, rate limit"`.

- [ ] **P8-B** Retention: a sweep at start-up and daily; `RETENTION_DAYS` per mode; the banner states it.

A daemon thread runs `store.sweep(days)` at start and every 24 h; `days` = `RETENTION_DAYS` or 45 (private, U6) / 7 (public). `/healthz` reports `retention_days`; the banner shows "Sessions are deleted after N days".

```python
def test_startup_sweep_removes_expired_sessions_and_health_reports_the_policy(tmp_path):
    from app.store import Store
    s = Store(tmp_path)
    old = s.create_session("public", "a")
    s._con.execute("UPDATE sessions SET created_at='2000-01-01T00:00:00+00:00' WHERE id=?", (old["id"],))
    s._con.commit()
    s.close()
    with TestClient(create_app(engine="tiny", home=str(tmp_path), mode="public", token="t")) as c:
        assert c.get("/healthz").json()["retention_days"] == 7
        h = {"Authorization": "Bearer t", "X-Client-Id": "a"}
        assert c.get("/v1/sessions/{}".format(old["id"]), headers=h).status_code == 404
```

Commit `"P8-B: retention sweep"`.

- [ ] **P8-C** Load drill: 6 concurrent turns → 4 accepted, 2 × 429; all 4 complete with contiguous `seq`.

First extract the P3-D `live` fixture's body into `tests/app_helpers.py`, so both files share it (the fixture then calls it):

```python
def start_live_server(app):
    """A real uvicorn on an ephemeral port (TestClient buffers whole responses). -> (base_url, stop)."""
    import socket
    import threading
    import time
    import uvicorn
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.02)

    def stop():
        server.should_exit = True
        t.join(5)

    return "http://127.0.0.1:{}".format(port), stop
```

```python
def test_six_concurrent_turns_four_accepted_two_refused(tmp_path):
    base, stop = start_live_server(create_app(engine="tiny", home=str(tmp_path), queue_cap=4,
                                              tiny_step_delay_s=0.02))
    try:
        sid = httpx.post(base + "/v1/sessions", json={}).json()["id"]
        url = base + "/v1/sessions/{}/messages".format(sid)
        results = []

        def one():
            with httpx.stream("POST", url, files={"image": ("x.png", png_bytes(), "image/png")},
                              data={"text": "", "options": json.dumps({"max_new_tokens": 60})}, timeout=60) as r:
                frames = list(iter_sse(r.iter_text())) if r.status_code == 200 else []
                results.append((r.status_code, frames))

        threads = [threading.Thread(target=one) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(120)
    finally:
        stop()
    assert sorted(code for code, _ in results) == [200, 200, 200, 200, 429, 429]
    for code, frames in results:
        if code == 200:
            assert frames[-1]["data"]["status"] == "done"
            assert [f["data"]["seq"] for f in frames] == list(range(1, len(frames) + 1))
``` The same drill runs from the laptop against the cluster server through the tunnel (`scripts/chat_load_drill.py URL TOKEN`, httpx only). Record both. Commit `"P8-C: load drill"`.

- [ ] **P8-D** Chaos checklist: API killed mid-turn, tunnel dropped mid-stream, oversize upload, wrong token, labeller down, gallery missing, disk quota error on upload.

Each item with its expected behaviour, run and recorded: kill → turn `error/server_restart` after restart; tunnel drop → polling completes the card; oversize → 413 and the preview stays; wrong token → 401 and the drawer asks for the token; labeller down → `label` skipped, turn `done`; gallery missing → `retrieve` skipped, turn `done`; `OSError` on upload write → `error` event with a readable message, no partial files. Commit `"P8-D: chaos checklist"`.

- [ ] **P8-E** `app/README.md` runbook (start, tunnel, public demo, stop, retention, what is stored, the DUA rules); allow-listed in `.gitignore`.

Commit `"P8-E: runbook"`.

- [ ] **P8-F** Non-functional targets measured: first byte ≤ 300 ms, full turn (P1-D target), 375 px, no external requests from the page, no `EventSource` in `static/`.

Record the numbers from P7-D and a laptop tiny run; the static checks already exist as tests (P4-A). Commit `"P8-F: targets measured"`.

### P9 — Demo and close

- [ ] **P9-A** `scripts/check_no_restricted_files.sh` and `scripts/install_hooks.sh` (pre-commit): refuse staged `outputs/`, `results/`, `uploads/`, `chat_sessions/`, `*.ckpt`, `*.parquet`, `*.npy`, `*.db`.

```bash
#!/bin/bash
# CHAT_UI_PLAN.md P9-A — refuse a commit that stages MIMIC-derived or binary artefacts (R1).
set -euo pipefail
bad=$(git diff --cached --name-only --diff-filter=ACMR | grep -E \
  '^(outputs|results|uploads|chat_sessions)/|\.(ckpt|parquet|npy|npz|db|pt)$' || true)
if [ -n "${bad}" ]; then
  echo "Refusing to commit restricted or binary artefacts:"; echo "${bad}"; exit 1
fi
```

Tests: a temporary git repo with a staged `x.npy` → exit 1; a staged `.py` → exit 0. Commit `"P9-A: restricted-file pre-commit check"`.

- [ ] **P9-B** (optional) GitHub Pages copy of `app/static/` with a configurable API base; the CORS allow-list gains the Pages origin.

Only if the user wants it. The drawer already has "API base URL"; `create_app(cors_origins=[...])` adds the origin. Commit `"P9-B: Pages build"`.

- [ ] **P9-C** Supervisor demo: five test studies (two normal, two with support devices, one pneumothorax) chosen from the published labels; generated vs reference vs floor; the session exported (it stays on the cluster).

`scripts/chat_demo_cases.py` (CPU job) lists candidate test rows by `y_true` in the published `chexbert_labels.json`. Run the five through the UI in private mode; export the session as JSON and Markdown into `CHAT_HOME/exports/`; record the rows and the export paths.

- [ ] **P9-D** (deferred: a later stage, only on the user's word, U3) Public quick-tunnel smoke: POST streaming through trycloudflare measured; the tunnel stopped afterwards.

Measure first-byte and frame gaps through the quick tunnel with `probe_client.py`-style timing against the real server; confirm the polling fallback completes a turn when streaming stalls. Stop `cloudflared` and record that it was stopped.

- [ ] **P9-E** Close: state COMPLETE, final notes, CLAUDE.md paragraph updated.

- [ ] **P9-F** (optional, U5 "a mix between the two") Compare mode: one image decoded by both the Mamba-3 and the 13D decoders, the two reports and label sets side by side in one card.

Only after P9-E's other boxes, and only if the user still wants it. It reuses `Pipeline` with two engines in one turn (`options.compare: true`, private and public alike), two `content_block`s (index 0 and 1), and one `label` stage that labels both. Tests on the tiny stack with two tiny engines.

### P9-G: a learned stop condition. Approved 2026-10-09; scheduled right after P4-G.

**The user's approval** (2026-10-09, verbatim): "yes approve this but make sure that the response is stopped in correct manner and not at any point the result that i want to get is the similar images sluster and also the report generation . so it needs these two things and then it should stop ."

What it means for the plan:
- A turn must deliver two things, then end cleanly:
  - the similar X-rays from the gallery (P5-B/D/E, P6-C);
  - a generated report that ends at a natural end of its own, not at an arbitrary point.
- P9-G is no longer optional, and it runs before P5. Its single H100 training job (about 1.3 h, the same as the published Mamba-3 run, per `analysis/ARCHIVE_MANIFEST.md`) then trains while P5's laptop work goes on.

**Why P4-G is not enough.**
- P4-G's decode-time stop only notices a loop after it has begun.
- The trainer sets `pad_token = eos_token` (`scripts/train_report_generation.py:223`).
- The training step then masks every `attention_mask == 0` position (`hybrid_xmamba/training/lightning_module.py:1634-1640`, on `ImageTextDataset`'s right-padded tokens in `scripts/train_contrastive.py`).
- So no end-of-report token is ever a target, and the model cannot learn to stop (V5-D).

- [x] **P9-G1** (laptop) An EOS target behind a new flag. The flag is `dataset.report_eos_target`, default `false`.
  - When it is on, each report that fits in `max_length` ends with exactly one EOS whose `attention_mask` is 1, so it is supervised. The padding after it stays masked.
  - A report cut by `max_length` gets no EOS, because the budget, not the report, ended it.
  - Off is byte-identical to today: the same `input_ids` and `attention_mask` for every row.
  - Tests:
    - a unit test on the dataset and collate output, for both settings;
    - a parity test that the flag-off batch is unchanged;
    - one `test_willi_parity.py` assertion that every published `_rrg` yaml leaves the flag off.

  *Rulings (controller, 2026-10-09):*
  - **R3 exception.** For P9-G the user approved an R3 exception: thesis code may change additively, behind the flag.
  - **Where the flag is read.** `ImageTextDataset` (`scripts/train_contrastive.py`; the report-gen trainer reuses it through `load_mimic_cxr`) reads `cfg.dataset.get("report_eos_target", False)` once, in `__init__`.
    - It is not a constructor argument, so the shared loaders stay untouched.
    - A non-boolean value raises.
    - No yaml declares the key. Hydra's struct mode therefore needs `+dataset.report_eos_target=true` on the command line.
  - **Flag on.** Tokenise the report without padding. If it is at most `max_length - 1` tokens, append the EOS (`attention_mask` 1) and pad with `attention_mask` 0. Otherwise keep today's encoding exactly: empty, exact fit, or cut by `max_length`.
  - **Where the EOS becomes a target.** `ReportGenerationLightningModule._step` masks only `attention_mask == 0` (`hybrid_xmamba/training/lightning_module.py:1634-1640`), so the appended EOS is a supervised target.
  - **Tests.**
    - `tests/test_report_eos_target.py`: synthetic text, and flag-off byte parity for short, exact-fit and over-long texts.
    - `tests/test_willi_parity.py`: no yaml under `configs/` sets the flag.

  *As built (3a15159; reviewed clean, no fix round):*
  - **Code.** `ImageTextDataset._eos_row` builds the flag-on row. The flag-off path is the original tokenizer call, verbatim. A non-boolean flag, or a missing eos/pad id with the flag on, raises.
  - **Tests.** `tests/test_report_eos_target.py` has 41 tests, run with a fake tokenizer and with the real GPT-2. They include `_step`'s masking end to end.
  - **The review's RED check.** With `_eos_row` disabled, the 12 tests that need an EOS fail.
  - **Parked for the final review:**
    - the class docstring says "fits in max_length" where the rule is "fits with room for one more token";
    - the `+` test matches Hydra's error text.
  - **A note for P9-G3 and G4.** The flag also gives the validation split an EOS target, so the EOS run's `val/lm_loss` is not exactly comparable with the published run's.
- [x] **P9-G2** (laptop) EOS-stop decoders and the engine.
  - `beam_search_decode_eos` and a cached twin live beside the published decoders, which stay byte-identical (R3).
  - A beam that emits EOS is finished and set aside. Decoding ends when the best `beam_size` candidates are all finished, or at the budget.
  - The finished score uses the same length penalty as the published decoders.
  - Parity tests:
    - with EOS never emitted, the output equals the published decoders token for token, cached and uncached;
    - a scripted model that emits EOS stops there.
  - The engine:
    - uses them when the model card says `eos_trained: true`;
    - sets `generate.detail.stopped` to `"eos"`;
    - P4-G's repeat stop stays as a backstop.
  - The card shows no budget note on an EOS stop.

  *Rulings (controller, 2026-10-09):*
  - **Placement.** New functions sit beside the published ones, which stay byte-identical. The EOS token id is `eos_token_id=50256` (GPT-2; also the pad id).
    - `beam_search_decode_eos` sits next to `beam_search_decode` (`scripts/evaluate_report_generation.py`).
    - `HybridLanguageModel.beam_search_cached_eos` sits next to `beam_search_cached` (`hybrid_xmamba/models/hybrid_lm.py`). The live beams stay in the cache's batch axis.
  - **Early stopping, as in Hugging Face.** These steps repeat at every decoding step:
    1. Rank all `beam × vocab` candidates exactly as the published code does. Take the top `2 * beam`.
    2. Walk them in rank order:
       - an EOS candidate ranked within the first `beam` goes to the finished pool, with its normalised score (the EOS counts in the length);
       - non-EOS candidates fill the `beam` live slots.
    3. Stop at `beam` finished hypotheses, or at the budget.
  - **Output.** Return the best of finished ∪ live, without the trailing EOS, together with whether EOS ended it.
  - **Hard gate.** When EOS never ranks in the top `2 * beam`, both functions equal the published decoders token for token: beam 1 and beam 3, cached and uncached. A scripted model that emits EOS stops there.
  - **Evaluation flag.** `evaluate_report_generation.py --stop-at-eos` (default off) selects these functions for P9-G4.
  - **Engine.**
    - `RealEngine` reads `eos_trained` from `run_metadata.json` (`resolved_config.dataset.report_eos_target`) and puts it on the card.
    - `Engine.generate` uses the EOS decoders for such a model and reports `stopped: "eos"`, with no card note.
    - `stop_on_repeat` stays as a backstop.
    - The tiny engine stays `eos_trained: false`. A scripted tiny decoder drives the `"eos"` path end to end.

  *Amended (controller, 2026-10-09, fix rounds 0 and 1). These rules replace the Hugging Face-style stop above.*
  - **Why they changed.** The HF rule can put "budget" on the card with fewer tokens than the budget. It can also drop a natural ending when the repeat backstop fires, and the stream can run past the report's end.
  - **Stopping, OpenNMT-style.** After each step's walk:
    - the *answer* is the best of finished ∪ live, by normalised score, with ties going to finished;
    - `on_step` receives the answer;
    - decoding stops right after the first step whose answer is finished;
    - at the budget, the full-length live answer is returned with `ended_by_eos` false.
    So the stop label is always truthful, and the stream's last frame is the report.
  - **Candidates.** They are built published-first: the published `topk(beam)` call, then the extras from the wider `topk(2*beam)`. That way exact ties break as in the published decoders. The cached twin's frame is `tokens[argmax(scores)]`, as published.
  - **No EOS as the first generated token**, as with HF's `min_new_tokens=1`. A turn never ends with an empty report.
  - **The soft stop.** `StopDecoding` is defined in `hybrid_xmamba/models/hybrid_lm.py`; the engine's `_RepeatStop` subclasses it.
    - Raised from `on_step`, it returns the best finished hypothesis if one exists; otherwise it propagates.
    - So a natural ending survives the repeat backstop, with `stopped: "eos"`.
    - Every other exception, `Cancelled` included, propagates.
  - **`--stop-at-eos`** works with `--checkpoint --decode beam` only. With `--retrieval-baseline`, `--smoke-test` or hyp/ref mode it fails, instead of being silently ignored. The script prints `EOS stop: k/n reports ended at the end-of-report token, m were cut at max_new_tokens=B`.

  *As built (1fcae9c, f417165, cc96a6a, 6136cf1; reviewed clean after fix rounds 0 and 1):*
  - **Tests:**
    - `tests/test_beam_search_eos.py` (231) covers:
      - parity on tiny and real weights;
      - 54 exact-tie runs comparing ids and the stream;
      - an independent oracle;
      - scripted EOS cases;
      - the soft stop;
      - step 0;
      - the parser.
    - Also engine (120), API, OpenAPI and node tests.
  - **Mutation checks.** 27 mutants in fix 0 and 15 in fix 1, all killed.
  - **The re-review's stress run.** 1,152 runs with quantised logits (beams 1–8, cached and uncached) gave 0 mismatches against the published decoders in ids or stream.
  - **R3.** The published decoders are byte-identical; the AST was compared with 7a21925.
  - **Parked for the final review:**
    - no test pairs a non-empty prompt with an EOS at step 0 (the code is correct, and the engine never passes a prompt);
    - the `[1]` variant of the argmax-frame test cannot discriminate;
    - the `[:2*beam]` cap is inert;
    - three wording nits.
- [ ] **P9-G3** (cluster) The training job.
  - The recipe is the published `h100_report_gen_m3_tower13d_s42` (`hybrid_150m_m3_rrg`, the same tower, Mamba-3 backbone, data and seed 42), with only `dataset.report_eos_target=true`.
  - It writes to a new directory outside `MAIN_REPO` (R8): `CHAT_HOME/models/report_gen_m3_eos_s42/`. Checkpoints are MIMIC-derived (Class R), so they stay on the cluster.
  - Before submitting, record a prediction (R4): wall time ≈ the published 1.3 h, and a final val loss close to the published run's, because the target gains one token per report.
  - Read the job's results only through `scripts/chat_remote.sh summary` (R7).

  *Rulings (controller, 2026-10-09):*
  - **The recipe.** Copy it from the published run's own `run_metadata.json` `resolved_config`: config fields and paths only, no MIMIC content (R7). Change nothing except the override `+dataset.report_eos_target=true`.
  - **The wrapper.** It is a new `scripts/train_report_eos_h100.sh`. It follows the SLURM invariants:
    - H100 via `--gpus=N`, never `--gres`;
    - `--requeue` and `--open-mode=append`;
    - `HF_HUB_OFFLINE=1`.
  - **Its output** goes under `CHAT_HOME/models/report_gen_m3_eos_s42/`, never `MAIN_REPO/outputs`.
  - **Tests.** Parity tests for the wrapper go in `tests/test_willi_parity.py`.

  *As built (22fa8c2, 0bb201d; wrapper review approved, minors fixed in round 1):*
  - **The recipe.** `scripts/train_report_eos_h100.sh` holds it as plain assignments, so a stray env var cannot change it. One `OVERRIDES` array feeds both the preflight and the trainer. It equals the published wrapper's list, with the V3-chain decoder values: `hybrid_150m_m3_rrg`, the Stage-0 m3 checkpoint, the 13D tower, 4 GPUs, 12,000 steps, seed 42, `save_top_k=0`, `aux_lambda=0`, `prefix_k=32`. The only additions:
    - `experiment_name=report_gen_m3_eos_s42`;
    - `output_dir=${OUT_DIR}`;
    - `+dataset.report_eos_target=true`;
    - `hydra.run.dir=${OUT_DIR}/hydra`. Without it, Hydra writes a dated dir into `./outputs`, which is the thesis checkout.
  - **SLURM.** `--gpus=4`, `--time=04:00:00`, `--requeue`, `--open-mode=append`, job name `chat_report_eos`.
  - **The job log (R7).** It carries only `===`, `RESULT` and `ERROR` lines. Its first line is `=== sync <sha> <clean|dirty> ===`, read from `.sync_stamp`. The trainer's output goes to `OUT_DIR/train.log`; the result script's stderr goes to `result.err`.
  - **The preflight** (`scripts/report_eos_preflight.py`). It runs on the compute node before training and stops the job on either of two checks:
    - **(a) Code version.** `ImageTextDataset` must be imported through the trainer's own route (`load_mimic_cxr.__globals__`), have `_eos_row`, and resolve inside the cwd tree (`CLUSTER_REPO`), not the thesis checkout.
    - **(b) Recipe.** Hydra compose of the same overrides is compared with the published run's `resolved_config`.
      - Changed keys are allowed only when the new roots explain them. On the laptop configs that gives six: `checkpoint_dir`, `experiment_name`, `log_dir`, `output_dir`, `trainer.default_root_dir`, `wandb.name`.
      - The only added key allowed is `dataset.report_eos_target`. Nothing may be removed.
      - Unchanged roots fail, because they would overwrite the published run.
  - **DONE.** `OUT_DIR/DONE` is written only when:
    - training exited 0;
    - no `interrupt.ckpt` is newer than this attempt's start;
    - `last.ckpt` exists and is newer than that start;
    - `steps == 12000`. `steps` is the largest TensorBoard stamp + 1, because Lightning stamps from zero; a test pins this against the real module.

    An existing DONE refuses a rerun (R8).
  - **The result line.** `scripts/report_eos_result.py` prints one RESULT with `steps`, `wall_s`, this run's last `val/lm_loss`, the published run's last `val/lm_loss` and `ckpt_exists`.
  - **Tests.** `tests/test_report_eos_preflight.py`, `tests/test_report_eos_job.py` and 8 parity tests. Together they hold 110 tests, including a bash 3.2 sandbox rehearsal of the real wrapper.

  **Prediction (R4), recorded before submission:**
  - **Preflight.** `RESULT {"preflight":"code","eos_flag":true,"module_in_cwd":true}`. The recipe check reports `ok:true`, with exactly the six root-derived keys as `changed`, `added = [dataset.report_eos_target]` and nothing removed. Confidence 80%. The likeliest failure is a launch value that differs from the transcription; it shows as a named key before any training.
  - **Wall time.** 1.1–1.8 h for 12,000 steps on 4 H100. The published run took about 1.3 h.
  - **Final `val/lm_loss`.** Within 0.03 of the published run's last value, slightly higher more likely than lower: one more target per report.
  - **DONE.** Written, with `steps:12000`.
- [ ] **P9-G4** (cluster) The evaluation job, on the official test split (n = 2,663).
  - The EOS model is decoded with the EOS-stop beam search (beam 3, budget 200, so the model, not the budget, ends the report).
  - The baseline is the published Mamba-3 run, decoded with the published protocol.
  - Report:
    - ROUGE-L, BLEU-1/4, CheXbert-14 micro/macro and exact match, with paired-bootstrap 95% CIs (`bootstrap_compare`);
    - mean length;
    - the share of reports that end by EOS, against those cut at the budget;
    - repeated sentences per report.
  - **Gate:** no metric significantly worse than the published run.
  - Prediction: the cut-at-budget share falls from 301/400 (V5-D) to under 10%.

  *Rulings (controller, 2026-10-09):*
  - **Three jobs, chained with `--dependency=afterok`.** The controller submits them; the implementer only writes them.
  - **(1) The decode job.** A new GPU wrapper, `scripts/eval_report_eos_h100.sh`. It runs `evaluate_report_generation.py` with `--cached-decode --stop-at-eos --decode beam --beam-size 3 --max-new-tokens 200 --prefix-k 32` on `test.parquet`, with `--num-samples 999999`.
    - Input: the EOS checkpoint under `CHAT_HOME/models/report_gen_m3_eos_s42/`.
    - Output: `results/chat_report_eos_test_split_s42`.
    - It refuses without the training's `DONE`, or if the dump already exists (R8).
    - The raw output goes to `eval.log` in the dump dir. The job log carries only `===`, `RESULT` and `ERROR` lines, including the sync line.
  - **(2) CheXbert scoring.** The thesis wrapper `score_chexbert_h100.sh` is used unchanged, but only if every line of its log that `SUMMARY_PATTERN` matches is free of text. Otherwise a thin chat wrapper takes its place.
  - **(3) The comparison.** A new CPU wrapper, `scripts/eval_report_eos_compare_h100.sh`.
    - It runs `bootstrap_compare.py` unchanged: EOS dump against `results/report_gen_m3_test_split_s42`, with per-label CIs.
    - It also runs a new `scripts/report_eos_stats.py` on both dumps' hyps. That reports mean length, empty reports, repeated sentences per report (`repair_generations`, `dedup="all"`) and the share with an unterminated last sentence.
    - It prints RESULT lines, numbers only, ending with `RESULT {"gate":"pass"|"fail","worse":[…]}`. "worse" lists every metric whose 95% CI of (EOS − published) lies entirely below 0.
  - **Tests.** Parity tests, unit tests and a sandbox rehearsal, as in P9-G3.

  *As built, laptop part (070390d, d8857d7, 8618316; review approved, fix rounds 1 and 2 clean):*
  - **Job 1 (decode).** `scripts/eval_report_eos_h100.sh`.
  - **Job 2 (CheXbert).** `scripts/eval_report_eos_chexbert_h100.sh`, a thin wrapper.
    - Why: the thesis `score_chexbert_h100.sh` prints a `===` line with both file paths and no RESULT, so it fails R7.
    - It runs the same scorer, with the same arguments, venv and D24 environment.
  - **Job 3 (compare).** `scripts/eval_report_eos_compare_h100.sh`. It runs `bootstrap_compare.py` unchanged, with the thesis wrapper's 1000 resamples, seed 0 and per-label rows.
  - **`scripts/report_eos_stats.py`** has four subcommands: `decode`, `chexbert`, `hyps` and `gate`.
    - Its RESULT lines carry only allowlisted names (wrapper constants, `bootstrap_compare`'s 9 metrics, the 14 CheXbert labels) and numbers with |x| < 10^7.
    - Its ERROR lines are literals plus parsed digit counts.
  - **Fail-early guards:**
    - job 1 refuses without the training's DONE, an existing dump, a `results/chat_*` path, or a visible GPU;
    - an up-front check that the published `refs.txt` exists;
    - a byte compare of the two `refs.txt` after the decode, which stops the chain before CheXbert.
  - **The gate RESULT** lists `worse`, the main-table metrics whose CI upper bound is below 0. An informational `label_worse` lists the per-label rows that are worse, but does not gate.
  - **Tests:** `tests/test_report_eos_eval.py` (212) and 14 parity pins. The gate parser round-trips the real `bootstrap_compare.render`; the reviewer fuzzed 6,000 reports with 0 disagreements.
  - **Submit lines** (header of the compare wrapper), chained with `afterok`. Job 1 runs with `-- --time=04:00:00`.

  **Prediction (R4), recorded before submission:**
  - **Decode.** `ended_by_eos` is at least 90% of 2,663, so `share_cut` < 0.10, against 301/400 cut mid-sentence at 100 tokens for the published run (V5-D).
  - **Length.** EOS reports average 40–80 words. Repeated sentences per report fall to near 0.
  - **Gate.** Pass, meaning no main metric significantly worse: 65%. ROUGE-L may even rise, because the published reports are cut at 100 tokens. The most likely "worse" is BLEU-4 or exact match, if the EOS model writes shorter reports.
  - **Wall time.** Decode 0.8–1.5 h (cached, beam 3), CheXbert about 30 min, compare under 30 min.
- [ ] **P9-G5** (laptop, then cluster) The EOS model in the UI.
  - The model appears in `MODEL_CHECKPOINTS` as a selectable model. Its card names the EOS training and links the G4 numbers.
  - The published checkpoints and numbers are unchanged.
  - Making it the *default* model is the user's decision, once they have seen the G4 numbers.
  - Verify a real turn on the cluster: it shows similar X-rays (once P5-E and P6-C are done) and a report that ends with `stopped: "eos"`.
  - *Carried from P9-G2 (controller, 2026-10-09):*
    - **UI copy for an `eos_trained` model.** The `full budget` chip in `render.js`, and the stop-switch hint at `app.js:1360` ("Off: … always decodes the whole token budget"), are wrong for such a model. With the switch off, it still ends at its EOS.
    - **The golden driver's card line** should print `eos_trained`.
    - **An optional final engine snapshot.** In two cases the stream's last frame is not the report:
      - a soft stop that returns a finished report while the answer was live;
      - a late win.

      `message_stop.display_report` settles the card in both.

---

## 10. Estimates and gates

| Phase | Work (part-time days) | Cluster | Gate |
|---|---|---|---|
| P0 | 0.5 (done 2026-10-01, pending approval) | — | user approval |
| P1 | 0.5 + queue | 2 CPU jobs | decisions recorded with job ids |
| P2 | 2 | 2 jobs (CPU, ~0.2 H100-h) | engine == script same node; GPU uncached == published |
| P3 | 2 | — | API tests green |
| P4 | 3 | — | browser turn, replay, exports |
| P5 | 3 | GPU ~1.5 h + ~0.7 h, CPU 1 job | R@k equal; 0 label mismatches; 50/50 self-retrieval |
| P6 | 3 | — | five features, tiny |
| P7 | 1 + queue | serve job | real turn through the tunnel; requeue drill |
| P8 | 1.5 | — | chaos checklist |
| P9 | 1 | 1 CPU job | demo exported |
| **Total** | **≈ 17.5** | **≈ 2.5 H100-hours once; CPU for serving** | |

The first usable local demo (P2–P4 on the tiny engine) lands after about a week; the image features on the tiny stack after about two.

## 11. Risks

| Risk | What it breaks | Early warning | Mitigation |
|---|---|---|---|
| A MIMIC-derived byte leaves the cluster | the DUA | any public payload with a private field | R1 module, field and catch-all tests, P9-A hook, no laptop copy of weights or gallery |
| Login-to-compute forwarding blocked, or long-running services disallowed on the partition | the whole transport | P1-B | measured first; stop and ask if both paths fail |
| CPU decode too slow, or nondeterministic | the 8 s target, R2 | P1-C | pre-registered decision tree (P1-D); GPU serving only with the user's consent |
| CPU-vs-GPU drift misread as a bug, or hidden | trust in the numbers | P1-C, P7-D | same-device identity is the hard gate; drift shown on every card |
| SQLite on NFS (locking, latency) | the store | P1-B commit timing, P7-E | exclusive-lock WAL; rollback-journal fallback documented |
| Towers not identical | one-pass retrieval, memory | P5-B manifest | load the 13D tower as a second module (+~350 MB, +~0.5 s per turn) |
| f1chexbert slow or not on GPU in its venv | P5-C time | P5-C canary | projection exits early; split the job by group ranges |
| Preemption mid-demo | turns in flight | `sacct` REQUEUED | requeue, endpoint file, tunnel loop; turns stored as `error` |
| Home quota (200 GiB; 115 used, 86 free on 2026-10-01) | gallery, uploads, overlays | `df -h /sc/home` before P5-B | gallery ≈ 1 GB (no thumbnails, D3); overlays < 100 MB; retention sweep |
| An agent reads MIMIC text (a raw log, a dump) | the DUA (data sent off-cluster) | any `cat`/`tail` of a dump or raw log | R7: wrappers print summaries only; `chat_remote.sh summary` greps a fixed pattern list and masks 8-digit ids |
| A sync or setup step overwrites or deletes existing cluster files | the thesis checkout, shared venvs | — | R8: separate `CLUSTER_REPO`, rsync without `--delete`, symlink only where nothing exists, `pip --target` overlays |
| FastAPI or pydantic on Python 3.14 | laptop dev loop | P2-A install | pin a working version in `app/requirements.txt` |
| Users read the report as clinical | harm | — | permanent banner, disclaimer in `message_stop`, no diagnosis wording |
| The 100-token budget truncates mid-sentence | trust in the demo | `truncated_mid_sentence` on the card | stated on every card; repair is display-only (V5-D) |

## 12. Open questions

None. The seven questions of 2026-10-01 were answered by the user the same day; the answers are §3a (U1–U9).
