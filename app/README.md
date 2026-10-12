# Chat server runbook

The chat app turns a chest X-ray into a report. It shows the report as it is written, the most similar training X-rays, the
matching training reports and CheXbert labels. Sessions are stored, replayed and exported.

The app runs as one SLURM job on the cluster. You reach it from the laptop through an SSH tunnel. The plan of record is
`CHAT_UI_PLAN.md`, and the user's design is `docs/chat_ui/CHAT_UI_SPEC.md`.

## 1. The data rules (MIMIC-CXR DUA)

- Everything MIMIC-derived stays on the cluster, in the server's home: `CHAT_HOME`, by default `/sc/home/$USER/chat_sessions/`.
  - That covers the weights, the gallery, retrieved reports, reference text, test-split images, the session database, uploads and logs.
  - Never copy any of it to the laptop or to a synced folder (iCloud, Dropbox).
- Never commit `chat_sessions/`, `outputs/` or `results/`.
  - `.gitignore` guards them.
  - `bash scripts/install_hooks.sh` installs a pre-commit hook (`scripts/check_no_restricted_files.sh`) that refuses them, along with `*.ckpt`, `*.npy`, `*.parquet` and `*.db`.
- **Private mode** (the default) shows MIMIC-derived content to whoever holds the tunnel, which is only you.
- **Public mode** shows nothing MIMIC-derived (`app/redact.py`):
  - retrieval is rank and similarity only;
  - there is no reference text and no score stage;
  - checkpoint paths are cut to file names;
  - error text the server did not write is replaced.
- An **export from a private server** saves MIMIC-derived text where your browser downloads files. Keep it out of synced folders and out of git, and delete it when you are done.
- **Online AI services.** PhysioNet counts sending credentialed data to an online AI service as sharing it. Never paste a report, a study id or an image into one.
- **Cluster logs.** Read them only through `bash scripts/chat_remote.sh summary <log>`.
  - The summary shows fixed line shapes and masks long digit runs.
  - The job log carries no report text by design. The API's own stderr goes to `${CHAT_HOME}/logs/` on the cluster.

## 2. Start the server (cluster)

From the laptop, in this repository:

```bash
bash scripts/chat_remote.sh sync                                      # rsync the tree to the cluster checkout (never --delete)
bash scripts/chat_remote.sh submit scripts/serve_chat_h100.sh MODELS=m3eos,m3 \
    DRIFT_NOTE='CPU decode; matched the published GPU decode on 20/20 probe studies'
bash scripts/chat_remote.sh queue                                     # wait for state R
bash scripts/chat_remote.sh summary logs/chat_server_<jobid>.log      # the start-up lines below
```

The job runs on CPU with 8 cores and 32 GB, for up to 24 h. It is started with `--requeue`. It runs two processes:
- the CheXbert labeller, in its own venv;
- the API, `python -m app.server`.

`scripts/serve_chat_gpu_h100.sh` is the same job on one GPU.

A normal start-up summary looks like this:

```
=== sync <commit> clean ===
=== chat server: node=<node> mode=private bind=127.0.0.1 device=cpu gallery=g13d_m3_v1 job=<id> ===
=== labeller up ===
[server] starting: engine=real device=cpu mode=private threads=8 models=hybrid_150m_m3_rrg_eos,hybrid_150m_m3_rrg
[server] sqlite journal_mode=wal
[server] gallery: 191462 images, 194125 report rows, labels done
[server] retention: purged 0 session(s) (private 45 d, public 7 d)
[server] serving: retrieval=on labels=on published=on
```

**Levers.** Pass them as `NAME=value` after the wrapper.

| Lever | Meaning |
|---|---|
| `MODELS` | The models served; the first one is the default. Aliases:<br>- `m3`: `hybrid_150m_m3_rrg`, the published model;<br>- `13d`: `hybrid_150m_v2_rrg`;<br>- `m3eos`: `hybrid_150m_m3_rrg_eos`, trained to end its reports.<br>The default is `m3`. `MODELS=m3eos,m3` makes the EOS model the default and keeps the published model selectable. |
| `DRIFT_NOTE` | The CPU-versus-GPU drift sentence. It is shown on the published models' cards. The EOS model's card says that its CPU decode was not probed against a GPU decode. |
| `MODE` | `private` (the default) or `public`. Public mode needs `${CHAT_HOME}/app_token`: a file of mode 0600 that you own, which is never printed. |
| `BIND` | The bind address; the default is `127.0.0.1`. Any other address needs the token file. |
| `GALLERY` | A gallery build directory. The default is `${CHAT_HOME}/gallery/g13d_m3_v1`; `none` serves without retrieval. |
| `RETENTION_DAYS` | Days a session is kept: a whole number from 1 to 3650. See section 7. |
| `DEVICE` | `cpu` (the default) or `cuda`. |
| `CHAT_HOME`, `LABELER_WAIT_S`, `PUBLISHED_MODEL`, `PUBLISHED_FLOOR` | See the header of `scripts/serve_chat_h100.sh`. |

**Refusals.** The EOS model needs its finished run in `CHAT_HOME`: the checkpoint, the `DONE` marker, and run metadata saying it was trained to end its reports. If any of these is missing, the API refuses to start with one ERROR line.

**Retention at start.** The first start on a `CHAT_HOME` runs the retention sweep (section 7). It deletes:
- public sessions older than 7 days;
- private sessions older than 45 days;
- deleted sessions;
- upload directories that no session owns.

## 3. Reach it from the laptop (tunnel)

```bash
bash app/tunnel/tunnel.sh                                          # then open http://localhost:8000
LOCAL_PORT=8010 bash app/tunnel/tunnel.sh                          # when a dev server already uses 8000
nohup env LOCAL_PORT=8010 bash app/tunnel/tunnel.sh > ~/chat_tunnel.log 2>&1 &   # in the background
curl -s localhost:8010/healthz                                     # status, mode, default_model, code_version, retention_days
```

How the tunnel works:
- The job writes where it serves to `chat_sessions/endpoint` once the API is up. The tunnel reads that file over ssh and forwards `localhost:LOCAL_PORT` to it.
- While a forward runs, the tunnel reads the file again every `RECHECK_S` seconds (default 60). If the server moved (a requeue onto another node or port) or stopped, the forward is stopped and a new one starts when a server is back. No hand-work is needed.
- `VIA=jump` (the default) forwards to the node's loopback through the login node. A loopback server needs no token.
- `VIA=login` needs a server bound to another address, and so a token.
- Other settings: `LOGIN`, `CLUSTER_USER`, `RETRY_SLEEP` (default 15), `MAX_ATTEMPTS` (0 = never stop).
- Exit codes: 1 if the local port is in use, 2 for a bad setting.
- On macOS, a listener bound to a wildcard address can make a used port look free.

## 4. Use it

Open `http://localhost:<LOCAL_PORT>`.

- **Start a turn.** Attach an X-ray by click, drag and drop, or paste. On a private server you can instead pick a test-split study. The stages stream in order: preprocess → encode → retrieve → generate → label → score.
- **What a card shows.** The report, the similar X-rays and matching reports, and the labels. Click an image to open the full-screen viewer, with side by side, zoom, pan, brightness and contrast.
- **The drawer** holds the settings: model, beam, token budget, the number of similar X-rays and reports, and "Stop when the report starts repeating".
- **Text commands** change the options of the next turns: `beam N`, `greedy`, `tokens N`, `retrieve N`, `reference: <text>`, `repair on|off`.
- **The two models.**
  - The EOS model ends its report where it was trained to.
  - The published model decodes its whole token budget, which is the published protocol. A test study's score line compares it with the published dump.
- **The laptop dev server** runs the tiny engine:
  ```bash
  venv/bin/python -c "import uvicorn; from app.server import create_app; uvicorn.run(create_app(home='/tmp/cxrchat_dev', tiny_step_delay_s=0.05, tiny_gallery=True), host='127.0.0.1', port=8000, log_level='warning')"
  ```
  The tiny engine has random weights and a synthetic gallery, so its text is never a report. Its cards say so, and the banner shows a "test model" badge. Real reports come only from the cluster server.

## 5. Public demo (only on the user's word)

1. Start the server with `MODE=public` (this needs the token file), then start the tunnel.
2. Run `LOCAL_PORT=<port> bash app/tunnel/public_demo.sh`.
   - It refuses unless the server's `/healthz` says `"mode": "public"`.
   - It opens a cloudflared quick tunnel and checks the mode again every `DEMO_RECHECK_S` seconds (default 10).
   - It closes the link and exits 1 if the server stops saying public, or does not answer within 5 s.
3. Anyone with the link reaches the server until you stop the script. Stop it as soon as the demo ends (Ctrl-C). Afterwards, run `pgrep cloudflared`: a `public_demo.sh` killed with SIGKILL leaves cloudflared running.

Public mode adds these limits:
- **Rate limit.** 6 turns per client per 10 minutes. The 429 says when to try again (`Retry-After`). A text-only note that runs no model spends nothing.
- **Retention.** 7 days for public sessions.

## 6. Stop, restart, requeue

- **Stop the server:** `ssh hpi-hpc scancel <jobid>`. The API removes the endpoint file first, so the tunnel sees it at once. It then ends a running turn as an error with `server_restart`, and exits 0.
- **Requeue drill:** `ssh hpi-hpc scontrol requeue <jobid>`. The job comes back, possibly on another node, and appends to the same log. The cut-off turn stays an error. The tunnel re-attaches without help.
- **Stop the tunnel:** Ctrl-C, or kill its process. With no server, it waits and retries.

## 7. Retention

- A session is deleted `RETENTION_DAYS` days after it was **created**: 45 by default on a private server and 7 on a public one.
- The sweep runs at start-up and then every 24 hours.
- Each sweep uses the server's own days for its own mode and the default for the other mode. So a public demo's sessions are gone after 7 days, whatever the server runs now.
- A session with a running turn waits for the next sweep.
- Deleting a session hides it at once and removes its uploaded images at once. The sweep purges its rows later.

## 8. What is stored (all on the cluster, under `CHAT_HOME`)

| Path | What |
|---|---|
| `chat.db` | SQLite (WAL, exclusive locking, which is safe on NFS). It holds sessions, messages and every streamed event, in the order they were sent. The event log is the source of truth for replay, resume and export. |
| `uploads/<session>/<sha256>/` | An uploaded image and its model input, stored per session by content: turns of a session that send the same bytes share them. Removed when the session is deleted. |
| `logs/` | `server_<jobid>.log` (the API's stderr: the uvicorn log and tracebacks) and `labeler_<jobid>.log`. |
| `endpoint` | One JSON line, `{host, port, mode, pid, started_at}`, present while a server is serving. |
| `app_token` | The token for public mode or a non-loopback bind: mode 0600, never printed. |
| `models/`, `gallery/` | Inputs. The server reads them and never writes them. |

The API also logs one JSON line per request on stderr (with no query string). It sets `X-Request-Id` on every response and refuses a request over 21 MB with 413.

## 9. API walkthrough (curl)

This walkthrough uses a private server through the tunnel, so it sends no token. In public mode every request needs the header `Authorization: Bearer <token>`. Keep that header in a file of mode 0600 and pass it with `-H @<file>`, so the token never appears on a command line. The options below are the published protocol.

```bash
B=http://localhost:8010
S=$(curl -s -X POST "$B/v1/sessions" -H 'Content-Type: application/json' -d '{}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
curl -N -X POST "$B/v1/sessions/$S/messages" -F 'image=@xray.png' -F 'options={"decode": "beam", "beam_size": 3, "max_new_tokens": 100, "cached_decode": true, "compile": false, "k_images": 4, "k_reports": 3, "label": true, "reference": null, "display_repair": false, "stop_on_repeat": false}'
curl -s "$B/v1/sessions/$S/export?format=md" -o session.md    # MIMIC-derived on a private server: see section 1
curl -s -X DELETE "$B/v1/sessions/$S" -o /dev/null -w '%{http_code}\n'   # 204
```

The turn streams Server-Sent Events, in this order:
1. `message_start`;
2. for each stage, `stage_start` and then `stage_end` (a skipped stage ends with a `skipped` reason). Inside the generate stage come `content_block_start`, one `content_block_delta` per decode step (a `beam_snapshot`), and `content_block_stop`;
3. `message_stop`, which carries the report.

While no event comes, a `: ping` comment arrives every 15 s.

Every route is documented at `$B/docs`.
