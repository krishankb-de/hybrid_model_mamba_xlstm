#!/usr/bin/env python3
"""Build a compact, DUA-compliant local MIMIC-CXR image-text dataset from
PhysioNet WITHOUT ever storing the full ~570 GB (JPG) / 4.7 TB (DICOM) source.

Strategy: download in small chunks -> downscale in flight -> delete the
originals. Peak disk stays ~4 GB regardless of how many studies are built.
See H100_SCALING_PLAN.md Phase 8 for the full design discussion.

Emits, under --out:
    files/p10/p10000032/s50414267.jpg      # one frontal image per study, WxW
    train.parquet / validate.parquet / test.parquet   # official split
    manifest.parquet                        # everything, incl. excluded rows
    build_report.json                       # counts, so the thesis can cite them

The parquet schema matches what scripts/train_contrastive.py already expects
(MIMICJointDataset.__getitem__ needs zero changes -- it already branches on
isinstance(image, str)):
    image (str, absolute path)  findings (str)  impression (str)
plus study_id / subject_id / dicom_id / view / split / report_hash for
protocol work (Phase 8D leakage guard, Phase 6C-3 style duplicate audits).

PREREQUISITES
-------------
1. PhysioNet credentialing + signed DUA for BOTH:
     https://physionet.org/content/mimic-cxr/2.1.0/       (reports, 135 MB)
     https://physionet.org/content/mimic-cxr-jpg/2.1.0/   (images + split + metadata)
2. A session cookie in ~/.physionet_session, chmod 600, containing ONLY the
   'sessionid' cookie value (no other structure -- unlike .netrc).

   WHY NOT .netrc / HTTP Basic Auth: verified live 2026-08-16 that PhysioNet's
   Django deployment does NOT honour HTTP Basic Auth for this project --
   `curl -u user https://physionet.org/settings/profile/` returns 302 to
   /login/ regardless of credential correctness, and the same account's
   Authorization header against a /files/ URL returns 403, while the
   identical URL with a valid session cookie returns 200. The
   `wget --user --ask-password` recipe printed on PhysioNet project pages is
   stale for this deployment. Get the cookie value from a logged-in browser
   (dev tools -> Application/Storage -> Cookies -> physionet.org ->
   'sessionid'), then on the machine that will run `meta`/`fetch`:
     umask 077; printf '%s' 'SESSIONID_VALUE' > ~/.physionet_session
     chmod 600 ~/.physionet_session
   Treat this file with the same care as a password -- anyone holding a live
   session cookie can act as you on physionet.org until it expires or you
   log out. Refresh it (repeat the steps above) if a run reports the cookie
   has expired.
3. Outbound HTTPS from wherever `meta`/`fetch` run. On this project's
   cluster (aisc), the login node refuses to execute anything at all --
   route through scripts/build_mimic_cxr_local.sh (STAGE=... sbatch ...),
   which targets the general cpu-batch/cpu-interactive partitions.

USAGE
-----
    # 1. small files only: split csv, metadata csv, filename list, reports zip (~150 MB)
    python build_mimic_cxr_local.py meta --out /sc/home/$USER/dataset/mimic_full

    # 2. decide what to build (no network) -- writes manifest.parquet + build_report.json
    python build_mimic_cxr_local.py manifest --out /sc/home/$USER/dataset/mimic_full \\
        --views PA AP --size 320

    # STOP AND READ build_report.json here. If with_findings is far below
    # ~150k, the section parser mismatched the report format (it shouldn't --
    # it's the official one -- but verify before spending hours on `fetch`).

    # 3. smoke test (~15 min for 500 images), THEN the long one (resumable --
    #    a killed job costs at most one in-flight chunk, rerun the same command)
    python build_mimic_cxr_local.py fetch --out /sc/home/$USER/dataset/mimic_full \\
        --size 320 --chunk 2000 --workers 8 --limit 500
    python build_mimic_cxr_local.py fetch --out /sc/home/$USER/dataset/mimic_full \\
        --size 320 --chunk 2000 --workers 8

    # 4. leakage guard + write train/validate/test parquet
    python build_mimic_cxr_local.py pack --out /sc/home/$USER/dataset/mimic_full \\
        --exclude-hashes legacy_gallery_hashes.txt

Add --limit N to any stage for a smoke test before committing to the full run.
Run `fetch` under tmux/screen/an sbatch job -- it is multi-hour and a dropped
SSH session should not kill it.
"""

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mimic_cxr_vendor.extract import extract_findings_impression  # noqa: E402

CXR = "https://physionet.org/files/mimic-cxr/2.1.0"
JPG = "https://physionet.org/files/mimic-cxr-jpg/2.1.0"

SECTION_PARSER_COMMIT = "e8d26fff0fe1632f6cf53d31158e888cffb18ab8"
CREATE_SECTION_FILES_COMMIT = "18cdc41ca483f98659a8e649081f17c10558c3c3"

SMALL_FILES = [
    (JPG + "/mimic-cxr-2.0.0-split.csv.gz", "mimic-cxr-2.0.0-split.csv.gz"),
    (JPG + "/mimic-cxr-2.0.0-metadata.csv.gz", "mimic-cxr-2.0.0-metadata.csv.gz"),
    (JPG + "/mimic-cxr-2.0.0-chexpert.csv.gz", "mimic-cxr-2.0.0-chexpert.csv.gz"),
    (CXR + "/mimic-cxr-reports.zip", "mimic-cxr-reports.zip"),
]

_session = None  # type: Optional[requests.Session]
_session_lock = threading.Lock()

SESSION_EXPIRED = 498  # local sentinel (not a real HTTP status) -- see _download()


def _get_session() -> requests.Session:
    """PhysioNet's Django app does NOT honour HTTP Basic Auth (verified live,
    2026-08-16: `curl -u user https://physionet.org/settings/profile/`
    returns 302 to /login/ regardless of credentials, and the same account's
    Authorization header against a /files/ URL returns 403, while the
    identical URL with a valid session cookie returns 200). The
    `wget --user --ask-password` recipe printed on PhysioNet project pages is
    stale for this deployment. Session-cookie auth (~/.physionet_session) is
    the working path -- see the module docstring's PREREQUISITES. A leftover
    ~/.netrc for physionet.org is simply unused now, not actively read.

    Thread-safe (REGRESSION, caught live 2026-08-16): stage_fetch calls this
    from up to `workers` ThreadPoolExecutor threads concurrently on the first
    chunk. The original check-then-set (`if _session is None: ... _session =
    ...`) is not atomic, so multiple threads raced past the None check before
    any of them finished creating a Session -- harmless correctness-wise
    (every racing thread reads the same cookie file and sets the same
    cookie), but printed "[auth] session cookie loaded" once per racing
    thread (5 times, observed live) and meant later callers could end up
    holding different Session objects, splitting the connection pool
    needlessly. Double-checked locking: the fast path (session already
    created) takes no lock, so this costs nothing after the first call.
    """
    global _session
    if _session is not None:
        return _session
    with _session_lock:
        if _session is None:
            cookie_file = Path.home() / ".physionet_session"
            if not cookie_file.exists():
                raise RuntimeError(
                    "[auth] {} not found. PhysioNet does not accept HTTP Basic "
                    "Auth for this project -- log in at physionet.org in a "
                    "browser, copy the 'sessionid' cookie value (dev tools -> "
                    "Application/Storage -> Cookies -> physionet.org), then on "
                    "this machine: umask 077; printf '%s' 'SESSIONID_VALUE' > "
                    "{}; chmod 600 {}".format(cookie_file, cookie_file, cookie_file)
                )
            sid = cookie_file.read_text().strip()
            sess = requests.Session()
            sess.cookies.set("sessionid", sid, domain="physionet.org")
            print("[auth] session cookie loaded from {} ({} chars)".format(cookie_file, len(sid)))
            _session = sess
    return _session


def _download(url: str, dest: Path, timeout: int = 60, retries: int = 3) -> int:
    """Returns the HTTP status code, SESSION_EXPIRED (498, a local sentinel
    meaning "this looked like a 200 but was actually a login page"), or -1
    on a connection-level failure after all retries. Streams to avoid
    holding large files in memory.

    IMPORTANT: 200 is returned ONLY after the full body has been consumed
    AND atomically renamed into dest. A status LINE of 200 that then fails
    mid-body (connection dropped, SLURM kill) must never be reported as a
    successful 200 to the caller -- see the REGRESSION note below.
    """
    sess = _get_session()
    last_status = -1
    for attempt in range(retries):
        try:
            resp = sess.get(url, timeout=timeout, stream=True)
        except requests.RequestException as e:
            print("  [download] {} attempt {}/{}: {}".format(url, attempt + 1, retries, e),
                  file=sys.stderr)
            time.sleep(2 ** attempt)
            continue

        if resp.status_code == 200:
            # A session cookie can expire mid-run. PhysioNet then 302s to
            # /login/, and requests follows redirects by default -- so this
            # arrives as an ordinary 200 with an HTML login page as the
            # body. Check BEFORE streaming any bytes to disk, or a dead
            # cookie silently writes login pages into .jpg/.csv.gz files,
            # and the resume check (Path.exists()) would then skip them
            # forever on every subsequent run.
            ctype = resp.headers.get("Content-Type", "")
            if "text/html" in ctype or "/login" in resp.url:
                resp.close()
                return SESSION_EXPIRED

            # Stream to a .part sibling and Path.replace() into dest only
            # after the full body is consumed. Path.replace() is an atomic
            # rename on the same filesystem (POSIX guarantee), so dest
            # either doesn't exist or is complete -- never partial. Without
            # this, a SLURM kill (time limit / preemption) mid-write leaves
            # a truncated-but-nonzero-size file at dest, and every caller's
            # resume check (dest.exists() and dest.stat().st_size > 0) then
            # trusts it as done.
            #
            # REGRESSION (caught live, 2026-08-16): a --time=00:10:00 kill
            # mid-download of mimic-cxr-reports.zip left a corrupt zip that
            # the next run's stage_meta happily skipped as "[meta] have
            # mimic-cxr-reports.zip" -- the corruption only surfaced later
            # as zipfile.BadZipFile at unzip time. Fixing that also
            # surfaced a SECOND, worse bug in this function: the status
            # code arrives with the response HEADERS, before the body is
            # streamed, so a connection that dies partway through the body
            # (same failure class, just interrupted a step earlier) would
            # have this function return the status LINE's 200 even though
            # dest was correctly never written -- silently reporting
            # success for a download that produced nothing. The inner
            # try/except below scopes body-streaming failures separately
            # from the outer connection-attempt try/except so a body
            # failure retries as a fresh request rather than exiting the
            # loop with a stale, misleading "200".
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_name(dest.name + ".part")
            try:
                with open(part, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            except requests.RequestException as e:
                resp.close()
                print("  [download] {} attempt {}/{}: body stream interrupted: {}".format(
                    url, attempt + 1, retries, e), file=sys.stderr)
                last_status = -1
                time.sleep(2 ** attempt)
                continue
            resp.close()
            part.replace(dest)
            return 200

        last_status = resp.status_code
        resp.close()
        if resp.status_code in (401, 403):
            return resp.status_code  # auth failure -- retrying won't help
        time.sleep(2 ** attempt)
    return last_status


# --------------------------------------------------------------------------- #
# stage: meta
# --------------------------------------------------------------------------- #
def stage_meta(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    # Small companion file with no query params -- fetch it too, it is the
    # partial-download filename manifest referenced by the PhysioNet page.
    small = list(SMALL_FILES) + [(JPG + "/IMAGE_FILENAMES", "IMAGE_FILENAMES")]

    for url, name in small:
        dest = out / name
        if dest.exists() and dest.stat().st_size > 0:
            print("[meta] have {}".format(name))
            continue
        print("[meta] GET {}".format(name))
        status = _download(url, dest)
        if status != 200:
            if dest.exists():
                dest.unlink()
            if status == SESSION_EXPIRED:
                raise RuntimeError(
                    "[meta] {} -> PhysioNet redirected to /login/ (session "
                    "cookie missing, expired, or wrong). Refresh "
                    "~/.physionet_session with a current 'sessionid' cookie "
                    "value from a logged-in browser session.".format(name)
                )
            raise RuntimeError(
                "[meta] {} -> HTTP {}. Verify the mimic-cxr-jpg DUA is "
                "signed on this account and ~/.physionet_session holds a "
                "valid, current session cookie.".format(name, status)
            )

    rep_dir = out / "reports"
    if not rep_dir.exists():
        print("[meta] unzipping reports ...")
        with zipfile.ZipFile(out / "mimic-cxr-reports.zip") as z:
            z.extractall(rep_dir)
    print("[meta] done -> {}".format(out))


# --------------------------------------------------------------------------- #
# report parsing
# --------------------------------------------------------------------------- #
def report_path(root: Path, subject_id: int, study_id: int) -> Path:
    s = "p{}".format(subject_id)
    return root / "files" / s[:3] / s / "s{}.txt".format(study_id)


def norm_hash(text: str) -> str:
    """Case/whitespace-normalised blake2b, matching the EXACT convention
    already used in this repo (normalize_report_text @ evaluate_cxr_retrieval.py:414,
    the text_hash construction @ train_contrastive.py:419-424) so hashes here
    are directly joinable against the legacy gallery for the Phase 8D leakage
    guard without a second normalisation scheme to keep in sync."""
    norm = " ".join((text or "").lower().split())
    return hashlib.blake2b(norm.encode("utf-8"), digest_size=8).hexdigest()


# --------------------------------------------------------------------------- #
# stage: manifest (no network)
# --------------------------------------------------------------------------- #
def stage_manifest(out: Path, views: List[str], size: int, limit: int) -> None:
    split = pd.read_csv(out / "mimic-cxr-2.0.0-split.csv.gz")
    meta = pd.read_csv(
        out / "mimic-cxr-2.0.0-metadata.csv.gz",
        usecols=["dicom_id", "subject_id", "study_id", "ViewPosition", "Rows", "Columns"],
    )
    df = split.merge(meta, on=["dicom_id", "subject_id", "study_id"], how="left")
    n_all = len(df)

    # 1. frontal views only. Laterals pair to the SAME report as the frontal,
    #    which manufactures guaranteed in-batch false negatives (the thing
    #    6C-3 measured this project's data does NOT currently have) and is a
    #    different visual distribution.
    df = df[df["ViewPosition"].isin(views)]
    n_frontal = len(df)

    # 2. one image per study -- the report is study-level, so >1 image/study
    #    duplicates the text side. Prefer the earlier entry in `views`
    #    (default PA over AP), then the largest image.
    order = {v: i for i, v in enumerate(views)}
    df = df.assign(
        _v=df["ViewPosition"].map(order).fillna(99),
        _px=-(df["Rows"].fillna(0) * df["Columns"].fillna(0)),
    )
    df = df.sort_values(["study_id", "_v", "_px"]).drop_duplicates("study_id")
    df = df.drop(columns=["_v", "_px"])
    n_study = len(df)

    if limit:
        df = df.head(limit)

    # 3. attach report text via the VENDORED official section parser
    #    (H100_SCALING_PLAN.md Phase 8C -- provenance is the reason this
    #    build exists at all, so an unaudited regex here would forfeit it).
    rep_root = out / "reports"
    findings_col, impression_col = [], []
    missing = 0
    for sid, stid in zip(df["subject_id"], df["study_id"]):
        p = report_path(rep_root, sid, stid)
        if not p.exists():
            findings_col.append("")
            impression_col.append("")
            missing += 1
            continue
        text = p.read_text(errors="ignore")
        f, i = extract_findings_impression(text, "s{}".format(stid))
        findings_col.append(f)
        impression_col.append(i)
    df["findings"] = findings_col
    df["impression"] = impression_col

    # 4. two counts, kept SEPARATE per Phase 8F -- the RRG literature
    #    conditions on FINDINGS specifically; the retrieval chapter
    #    concatenated both. Do not collapse this choice into one filter.
    df["has_findings"] = df["findings"].str.len() > 0
    df["has_text"] = (df["findings"].str.len() + df["impression"].str.len()) > 0
    n_findings = int(df["has_findings"].sum())
    n_text = int(df["has_text"].sum())

    df["report_hash"] = [
        norm_hash("Findings: {} Impression: {}".format(f, i))
        for f, i in zip(df["findings"], df["impression"])
    ]
    df["rel_jpg"] = [
        "files/p{}/p{}/s{}/{}.jpg".format(str(s)[:2], s, t, d)
        for s, t, d in zip(df["subject_id"], df["study_id"], df["dicom_id"])
    ]
    df["local_jpg"] = [
        str(out / "files" / "p{}".format(str(s)[:2]) / "p{}".format(s) / "s{}.jpg".format(t))
        for s, t in zip(df["subject_id"], df["study_id"])
    ]

    df.to_parquet(out / "manifest.parquet", index=False)
    split_counts = df[df["has_text"]]["split"].value_counts().to_dict()
    rep = {
        "rows_in_split_csv": n_all,
        "after_frontal_filter": n_frontal,
        "after_one_per_study": n_study,
        "reports_missing_on_disk": missing,
        "with_findings": n_findings,
        "with_findings_or_impression": n_text,
        "split_counts_with_text": {str(k): int(v) for k, v in split_counts.items()},
        "views_kept": views,
        "stored_resolution": size,
        "est_stored_gb": round(n_text * (size / 320.0) ** 2 * 30e3 / 1e9, 2),
        "est_transfer_gb": round(n_text * 1.48e6 / 1e9, 1),
        "section_parser_commit": SECTION_PARSER_COMMIT,
        "create_section_files_commit": CREATE_SECTION_FILES_COMMIT,
    }
    (out / "build_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print("[manifest] -> {}".format(out / "manifest.parquet"))
    if n_findings < 140000:
        print(
            "[manifest] WARNING: with_findings ({}) is well below the ~150-160k "
            "expected from the official parser on the full corpus. If --limit "
            "was not set, stop and check the report path convention "
            "(files/pXX/pSUBJECT/sSTUDY.txt) before running `fetch`.".format(n_findings),
            file=sys.stderr,
        )


# --------------------------------------------------------------------------- #
# stage: fetch
# --------------------------------------------------------------------------- #
def _resize_one(args: Tuple[str, str, int]) -> bool:
    src, dst, size = args
    try:
        with Image.open(src) as im:
            im = im.convert("L").resize((size, size), Image.BICUBIC)
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, "JPEG", quality=90, optimize=True)
        return True
    except Exception as e:  # corrupt/truncated download -- skip, retry next run
        print("[fetch] BAD {}: {}".format(src, e), file=sys.stderr)
        return False


def stage_fetch(out: Path, size: int, chunk: int, workers: int, limit: int) -> None:
    df = pd.read_parquet(out / "manifest.parquet")
    df = df[df["has_text"]]
    if limit:
        df = df.head(limit)

    todo = df[~df["local_jpg"].map(lambda p: Path(p).exists())]
    print(
        "[fetch] {} target / {} still to do (~{:.0f} GB of transfer, none of it kept)".format(
            len(df), len(todo), len(todo) * 1.48e6 / 1e9
        )
    )

    stage = out / "_stage"
    done = 0
    for start in range(0, len(todo), chunk):
        part = todo.iloc[start:start + chunk]
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)

        # Absolute per-file URL + explicit destination -- no --cut-dirs, no
        # wget --base list. Threaded (I/O-bound), status codes tallied so an
        # all-401 chunk is distinguishable from a few tolerated 404s.
        statuses = {}  # type: Dict[int, int]
        staged_srcs = []  # type: List[Tuple[str, str]]  # (staged_path, rel_jpg)

        def _fetch_one(rel_jpg: str) -> Tuple[str, int]:
            dest = stage / rel_jpg
            status = _download(JPG + "/" + rel_jpg, dest)
            return rel_jpg, status

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_fetch_one, r) for r in part["rel_jpg"]]
            # Progress inside the download phase, not just after each full
            # chunk completes -- a 2000-file chunk at real-world PhysioNet/
            # shared-uplink throughput can take well over an hour, and with
            # no output during that time it is impossible to tell "slow but
            # working" from "hung" without shelling out to `du` on the
            # staging dir. ~10 lines per chunk regardless of chunk size.
            report_every = max(1, len(futures) // 10)
            n_completed = 0
            for fut in as_completed(futures):
                rel_jpg, status = fut.result()
                statuses[status] = statuses.get(status, 0) + 1
                if status == 200:
                    staged_srcs.append((str(stage / rel_jpg), rel_jpg))
                n_completed += 1
                if n_completed % report_every == 0 or n_completed == len(futures):
                    print("[fetch]   downloading {}/{} in this chunk ({} ok so far)".format(
                        n_completed, len(futures), statuses.get(200, 0)))

        # A cookie can expire MID-CHUNK: some files in this same chunk may
        # have succeeded before it did. That means `ok` can be > 0 even when
        # SESSION_EXPIRED also appears, so this check must be unconditional
        # and come before the ok==0 check below, not folded into it.
        expired = statuses.get(SESSION_EXPIRED, 0)
        if expired > 0:
            shutil.rmtree(stage, ignore_errors=True)
            raise RuntimeError(
                "[fetch] chunk at row {} hit {} session-expired responses "
                "(of {} files) -- PhysioNet redirected to /login/ mid-chunk. "
                "STOP-EVERYTHING condition, not a per-file one: files "
                "converted in earlier chunks are safe (this run is "
                "resumable), but continuing past this risks writing HTML "
                "login pages in as image files. Refresh "
                "~/.physionet_session with a fresh 'sessionid' cookie and "
                "resubmit -- it picks up where it stopped.".format(
                    start, expired, len(part)
                )
            )

        ok = statuses.get(200, 0)
        if ok == 0 and len(part) > 0:
            shutil.rmtree(stage, ignore_errors=True)
            raise RuntimeError(
                "[fetch] chunk at row {} converted 0 of {} images. Status "
                "histogram: {}. Verify ~/.physionet_session is present and "
                "current, and that the mimic-cxr-jpg DUA is signed on this "
                "account -- fix that before rerunning (the run is "
                "resumable; nothing before this chunk was lost).".format(
                    start, len(part), statuses
                )
            )
        if statuses.get(401, 0) + statuses.get(403, 0) > 0:
            print(
                "[fetch] WARNING: {} auth failures in this chunk (of {} ok). "
                "Investigate before the failures compound.".format(
                    statuses.get(401, 0) + statuses.get(403, 0), ok
                ),
                file=sys.stderr,
            )

        rel_to_local = dict(zip(part["rel_jpg"], part["local_jpg"]))
        jobs = [(src, rel_to_local[rel], size) for src, rel in staged_srcs]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_resize_one, jobs, chunksize=16))
        done += sum(results)

        shutil.rmtree(stage, ignore_errors=True)  # the whole point: originals never accumulate
        pct = 100.0 * (start + len(part)) / max(len(todo), 1)
        print("[fetch] {:5.1f}%  status={}  converted so far: {}".format(pct, statuses, done))

    print("[fetch] complete: {} images at {}px".format(done, size))


# --------------------------------------------------------------------------- #
# stage: pack
# --------------------------------------------------------------------------- #
def stage_pack(out: Path, exclude_hashes: str, min_match_frac: float, allow_low_match: bool) -> None:
    df = pd.read_parquet(out / "manifest.parquet")
    df = df[df["has_text"]]
    df = df[df["local_jpg"].map(lambda p: Path(p).exists())]
    print("[pack] {} rows with both text and a fetched image".format(len(df)))

    # Phase 8D leakage guard: subject-level exclusion, hash-joined against the
    # legacy eval gallery (evaluate_cxr_retrieval.py's train[90%:] N=3063).
    # MANDATORY recall check -- the legacy gallery was built by a mirror with
    # an UNKNOWN section parser, so this join WILL silently under-match, and
    # every miss is a leaked subject. <95% match is a FAIL by default: the
    # legacy gallery becomes unusable as a comparison and the official split
    # is the only defensible metric.
    if exclude_hashes:
        bad = {h.strip() for h in Path(exclude_hashes).read_text().split() if h.strip()}
        hit = df["report_hash"].isin(bad)
        matched = int(hit.sum())
        total = len(bad)
        match_frac = matched / total if total else 0.0
        print(
            "[pack] legacy-gallery hash join: {}/{} ({:.1%}) matched".format(
                matched, total, match_frac
            )
        )
        if match_frac < min_match_frac and not allow_low_match:
            raise RuntimeError(
                "[pack] leakage-guard match rate {:.1%} is below the required {:.0%}. "
                "Per H100_SCALING_PLAN.md Phase 8D this is a HARD FAIL: the legacy "
                "gallery's provenance cannot be trusted enough to guarantee subject "
                "exclusion. Drop the legacy train[90%:] comparison and report only "
                "the official subject-disjoint split, OR pass --allow-low-match to "
                "proceed anyway (only if you have independently verified the miss "
                "reason, e.g. via manual spot checks).".format(match_frac, min_match_frac)
            )
        subjects = set(df.loc[hit, "subject_id"])
        keep = ~df["subject_id"].isin(subjects)
        print(
            "[pack] dropping {} rows across {} leaked subjects".format(
                int((~keep).sum()), len(subjects)
            )
        )
        df = df[keep]

    cols = ["local_jpg", "findings", "impression", "study_id", "subject_id",
            "dicom_id", "ViewPosition", "report_hash", "split"]
    df = df[cols].rename(columns={"local_jpg": "image", "ViewPosition": "view"})

    split_file = {"train": "train", "validate": "validate", "test": "test"}
    for name, fname in split_file.items():
        part = df[df["split"] == name].reset_index(drop=True)
        part.to_parquet(out / "{}.parquet".format(fname), index=False)
        print(
            "[pack] {:9s} {:>7d} pairs  ({} subjects)".format(
                name, len(part), part["subject_id"].nunique()
            )
        )
    print("[pack] -> {}/*.parquet".format(out))


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("stage", choices=["meta", "manifest", "fetch", "pack"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--size", type=int, default=320,
                     help="stored square resolution (224 = exact, 320 = crop headroom)")
    ap.add_argument("--views", nargs="+", default=["PA", "AP"])
    ap.add_argument("--chunk", type=int, default=2000, help="images per download/resize batch")
    ap.add_argument("--workers", type=int, default=8, help="download threads / resize processes")
    ap.add_argument("--limit", type=int, default=0, help="smoke-test cap")
    ap.add_argument("--exclude-hashes", default="",
                     help="pack: file of report_hash values whose subjects to drop")
    ap.add_argument("--min-match-frac", type=float, default=0.95,
                     help="pack: minimum leakage-guard hash-match rate before failing")
    ap.add_argument("--allow-low-match", action="store_true",
                     help="pack: proceed even if --min-match-frac is not met (see error message)")
    a = ap.parse_args()

    if a.stage == "meta":
        stage_meta(a.out)
    elif a.stage == "manifest":
        stage_manifest(a.out, a.views, a.size, a.limit)
    elif a.stage == "fetch":
        stage_fetch(a.out, a.size, a.chunk, a.workers, a.limit)
    else:
        stage_pack(a.out, a.exclude_hashes, a.min_match_frac, a.allow_low_match)


if __name__ == "__main__":
    main()
