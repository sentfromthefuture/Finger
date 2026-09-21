"""
Audio fingerprint service.

Supabase Database Webhook (tracks / beats, Insert + Update) -> POST /fingerprint
-> download the file -> fpcalc -> compare with stored fingerprints -> save result.

Environment variables:
  SUPABASE_URL                https://<project>.supabase.co
  SUPABASE_SERVICE_ROLE_KEY   service-role key (server only, never in the browser)
  WEBHOOK_SECRET              the webhook sends it as "Authorization: Bearer <secret>"
  MIN_OVERLAP                 optional, default 0.20: flag when this much of the shorter track matches
  WINDOW_MATCH                optional, default 0.72: how closely a 10-second window must match (0..1)
  FP_SECONDS                  optional, default 300: how many seconds of each track to fingerprint
"""
import hmac
import os
import subprocess
import tempfile
import threading
import time
import traceback
from contextlib import asynccontextmanager
from urllib.parse import quote

import httpx
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SERVICE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
MIN_OVERLAP = float(os.getenv("MIN_OVERLAP", "0.20"))
WINDOW_MATCH = float(os.getenv("WINDOW_MATCH", "0.72"))
FP_SECONDS = int(os.getenv("FP_SECONDS", "300"))

BUCKETS = ["private-full", "public-previews"]  # paid uploads, free uploads
WINDOW = 80  # fingerprint items per comparison window (about 10 seconds)
STEP = 40    # window spacing (about 5 seconds)
PAGE = 500
SWEEP_SECONDS = int(os.getenv("SWEEP_SECONDS", "300"))  # catch-up scan interval
MAX_FAILURES = 3  # give up on a broken file after this many tries (until restart)

PROCESS_LOCK = threading.Lock()  # one file at a time (small free instance, and avoids races)
failures: dict = {}

# table -> (item_type, column holding the storage path)
TABLES = {"tracks": ("track", "file_path"), "beats": ("beat", "full_path")}
TABLE_FOR_TYPE = {"track": "tracks", "beat": "beats"}

HEADERS = {"apikey": SERVICE_KEY, "Authorization": f"Bearer {SERVICE_KEY}"}



@asynccontextmanager
async def lifespan(app):
    # Catch-up loop: on every start (e.g. waking from sleep) and every few minutes,
    # fingerprint anything that was missed while the service was offline.
    threading.Thread(target=sweep_loop, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)


# ---------- fingerprinting ----------

def fingerprint_file(path: str) -> list[int]:
    """Run fpcalc and return the raw fingerprint as signed 32-bit ints (first FP_SECONDS seconds)."""
    out = subprocess.run(
        ["fpcalc", "-raw", "-signed", "-length", str(FP_SECONDS), path],
        capture_output=True, text=True, timeout=600, check=True,
    ).stdout
    for line in out.splitlines():
        if line.startswith("FINGERPRINT="):
            values = [int(x) for x in line[len("FINGERPRINT="):].split(",") if x]
            if values:
                return values
    raise RuntimeError("fpcalc returned no fingerprint")


def to_array(fp: list[int]) -> np.ndarray:
    return np.array(fp, dtype=np.int32).view(np.uint32)


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction (0..1) of the shorter track that also appears somewhere in the longer one.

    Slides ~10-second windows of the shorter track across the longer one and counts
    the windows that find a close match, so reused sections are caught even when the
    rest of the song is different. Silent/near-constant windows are ignored.
    """
    if len(a) > len(b):
        a, b = b, a
    w = min(WINDOW, len(a))
    if w < 8:
        return 0.0
    views = sliding_window_view(b, w)  # every possible alignment in the longer track
    matched = total = 0
    for start in range(0, len(a) - w + 1, STEP):
        win = a[start:start + w]
        if len(np.unique(win)) < w // 4:  # silence or a flat tone: skip
            continue
        total += 1
        bits = np.bitwise_count(views ^ win).sum(axis=1)
        if 1.0 - int(bits.min()) / (32.0 * w) >= WINDOW_MATCH:
            matched += 1
    return matched / total if total else 0.0


# ---------- Supabase helpers ----------

def check(r):
    """Like raise_for_status, but includes the server's message so the logs explain the failure."""
    if r.status_code >= 400:
        raise RuntimeError(f"{r.request.method} {r.request.url.path} -> {r.status_code}: {r.text[:300]}")


def download_to_temp(storage_path: str) -> str:
    """Free uploads live in public-previews, paid ones in private-full: try both."""
    ext = os.path.splitext(storage_path)[1] or ".mp3"
    quoted = quote(storage_path, safe="/")
    fd, tmp = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    errors = []
    for bucket in BUCKETS:
        for route in ("object/authenticated", "object"):
            url = f"{SUPABASE_URL}/storage/v1/{route}/{bucket}/{quoted}"
            with httpx.stream("GET", url, headers=HEADERS, timeout=120) as r:
                if r.status_code == 200:
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_bytes():
                            f.write(chunk)
                    return tmp
                r.read()
                errors.append(f"{bucket}/{route}: {r.status_code} {r.text[:120]}")
    os.remove(tmp)
    raise RuntimeError(f"storage download failed ({storage_path}): " + " | ".join(errors))


def already_done(item_type: str, item_id: str) -> bool:
    r = httpx.get(
        f"{SUPABASE_URL}/rest/v1/audio_fingerprints",
        headers=HEADERS,
        params={"select": "item_id", "item_type": f"eq.{item_type}", "item_id": f"eq.{item_id}"},
        timeout=30,
    )
    check(r)
    return len(r.json()) > 0


def find_best_match(item_type: str, item_id: str, fp: list[int]):
    """Compare against every stored fingerprint. Returns (score, row or None)."""
    mine = to_array(fp)
    best_score, best_row = 0.0, None
    offset = 0
    while True:
        r = httpx.get(
            f"{SUPABASE_URL}/rest/v1/audio_fingerprints",
            headers=HEADERS,
            params={
                "select": "item_type,item_id,fingerprint",
                "order": "item_type,item_id",
                "limit": PAGE,
                "offset": offset,
            },
            timeout=60,
        )
        check(r)
        rows = r.json()
        for row in rows:
            if row["item_type"] == item_type and row["item_id"] == item_id:
                continue
            score = overlap(mine, to_array(row["fingerprint"]))
            if score > best_score:
                best_score, best_row = score, row
        if len(rows) < PAGE:
            break
        offset += PAGE
    return best_score, best_row


def get_title(item_type: str, item_id: str):
    r = httpx.get(
        f"{SUPABASE_URL}/rest/v1/{TABLE_FOR_TYPE[item_type]}",
        headers=HEADERS,
        params={"select": "title", "id": f"eq.{item_id}"},
        timeout=30,
    )
    check(r)
    rows = r.json()
    return rows[0]["title"] if rows else None


def save_result(payload: dict):
    r = httpx.post(
        f"{SUPABASE_URL}/rest/v1/audio_fingerprints",
        headers={**HEADERS, "Content-Type": "application/json",
                 "Prefer": "resolution=merge-duplicates,return=minimal"},
        params={"on_conflict": "item_type,item_id"},
        json=payload,
        timeout=60,
    )
    check(r)


# ---------- main pipeline ----------

def process_record(table: str, record: dict, force: bool = False) -> str:
    with PROCESS_LOCK:
        return _process_record(table, record, force)


def _process_record(table: str, record: dict, force: bool = False) -> str:
    item_type, path_col = TABLES[table]
    item_id = str(record["id"])
    storage_path = record.get(path_col)
    if not storage_path:
        return "no file yet"
    if not force and already_done(item_type, item_id):
        return "already fingerprinted"

    tmp = download_to_temp(storage_path)
    try:
        fp = fingerprint_file(tmp)
    finally:
        os.remove(tmp)

    score, row = find_best_match(item_type, item_id, fp)
    is_match = row is not None and score >= MIN_OVERLAP

    save_result({
        "item_type": item_type,
        "item_id": item_id,
        "fingerprint": fp,
        "match_item_type": row["item_type"] if is_match else None,
        "match_item_id": row["item_id"] if is_match else None,
        "match_title": get_title(row["item_type"], row["item_id"]) if is_match else None,
        "match_score": round(score, 4) if is_match else None,
    })
    return f"done (best score {score:.3f}, {'MATCH' if is_match else 'no match'})"


def safe_process(table: str, record: dict, force: bool):
    try:
        print(f"[{table} {record.get('id')}] {process_record(table, record, force)}")
    except Exception:
        print(f"[{table} {record.get('id')}] FAILED")
        traceback.print_exc()


# ---------- catch-up sweep ----------

def fetch_paged(path: str, params: dict):
    offset = 0
    while True:
        r = httpx.get(
            f"{SUPABASE_URL}/rest/v1/{path}",
            headers=HEADERS,
            params={**params, "limit": PAGE, "offset": offset},
            timeout=60,
        )
        check(r)
        rows = r.json()
        yield from rows
        if len(rows) < PAGE:
            return
        offset += PAGE


def sweep():
    """Fingerprint every track/beat that has a file but no fingerprint yet.
    Items waiting for review go first, so approvals are never stuck behind old uploads."""
    done = {
        (r["item_type"], r["item_id"])
        for r in fetch_paged("audio_fingerprints", {"select": "item_type,item_id", "order": "item_type,item_id"})
    }
    todo = []
    for table, (item_type, path_col) in TABLES.items():
        for rec in fetch_paged(table, {
            "select": f"id,title,status,{path_col}",
            path_col: "not.is.null",
            "order": "id",
        }):
            key = (item_type, str(rec["id"]))
            if key in done or failures.get(key, 0) >= MAX_FAILURES:
                continue
            todo.append((0 if rec.get("status") == "pending_review" else 1, table, key, rec))
    todo.sort(key=lambda t: t[0])

    for _, table, key, rec in todo:
        try:
            print(f"[sweep {table} {rec['id']}] {process_record(table, rec)}")
        except Exception:
            failures[key] = failures.get(key, 0) + 1
            print(f"[sweep {table} {rec['id']}] FAILED ({failures[key]}/{MAX_FAILURES})")
            traceback.print_exc()


def sweep_loop():
    while True:
        try:
            sweep()
        except Exception:
            traceback.print_exc()
        time.sleep(SWEEP_SECONDS)


# ---------- HTTP ----------

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/fingerprint")
async def fingerprint(
    request: Request,
    background: BackgroundTasks,
    authorization: str = Header(default=""),
):
    expected = f"Bearer {WEBHOOK_SECRET}"
    if not WEBHOOK_SECRET or not hmac.compare_digest(authorization.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="unauthorized")

    body = await request.json()
    table = body.get("table")
    record = body.get("record") or {}
    if table not in TABLES or not record.get("id"):
        return {"skipped": True}

    # Re-fingerprint if the file path changed on an update.
    old = body.get("old_record") or {}
    path_col = TABLES[table][1]
    force = bool(old) and old.get(path_col) != record.get(path_col)

    # Answer right away (webhooks time out quickly); work continues in the background.
    background.add_task(safe_process, table, record, force)
    return {"queued": True}
