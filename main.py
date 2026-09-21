"""
Audio fingerprint service.

Supabase Database Webhook (tracks / beats, Insert + Update) -> POST /fingerprint
-> download the file -> fpcalc -> compare with stored fingerprints -> save result.

Environment variables:
  SUPABASE_URL                https://<project>.supabase.co
  SUPABASE_SERVICE_ROLE_KEY   service-role key (server only, never in the browser)
  WEBHOOK_SECRET              the webhook sends it as "Authorization: Bearer <secret>"
  MATCH_THRESHOLD             optional, default 0.85 (tune on your own tracks)
"""
import hmac
import os
import subprocess
import tempfile
import traceback
from urllib.parse import quote

import httpx
import numpy as np
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SERVICE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.85"))

BUCKET = "private-full"
MAX_OFFSET = 60  # alignment slack in fingerprint items (roughly 8 seconds)
PAGE = 500

# table -> (item_type, column holding the storage path)
TABLES = {"tracks": ("track", "file_path"), "beats": ("beat", "full_path")}
TABLE_FOR_TYPE = {"track": "tracks", "beat": "beats"}

HEADERS = {"apikey": SERVICE_KEY, "Authorization": f"Bearer {SERVICE_KEY}"}

app = FastAPI()


# ---------- fingerprinting ----------

def fingerprint_file(path: str) -> list[int]:
    """Run fpcalc and return the raw fingerprint as signed 32-bit ints (first ~2 minutes)."""
    out = subprocess.run(
        ["fpcalc", "-raw", "-signed", path],
        capture_output=True, text=True, timeout=180, check=True,
    ).stdout
    for line in out.splitlines():
        if line.startswith("FINGERPRINT="):
            values = [int(x) for x in line[len("FINGERPRINT="):].split(",") if x]
            if values:
                return values
    raise RuntimeError("fpcalc returned no fingerprint")


def to_array(fp: list[int]) -> np.ndarray:
    return np.array(fp, dtype=np.int32).view(np.uint32)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Best bit-level similarity (0..1) over a range of alignments."""
    best = 0.0
    min_overlap = max(1, int(0.5 * min(len(a), len(b))))
    for off in range(-MAX_OFFSET, MAX_OFFSET + 1):
        if off >= 0:
            x, y = a[off:], b
        else:
            x, y = a, b[-off:]
        n = min(len(x), len(y))
        if n < min_overlap:
            continue
        bits = int(np.bitwise_count(x[:n] ^ y[:n]).sum())
        best = max(best, 1.0 - bits / (32.0 * n))
    return best


# ---------- Supabase helpers ----------

def download_to_temp(storage_path: str) -> str:
    ext = os.path.splitext(storage_path)[1] or ".mp3"
    url = f"{SUPABASE_URL}/storage/v1/object/authenticated/{BUCKET}/{quote(storage_path, safe='/')}"
    fd, tmp = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    try:
        with httpx.stream("GET", url, headers=HEADERS, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    except Exception:
        os.remove(tmp)
        raise
    return tmp


def already_done(item_type: str, item_id: str) -> bool:
    r = httpx.get(
        f"{SUPABASE_URL}/rest/v1/audio_fingerprints",
        headers=HEADERS,
        params={"select": "item_id", "item_type": f"eq.{item_type}", "item_id": f"eq.{item_id}"},
        timeout=30,
    )
    r.raise_for_status()
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
        r.raise_for_status()
        rows = r.json()
        for row in rows:
            if row["item_type"] == item_type and row["item_id"] == item_id:
                continue
            score = similarity(mine, to_array(row["fingerprint"]))
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
    r.raise_for_status()
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
    r.raise_for_status()


# ---------- main pipeline ----------

def process_record(table: str, record: dict, force: bool = False) -> str:
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
    is_match = row is not None and score >= MATCH_THRESHOLD

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
