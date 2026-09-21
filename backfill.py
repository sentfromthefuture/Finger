"""
One-off backfill: fingerprint every existing track and beat.
Safe to re-run; already-fingerprinted items are skipped.

    SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=... python backfill.py

Needs fpcalc installed locally (Debian/Ubuntu: apt install libchromaprint-tools).
"""
import httpx

from main import HEADERS, PAGE, SUPABASE_URL, TABLES, process_record


def rows_with_files(table: str, path_col: str):
    offset = 0
    while True:
        r = httpx.get(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers=HEADERS,
            params={
                "select": f"id,title,{path_col}",
                path_col: "not.is.null",
                "order": "id",
                "limit": PAGE,
                "offset": offset,
            },
            timeout=60,
        )
        r.raise_for_status()
        rows = r.json()
        yield from rows
        if len(rows) < PAGE:
            return
        offset += PAGE


if __name__ == "__main__":
    for table, (_, path_col) in TABLES.items():
        for record in rows_with_files(table, path_col):
            label = f"{table} {record['id']} ({record.get('title')})"
            try:
                print(label, "->", process_record(table, record))
            except Exception as e:
                print(label, "-> FAILED:", e)
