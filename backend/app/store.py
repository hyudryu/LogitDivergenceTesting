"""SQLite-backed run store. Results are stored as JSON blobs; runs are small
enough that this is more than sufficient and keeps the project dependency-free."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = os.environ.get(
    "LOGIT_DB_PATH",
    str(Path(__file__).resolve().parents[1] / "data" / "harness.db"),
)

_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with _lock, _conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at REAL NOT NULL,
                status TEXT NOT NULL,
                progress TEXT,
                params TEXT,
                endpoints TEXT,
                result TEXT,
                error TEXT
            )
            """
        )


def create_run(name: str, endpoints: List[dict], params: dict) -> str:
    run_id = uuid.uuid4().hex[:12]
    with _lock, _conn() as c:
        c.execute(
            "INSERT INTO runs (id, name, created_at, status, progress, params, endpoints) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                name,
                time.time(),
                "queued",
                json.dumps({"step": "queued", "pct": 0}),
                json.dumps(params),
                json.dumps(endpoints),
            ),
        )
    return run_id


def update_run(
    run_id: str,
    *,
    status: Optional[str] = None,
    progress: Optional[Dict[str, Any]] = None,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    fields, values = [], []
    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if progress is not None:
        fields.append("progress = ?")
        values.append(json.dumps(progress))
    if result is not None:
        fields.append("result = ?")
        values.append(json.dumps(result))
    if error is not None:
        fields.append("error = ?")
        values.append(error)
    if not fields:
        return
    values.append(run_id)
    with _lock, _conn() as c:
        c.execute(f"UPDATE runs SET {', '.join(fields)} WHERE id = ?", values)


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    with _lock, _conn() as c:
        row = c.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return _row_to_dict(row) if row else None


def list_runs() -> List[Dict[str, Any]]:
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT id, name, created_at, status, progress, params, endpoints FROM runs "
            "ORDER BY created_at DESC"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def delete_run(run_id: str) -> None:
    with _lock, _conn() as c:
        c.execute("DELETE FROM runs WHERE id = ?", (run_id,))


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)

    def load(key: str) -> Any:
        raw = d.get(key)
        return json.loads(raw) if raw else None

    d["created_at"] = d.get("created_at")
    d["progress"] = load("progress")
    d["params"] = load("params")
    d["endpoints"] = load("endpoints")
    if "result" in d:
        d["result"] = load("result")
    return d
