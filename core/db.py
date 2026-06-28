"""SQLite helpers for BinSight."""

import sqlite3
import json

from .paths import DB_PATH, SCHEMA_PATH


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with open(SCHEMA_PATH) as f:
        schema = f.read()
    conn = _conn()
    conn.executescript(schema)
    conn.close()


def get_stats():
    conn = _conn()
    cur = conn.cursor()
    total = cur.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    # Each row counts once: the manual label wins, else the model's auto label.
    clean = cur.execute(
        "SELECT COUNT(*) FROM images WHERE COALESCE(manual_label, auto_label)='clean'"
    ).fetchone()[0]
    dirty = cur.execute(
        "SELECT COUNT(*) FROM images WHERE COALESCE(manual_label, auto_label)='dirty'"
    ).fetchone()[0]
    unlabelled = cur.execute(
        "SELECT COUNT(*) FROM images WHERE manual_label IS NULL"
    ).fetchone()[0]
    conn.close()
    return {
        "total": total,
        "clean": clean,
        "dirty": dirty,
        "unlabelled": unlabelled,
        "clean_pct": round(clean / total * 100) if total else 0,
        "dirty_pct": round(dirty / total * 100) if total else 0,
    }


def get_recent(limit=6):
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM images ORDER BY upload_date DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def insert(record: dict):
    conn = _conn()
    cols = list(record.keys())
    placeholders = ",".join("?" for _ in cols)
    conn.execute(
        f"INSERT INTO images ({','.join(cols)}) VALUES ({placeholders})",
        list(record.values()),
    )
    conn.commit()
    last_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return last_id


def set_label(image_id: int, label: str):
    conn = _conn()
    conn.execute(
        "UPDATE images SET manual_label=? WHERE id=?", (label, image_id)
    )
    conn.commit()
    conn.close()


def get_next_unannotated(current_id=None):
    conn = _conn()
    if current_id:
        row = conn.execute("SELECT * FROM images WHERE id=?", (current_id,)).fetchone()
        if row:
            conn.close()
            return dict(row)
    row = conn.execute(
        "SELECT * FROM images WHERE manual_label IS NULL ORDER BY upload_date ASC LIMIT 1"
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_by_id(image_id: int):
    conn = _conn()
    row = conn.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_adjacent_ids(image_id: int):
    """Return (prev_id, next_id) for navigation."""
    conn = _conn()
    prev_row = conn.execute(
        "SELECT id FROM images WHERE id < ? ORDER BY id DESC LIMIT 1", (image_id,)
    ).fetchone()
    next_row = conn.execute(
        "SELECT id FROM images WHERE id > ? ORDER BY id ASC LIMIT 1", (image_id,)
    ).fetchone()
    conn.close()
    return (prev_row[0] if prev_row else None, next_row[0] if next_row else None)


def list_images():
    """Lightweight list for the image picker dropdown (newest first)."""
    conn = _conn()
    rows = conn.execute(
        "SELECT id, filename, manual_label, auto_label FROM images "
        "ORDER BY upload_date DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_located():
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM images WHERE latitude IS NOT NULL AND longitude IS NOT NULL ORDER BY upload_date DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def count_annotated():
    conn = _conn()
    total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    done = conn.execute(
        "SELECT COUNT(*) FROM images WHERE manual_label IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    return done, total


def get_position(image_id: int):
    """1-based position of an image in the id-ordered queue, and the total."""
    conn = _conn()
    pos = conn.execute(
        "SELECT COUNT(*) FROM images WHERE id <= ?", (image_id,)
    ).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    conn.close()
    return pos, total


def get_all_for_charts():
    conn = _conn()
    rows = conn.execute("SELECT * FROM images ORDER BY upload_date ASC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ----- configurable rule settings -----

def get_setting(key: str, default=None):
    conn = _conn()
    row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    conn.close()
    if row is None:
        return default
    try:
        return json.loads(row[0])
    except (ValueError, TypeError):
        return row[0]


def set_setting(key: str, value):
    conn = _conn()
    conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, json.dumps(value)),
    )
    conn.commit()
    conn.close()


def update_auto_label(image_id: int, label: str, confidence: float, reason: str):
    """Persist a (re)classification for one image."""
    conn = _conn()
    conn.execute(
        "UPDATE images SET auto_label=?, confidence=?, rule_reason=? WHERE id=?",
        (label, confidence, reason, image_id),
    )
    conn.commit()
    conn.close()
