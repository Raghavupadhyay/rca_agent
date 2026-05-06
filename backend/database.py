"""
database.py — SQLite connection and safe query helper.
All SQL access goes through this module.
"""

import sqlite3
import os
from typing import Any

# Path to the SQLite database — can be overridden via env var
DB_PATH = os.environ.get("DB_PATH", "data.sqlite")

# The gold table name as loaded from CSV
TABLE_NAME = "my_table"


def get_connection() -> sqlite3.Connection:
    """Return a new SQLite connection with row_factory set to dict-like rows."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def query(sql: str, params: tuple = ()) -> list[dict]:
    """
    Execute a read-only SELECT query and return results as a list of dicts.
    Raises ValueError if the statement is not a SELECT.
    """
    stripped = sql.strip().upper()
    if not stripped.startswith("SELECT"):
        raise ValueError(f"Only SELECT statements are allowed. Got: {sql[:80]}")

    with get_connection() as conn:
        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def query_one(sql: str, params: tuple = ()) -> dict | None:
    """Execute a SELECT and return the first row as a dict, or None."""
    results = query(sql, params)
    return results[0] if results else None


def table_exists() -> bool:
    """Verify that the gold table exists in the database."""
    result = query_one(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (TABLE_NAME,),
    )
    return result is not None