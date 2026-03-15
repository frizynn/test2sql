"""SQL validation and extraction utilities."""

import re
import sqlite3

from fastapi import HTTPException

from config import logger

# ---------------------------------------------------------------------------
# SQL validation — SELECT-only enforcement
# ---------------------------------------------------------------------------
_BLOCKED_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|PRAGMA)\b",
    re.IGNORECASE,
)


def validate_sql(sql: str) -> None:
    """Raise HTTPException(400) if *sql* contains write/DDL keywords."""
    stripped = sql.strip()
    if not stripped.upper().startswith("SELECT"):
        raise HTTPException(
            status_code=400,
            detail={"error": "only_select_allowed", "detail": "Only SELECT statements are allowed."},
        )
    if _BLOCKED_KEYWORDS.search(stripped):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "blocked_keyword",
                "detail": "Statement contains a blocked keyword. Only SELECT queries are allowed.",
            },
        )


def extract_sql(raw_response: str) -> str:
    """Extract SQL from model response, stripping markdown fences and prefixes."""
    text = raw_response.strip()

    # Strip markdown code fences (```sql ... ``` or ``` ... ```)
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Remove leading "SQL:" or "sql:" prefixes
    for prefix in ("SQL:", "sql:", "SQL :", "sql :"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    return text


def validate_sql_explain(sql: str, db_path: str):
    """Validate SQL via EXPLAIN. Returns (True, '') on success, (False, error) on failure.

    Also rejects non-SELECT statements via validate_sql().
    """
    # Reject non-SELECT via existing validator
    try:
        validate_sql(sql)
    except HTTPException as exc:
        detail = exc.detail
        if isinstance(detail, dict):
            detail = detail.get("detail", str(detail))
        return (False, str(detail))

    # Run EXPLAIN to catch syntax errors, invalid columns/tables
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("EXPLAIN " + sql)
        return (True, "")
    except sqlite3.Error as exc:
        return (False, str(exc))
    finally:
        conn.close()
