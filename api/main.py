"""FastAPI service: CSV->SQLite loader + query/schema/health endpoints + text-to-SQL + NLG pipeline."""

import csv
import os
import sqlite3
import sys
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from config import (
    SCRIPT_DIR, CSV_PATH, DB_PATH, SCHEMA_PATH, HOST, PORT,
    EXPECTED_ROW_COUNT, logger, FEW_SHOT_DIR,
    _PROMPT_TEMPLATE, _SCHEMA_TEXT, _GRAMMAR_TEXT, _NLG_TEMPLATE,
    TEXT2SQL_URL, NLG_URL, MODEL_TIMEOUT, MAX_ATTEMPTS,
)
from few_shot_retriever import FewShotRetriever
from pipelines import PipelineContext, MCTSPipeline
from endpoints import router


# ---------------------------------------------------------------------------
# CSV -> SQLite loader (merged from db.py)
# ---------------------------------------------------------------------------
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sales (
    date TEXT NOT NULL,
    week_day TEXT NOT NULL,
    hour TEXT NOT NULL,
    ticket_number TEXT NOT NULL,
    waiter TEXT NOT NULL,
    product_name TEXT NOT NULL,
    quantity REAL NOT NULL,
    unitary_price REAL NOT NULL,
    total REAL NOT NULL
);
"""

INSERT_SQL = """
INSERT INTO sales (date, week_day, hour, ticket_number, waiter, product_name,
                   quantity, unitary_price, total)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


def load_csv_to_sqlite(csv_path: str, db_path: str) -> int:
    """Load *csv_path* into SQLite at *db_path*. Returns row count."""
    if not os.path.exists(csv_path):
        logger.error("CSV file not found: %s", csv_path)
        sys.exit(1)

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(CREATE_TABLE_SQL)

    row_count = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cur.execute(INSERT_SQL, (
                row["date"],
                row["week_day"],
                row["hour"],
                row["ticket_number"],
                row["waiter"],
                row["product_name"],
                float(row["quantity"]),
                float(row["unitary_price"]),
                float(row["total"]),
            ))
            row_count += 1

    conn.commit()
    conn.close()
    return row_count


# ---------------------------------------------------------------------------
# Lifespan -- load DB + pipeline artifacts on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading CSV -> SQLite  csv=%s  db=%s", CSV_PATH, DB_PATH)
    row_count = load_csv_to_sqlite(CSV_PATH, DB_PATH)
    if row_count != EXPECTED_ROW_COUNT:
        logger.error(
            "Row count mismatch: expected %d, got %d", EXPECTED_ROW_COUNT, row_count
        )
        sys.exit(1)
    logger.info(
        "DB ready  rows=%d  db=%s  schema=%s", row_count, DB_PATH, SCHEMA_PATH
    )
    app.state.db_path = DB_PATH
    app.state.row_count = row_count

    app.state.prompt_template = _PROMPT_TEMPLATE
    app.state.schema_text = _SCHEMA_TEXT
    app.state.grammar = _GRAMMAR_TEXT
    app.state.http_client = httpx.AsyncClient()
    app.state.nlg_template = _NLG_TEMPLATE
    logger.info(
        "Pipeline artifacts loaded  template=%d chars  schema=%d chars  grammar=%d chars  nlg=%d chars",
        len(_PROMPT_TEMPLATE), len(_SCHEMA_TEXT), len(_GRAMMAR_TEXT), len(_NLG_TEMPLATE),
    )

    pipeline_ctx = PipelineContext(
        http_client=app.state.http_client,
        grammar=app.state.grammar,
        db_path=app.state.db_path,
        text2sql_url=TEXT2SQL_URL,
        nlg_url=NLG_URL,
        model_timeout=MODEL_TIMEOUT,
        max_attempts=MAX_ATTEMPTS,
    )

    pool_path = os.path.join(FEW_SHOT_DIR, "text2sql_pool.json")
    try:
        pipeline_ctx.text2sql_retriever = FewShotRetriever(pool_path)
    except Exception as exc:
        logger.warning("Text2SQL retriever not loaded: %s", exc)

    app.state.pipeline = MCTSPipeline(pipeline_ctx)
    logger.info("Pipeline strategy: mcts")

    yield

    await app.state.http_client.aclose()
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Nivii Sales Query API",
    description="Execute SQL queries against the Nivii sales database.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Custom exception handler for HTTPException -- ensure structured JSON body
# ---------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Return structured error JSON for all HTTPExceptions."""
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "http_error", "detail": str(exc.detail)},
    )


# ---------------------------------------------------------------------------
# Register routes
# ---------------------------------------------------------------------------
app.include_router(router)


# ---------------------------------------------------------------------------
# Static file mount
# ---------------------------------------------------------------------------
_STATIC_DIR = os.path.join(SCRIPT_DIR, "..", "static")

if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
else:
    logger.warning("Static directory not found: %s -- /static mount skipped", _STATIC_DIR)


# ---------------------------------------------------------------------------
# __main__ -- quick dev startup
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
