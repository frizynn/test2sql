"""FastAPI route handlers using APIRouter (avoids circular import with main.py)."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from config import (
    SCRIPT_DIR, SCHEMA_PATH,
    NLG_URL, MODEL_TIMEOUT, logger,
)
from prompts import format_results_for_prompt, build_nlg_prompt
from sql_utils import validate_sql
from llm_client import call_nlg_model

# ---------------------------------------------------------------------------
# Pydantic models (merged from models.py)
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    sql: str


class QueryResponse(BaseModel):
    columns: list[str]
    rows: list[list]
    row_count: int


class SchemaResponse(BaseModel):
    schema_text: str


class HealthResponse(BaseModel):
    status: str
    rows: int


class ErrorResponse(BaseModel):
    error: str
    detail: str


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    sql: str
    results: dict
    raw_answer: str
    attempts: int
    metadata: Optional[dict] = None


class AnswerRequest(BaseModel):
    question: str
    sql: str
    results: dict


class AnswerResponse(BaseModel):
    answer: str


class PipelineRequest(BaseModel):
    question: str


class PipelineTimings(BaseModel):
    sql_latency_ms: float
    nlg_latency_ms: Optional[float] = None
    total_latency_ms: float


class PipelineNlgError(BaseModel):
    error: str
    detail: str


class PipelineResponse(BaseModel):
    question: str
    sql: str
    results: dict
    answer: Optional[str] = None
    nlg_error: Optional[PipelineNlgError] = None
    timings: PipelineTimings


router = APIRouter()

# ---------------------------------------------------------------------------
# Static file directory
# ---------------------------------------------------------------------------
_STATIC_DIR = os.path.join(SCRIPT_DIR, "..", "static")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("/query", response_model=QueryResponse, responses={400: {"model": ErrorResponse}})
async def execute_query(body: QueryRequest, request: Request):
    """Execute a SELECT query against the sales database."""
    validate_sql(body.sql)

    try:
        conn = sqlite3.connect(request.app.state.db_path)
        cur = conn.cursor()
        cur.execute(body.sql)
        columns = [desc[0] for desc in cur.description] if cur.description else []
        rows = [list(r) for r in cur.fetchall()]
        conn.close()
    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "sqlite_error",
                "detail": f"{exc}",
            },
        )

    return QueryResponse(columns=columns, rows=rows, row_count=len(rows))


@router.get("/schema")
async def get_schema():
    """Return the annotated CREATE TABLE statement."""
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema_text = f.read()
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail={"error": "schema_not_found", "detail": f"Schema file not found: {SCHEMA_PATH}"},
        )
    return {"schema": schema_text}


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Return service health with row count from the database."""
    try:
        conn = sqlite3.connect(request.app.state.db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sales")
        count = cur.fetchone()[0]
        conn.close()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "health_check_failed", "detail": str(exc)},
        )
    return HealthResponse(status="ok", rows=count)


def _translate_pipeline_result(result):
    """Translate PipelineResult to HTTP response or raise HTTPException."""
    if result.error:
        if result.error not in ("empty_model_output", "max_retries_exceeded"):
            raise HTTPException(
                status_code=503,
                detail={
                    "error": result.error,
                    "detail": result.error_detail or f"Model service error: {result.error}",
                },
            )
        if result.error == "empty_model_output":
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "empty_model_output",
                    "detail": result.error_detail or "Model returned empty SQL after extraction.",
                    "attempts": result.attempts,
                },
            )
        raise HTTPException(
            status_code=400,
            detail={
                "error": "max_retries_exceeded",
                "detail": result.error_detail or "Failed after max attempts.",
                "sql": result.sql,
                "attempts": result.attempts,
            },
        )

    return {
        "sql": result.sql,
        "results": result.results,
        "raw_answer": result.raw_answer,
        "attempts": result.attempts,
        "metadata": result.metadata,
    }


@router.post("/ask", response_model=AskResponse,
              responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
async def ask_question(body: AskRequest, request: Request):
    """Text-to-SQL pipeline: question -> prompt -> model -> extract -> validate -> execute -> respond."""
    question = body.question.strip()
    if not question:
        raise HTTPException(
            status_code=422,
            detail={"error": "empty_question", "detail": "Question must not be empty."},
        )

    pipeline = request.app.state.pipeline

    try:
        result = await pipeline.run(question)
    except Exception as exc:
        logger.exception("Pipeline crashed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "pipeline_error",
                "detail": f"Pipeline failed: {exc}",
            },
        )

    return _translate_pipeline_result(result)


@router.post("/ask/stream")
async def ask_stream(body: AskRequest, request: Request):
    """SSE streaming endpoint for live MCTS progress."""
    question = body.question.strip()
    if not question:
        raise HTTPException(
            status_code=422,
            detail={"error": "empty_question", "detail": "Question must not be empty."},
        )

    pipeline = request.app.state.pipeline
    event_queue: asyncio.Queue = asyncio.Queue()

    async def on_event(event: dict) -> None:
        await event_queue.put(event)

    async def generate():
        async def run_pipeline():
            try:
                result = await pipeline.run(question, on_event=on_event)
                if result.error:
                    await event_queue.put({
                        "_type": "error",
                        "error": result.error,
                        "detail": result.error_detail or "",
                    })
                else:
                    await event_queue.put({
                        "_type": "result",
                        "sql": result.sql,
                        "results": result.results,
                        "raw_answer": result.raw_answer,
                        "attempts": result.attempts,
                        "metadata": result.metadata,
                    })
            except Exception as exc:
                await event_queue.put({
                    "_type": "error",
                    "error": "pipeline_error",
                    "detail": str(exc),
                })
            finally:
                await event_queue.put(None)

        task = asyncio.create_task(run_pipeline())

        try:
            while True:
                if await request.is_disconnected():
                    task.cancel()
                    break
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if event is None:
                    break
                event_type = event.pop("_type", "step")
                yield f"event: {event_type}\ndata: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            task.cancel()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/answer", response_model=AnswerResponse,
              responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
async def generate_answer(body: AnswerRequest, request: Request):
    """NLG endpoint: question + SQL + results -> natural language Spanish answer."""
    question = body.question.strip()
    sql = body.sql.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail={"error": "empty_question", "detail": "Question must not be empty."},
        )
    if not sql:
        raise HTTPException(
            status_code=422,
            detail={"error": "empty_sql", "detail": "SQL must not be empty."},
        )

    rows = body.results.get("rows", []) if isinstance(body.results, dict) else []
    row_count = len(rows)
    results_str = format_results_for_prompt(rows)
    truncated = row_count > 20

    prompt = build_nlg_prompt(question, sql, results_str)

    model_result = await call_nlg_model(
        request.app.state.http_client,
        prompt,
        NLG_URL,
        MODEL_TIMEOUT,
    )

    if model_result["error"]:
        logger.warning(json.dumps({
            "event": "nlg_error",
            "question": question,
            "error": model_result["error"],
            "latency_ms": model_result["latency_ms"],
        }))
        raise HTTPException(
            status_code=503,
            detail={
                "error": model_result["error"],
                "detail": f"NLG model error: {model_result['error']}",
            },
        )

    content = model_result["content"]
    if not content:
        logger.warning(json.dumps({
            "event": "nlg_error",
            "question": question,
            "error": "empty_answer",
            "latency_ms": model_result["latency_ms"],
        }))
        raise HTTPException(
            status_code=503,
            detail={
                "error": "empty_answer",
                "detail": "NLG model returned an empty answer.",
            },
        )

    logger.info(json.dumps({
        "event": "nlg_attempt",
        "question": question,
        "latency_ms": model_result["latency_ms"],
        "row_count": row_count,
        "truncated": truncated,
    }))

    return {"answer": content}


@router.post("/pipeline", response_model=PipelineResponse,
              responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
async def pipeline_endpoint(body: PipelineRequest, request: Request):
    """Full pipeline: question -> text-to-SQL -> execute -> NLG answer, with per-step timings."""
    question = body.question.strip()
    if not question:
        raise HTTPException(
            status_code=422,
            detail={"error": "empty_question", "detail": "Question must not be empty."},
        )

    total_start = time.time()

    # --- Step 1: Text-to-SQL via pipeline delegation ---
    ask_start = time.time()
    pipeline = request.app.state.pipeline
    result = await pipeline.run(question)
    sql_latency_ms = round((time.time() - ask_start) * 1000, 1)

    if result.error:
        total_latency_ms = round((time.time() - total_start) * 1000, 1)
        err_detail: dict = {"error": result.error, "detail": result.error_detail or ""}
        logger.info(json.dumps({
            "event": "pipeline_request",
            "question": question,
            "sql_latency_ms": sql_latency_ms,
            "nlg_latency_ms": None,
            "total_latency_ms": total_latency_ms,
            "success": False,
            "error": err_detail,
        }))
        if result.error not in ("empty_model_output", "max_retries_exceeded"):
            status_code = 503
        else:
            status_code = 400
        raise HTTPException(status_code=status_code, detail=err_detail)

    sql = result.sql
    results = result.results

    # --- Step 2: NLG answer via /answer logic ---
    nlg_start = time.time()
    try:
        answer_result = await generate_answer(AnswerRequest(
            question=question, sql=sql, results=results,
        ), request)
    except HTTPException as exc:
        nlg_latency_ms = round((time.time() - nlg_start) * 1000, 1)
        total_latency_ms = round((time.time() - total_start) * 1000, 1)

        nlg_err_detail = exc.detail if isinstance(exc.detail, dict) else {"error": "nlg_failed", "detail": str(exc.detail)}
        logger.info(json.dumps({
            "event": "pipeline_request",
            "question": question,
            "sql_latency_ms": sql_latency_ms,
            "nlg_latency_ms": nlg_latency_ms,
            "total_latency_ms": total_latency_ms,
            "success": False,
            "error": nlg_err_detail,
        }))
        return PipelineResponse(
            question=question,
            sql=sql,
            results=results,
            answer=None,
            nlg_error=PipelineNlgError(
                error=nlg_err_detail.get("error", "nlg_failed"),
                detail=nlg_err_detail.get("detail", str(exc.detail)),
            ),
            timings=PipelineTimings(
                sql_latency_ms=sql_latency_ms,
                nlg_latency_ms=nlg_latency_ms,
                total_latency_ms=total_latency_ms,
            ),
        )

    nlg_latency_ms = round((time.time() - nlg_start) * 1000, 1)
    total_latency_ms = round((time.time() - total_start) * 1000, 1)

    if isinstance(answer_result, dict):
        answer = answer_result["answer"]
    else:
        answer = answer_result.answer

    logger.info(json.dumps({
        "event": "pipeline_request",
        "question": question,
        "sql_latency_ms": sql_latency_ms,
        "nlg_latency_ms": nlg_latency_ms,
        "total_latency_ms": total_latency_ms,
        "success": True,
    }))

    return PipelineResponse(
        question=question,
        sql=sql,
        results=results,
        answer=answer,
        timings=PipelineTimings(
            sql_latency_ms=sql_latency_ms,
            nlg_latency_ms=nlg_latency_ms,
            total_latency_ms=total_latency_ms,
        ),
    )


# ---------------------------------------------------------------------------
# Static file serving -- GET /
# ---------------------------------------------------------------------------
@router.get("/", include_in_schema=False)
async def root():
    """Serve the web UI index page."""
    index_path = os.path.join(_STATIC_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path, media_type="text/html")
    return JSONResponse(
        status_code=404,
        content={"error": "not_found", "detail": "index.html not found in static/"},
    )
