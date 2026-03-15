"""Interactive query with pipeline trace — `nivii ask` implementation.

Public API:
    run_ask(question, url) -> int   (exit code: 0=ok, 1=error)
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from cli.main import console

_TIMEOUT = 120
_MAX_DISPLAY_ROWS = 20


# ---------------------------------------------------------------------------
# HTTP call
# ---------------------------------------------------------------------------
def _call_ask_api(question: str, url: str) -> dict:
    endpoint = f"{url.rstrip('/')}/ask"
    payload = {"question": question}
    body = json.dumps(payload).encode("utf-8")

    try:
        req = urllib.request.Request(
            endpoint, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))

    except urllib.error.HTTPError as exc:
        try:
            err_body = json.loads(exc.read().decode("utf-8"))
        except Exception:
            err_body = {}
        return {
            "status": "error",
            "http_status": exc.code,
            "error": err_body.get("error", f"HTTP {exc.code}"),
            "detail": err_body.get("detail", str(exc.reason)),
            "sql": err_body.get("sql"),
            "attempts": err_body.get("attempts"),
        }
    except urllib.error.URLError as exc:
        return {"status": "error", "error": "Connection failed",
                "detail": f"{endpoint} — {exc.reason}"}
    except (OSError, ValueError) as exc:
        return {"status": "error", "error": "Request failed",
                "detail": str(exc)}


def _call_pipeline_api(question: str, url: str) -> dict:
    """Call the full pipeline endpoint (text2sql + NLG answer)."""
    endpoint = f"{url.rstrip('/')}/pipeline"
    payload = {"question": question}
    body = json.dumps(payload).encode("utf-8")

    try:
        req = urllib.request.Request(
            endpoint, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            err_body = json.loads(exc.read().decode("utf-8"))
        except Exception:
            err_body = {}
        return {
            "status": "error",
            "http_status": exc.code,
            "error": err_body.get("error", f"HTTP {exc.code}"),
            "detail": err_body.get("detail", str(exc.reason)),
        }
    except urllib.error.URLError as exc:
        return {"status": "error", "error": "Connection failed",
                "detail": f"{endpoint} — {exc.reason}"}
    except (OSError, ValueError) as exc:
        return {"status": "error", "error": "Request failed",
                "detail": str(exc)}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _render_results_table(columns: list, rows: list) -> None:
    table = Table(show_lines=True)
    for col in columns:
        table.add_column(str(col), style="cyan")

    total = len(rows)
    for row in rows[:_MAX_DISPLAY_ROWS]:
        table.add_row(*(str(v) if v is not None else "[dim]NULL[/dim]" for v in row))

    console.print(table)
    if total > _MAX_DISPLAY_ROWS:
        console.print(f"[dim]… {total - _MAX_DISPLAY_ROWS} more rows[/dim]")


def _render_error(data: dict) -> None:
    parts: list[str] = []
    status = data.get("http_status")
    if status:
        parts.append(f"[bold]Status:[/bold] {status}")
    parts.append(f"[bold]Error:[/bold] {data.get('error', 'Unknown')}")
    detail = data.get("detail")
    if detail:
        parts.append(f"[bold]Detail:[/bold] {detail}")
    sql = data.get("sql")
    if sql:
        parts.append(f"\n[bold]SQL attempted:[/bold]\n{sql}")

    console.print(Panel("\n".join(parts), title="[red]Error[/red]",
                        border_style="red", expand=False))


def _render_trace(data: dict) -> None:
    if data.get("status") == "error":
        _render_error(data)
        return

    # SQL
    sql = data.get("sql")
    if sql:
        console.print(Panel(
            Syntax(sql, "sql", theme="monokai", word_wrap=True),
            title="SQL", border_style="green", expand=False,
        ))

    # Results table
    results = data.get("results")
    if results and isinstance(results, dict):
        columns = results.get("columns", [])
        rows = results.get("rows", [])
        if columns:
            _render_results_table(columns, rows)

    # NLG answer (from /pipeline endpoint)
    answer = data.get("answer")
    if answer:
        console.print(Panel(answer, title="Respuesta",
                            border_style="blue", expand=False))

    # NLG error (partial success)
    nlg_err = data.get("nlg_error")
    if nlg_err and isinstance(nlg_err, dict):
        console.print(f"[yellow]NLG warning:[/yellow] {nlg_err.get('detail', '')}")

    # Timings
    timings = data.get("timings")
    if timings and isinstance(timings, dict):
        parts = []
        if timings.get("sql_latency_ms"):
            parts.append(f"SQL {timings['sql_latency_ms']:.0f}ms")
        if timings.get("nlg_latency_ms"):
            parts.append(f"NLG {timings['nlg_latency_ms']:.0f}ms")
        if timings.get("total_latency_ms"):
            parts.append(f"Total {timings['total_latency_ms']:.0f}ms")
        if parts:
            console.print(f"[dim]{' · '.join(parts)}[/dim]")

    # Attempts
    attempts = data.get("attempts")
    if attempts and attempts > 1:
        console.print(f"[dim]Attempts: {attempts}[/dim]")


# ---------------------------------------------------------------------------
# Progress phases
# ---------------------------------------------------------------------------
_PHASES = [
    (0, "Sending question"),
    (2, "Generating SQL"),
    (6, "Executing query"),
    (12, "Generating answer"),
]


def _progress_spinner(done: threading.Event) -> None:
    """Show progress phases while waiting for the API response."""
    start = time.monotonic()
    phase_idx = 0
    with console.status("") as status:
        while not done.is_set():
            elapsed = time.monotonic() - start
            # Advance phase
            while (phase_idx < len(_PHASES) - 1
                   and elapsed >= _PHASES[phase_idx + 1][0]):
                phase_idx += 1
            label = _PHASES[phase_idx][1]
            status.update(f"[bold]{label}...[/bold] [dim]({elapsed:.0f}s)[/dim]")
            done.wait(0.25)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_ask(
    question: str,
    *,
    full_pipeline: bool = False,
    url: str = "http://localhost:8000",
) -> int:
    """Execute a question and render results. Returns 0 on success, 1 on error."""
    done = threading.Event()
    spinner = threading.Thread(target=_progress_spinner, args=(done,), daemon=True)
    spinner.start()

    try:
        if full_pipeline:
            data = _call_pipeline_api(question, url)
        else:
            data = _call_ask_api(question, url)
    finally:
        done.set()
        spinner.join(timeout=2)

    console.print()
    _render_trace(data)

    return 0 if data.get("status") != "error" else 1
