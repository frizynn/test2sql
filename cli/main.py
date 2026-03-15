"""nivii — Natural Language SQL for a Real Sales Database.

Usage:
  python -m cli                Interactive mode (setup → choose UI or TUI)
  python -m cli ask "question" One-shot query (requires running servers)
  python -m cli setup          Run setup wizard only
"""

from __future__ import annotations

import atexit
import sys
from typing import Optional, Sequence

from rich.console import Console
from rich.panel import Panel

console = Console(stderr=True)

_LOGO = """[bold]  ┏┓╻╻╻ ╻╻╻
  ┃┃┃┃┃╻┃┃┃
  ╹ ╹╹┗┛╹╹[/bold]"""


def _header():
    console.print()
    console.print(_LOGO, style="bold")
    console.print("  [dim]Natural Language SQL · local inference[/dim]")
    console.print()


def _setup_and_start() -> bool:
    """Run setup checks, fix missing deps, start servers. Returns True if ready."""
    from cli.setup import (
        run_checks, render_checks, fix_missing,
        start_stack, dry_run, stop_stack,
    )

    console.print("[bold]Environment[/bold]")
    console.print()
    checks, all_ok = run_checks()
    render_checks(checks)

    if not all_ok:
        console.print()
        fixed = fix_missing(checks)
        if not fixed:
            console.print("\n[red]Setup incomplete. Fix the issues above and retry.[/red]")
            return False
        # Re-check
        checks, all_ok = run_checks()
        if not all_ok:
            console.print("\n[red]Still missing dependencies.[/red]")
            render_checks([c for c in checks if not c.ok])
            return False

    console.print()
    console.print("[bold]Servers[/bold]")
    console.print()
    if not start_stack():
        return False

    atexit.register(stop_stack)

    if not dry_run():
        console.print("[yellow]Dry run had issues, but servers are up. Proceeding.[/yellow]")

    return True


def _repl():
    """Interactive REPL — type questions, get SQL + results + NLG answer."""
    from cli.ask_cmd import run_ask

    console.print()
    console.print(Panel(
        "[bold]Ready.[/bold] Type a question in Spanish.\n"
        "[dim]Commands: /quit  /help[/dim]",
        border_style="green",
        expand=False,
    ))
    console.print()

    try:
        import readline
        import os
        hist = os.path.expanduser("~/.nivii_history")
        try:
            readline.read_history_file(hist)
        except (FileNotFoundError, OSError):
            pass
        readline.set_history_length(500)
    except ImportError:
        hist = None

    while True:
        try:
            q = console.input("[bold green]>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            break

        if not q:
            continue
        if q in ("/quit", "/exit", "/q"):
            console.print("[dim]Bye.[/dim]")
            break
        if q == "/help":
            console.print("[dim]Type a question about sales data. /quit to exit.[/dim]")
            continue

        run_ask(q, full_pipeline=True)
        console.print()

    if hist:
        try:
            readline.write_history_file(hist)  # type: ignore[possibly-undefined]
        except (OSError, NameError):
            pass


def _prompt_mode() -> str:
    """Ask user whether to keep the web UI running or enter TUI mode."""
    console.print()
    console.print(Panel(
        "[bold]Choose mode:[/bold]\n\n"
        "  [bold cyan]1[/bold cyan]  Web UI — keep servers running, open http://localhost:8000\n"
        "  [bold cyan]2[/bold cyan]  TUI  — interactive text-to-SQL in this terminal\n",
        border_style="blue",
        expand=False,
    ))

    while True:
        choice = console.input("[bold blue]Mode [1/2]:[/bold blue] ").strip()
        if choice in ("1", "web", "ui"):
            return "web"
        if choice in ("2", "tui", "repl"):
            return "tui"
        console.print("[dim]Enter 1 or 2[/dim]")


def _web_mode():
    """Keep servers running and show the URL. Wait for Ctrl+C."""
    import webbrowser

    url = "http://localhost:8000"
    console.print()
    console.print(Panel(
        f"[bold green]Web UI running at:[/bold green] {url}\n\n"
        "[dim]Press Ctrl+C to stop all servers and exit.[/dim]",
        border_style="green",
        expand=False,
    ))

    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopping servers...[/dim]")


def _handle_ask_cmd(question: str, url: str) -> int:
    """One-shot ask — assumes servers are already running."""
    from cli.ask_cmd import run_ask
    return run_ask(question, full_pipeline=True, url=url)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="nivii",
        description="nivii — Natural Language SQL for a Real Sales Database",
        add_help=True,
    )
    subparsers = parser.add_subparsers(dest="command")

    # ask
    ask_p = subparsers.add_parser("ask", help="One-shot query")
    ask_p.add_argument("question", help="Question in natural language")
    ask_p.add_argument("--url", default="http://localhost:8000")

    # setup
    subparsers.add_parser("setup", help="Run setup wizard only")

    args = parser.parse_args(argv)

    if args.command == "ask":
        return _handle_ask_cmd(args.question, args.url)

    if args.command == "setup":
        _header()
        ok = _setup_and_start()
        return 0 if ok else 1

    # Default: interactive mode — setup → mode choice
    _header()
    if not _setup_and_start():
        return 1

    mode = _prompt_mode()
    if mode == "web":
        _web_mode()
    else:
        _repl()

    return 0
