"""Setup wizard — check environment, models, dependencies, servers.

Runs on first launch or `nivii setup`. Validates everything needed
to run queries locally and offers to fix what's missing.

Dependencies resolved:
  - llama-server: installed via Homebrew (macOS) or built from source
  - Models: downloaded from HuggingFace
  - Python packages: installed via pip
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from cli.main import console

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _PROJECT_ROOT / "models"
_API_DIR = _PROJECT_ROOT / "api"

# ---------------------------------------------------------------------------
# Models — aligned with README and .env.example
# ---------------------------------------------------------------------------
TEXT2SQL_MODEL = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
NLG_MODEL = "Qwen3.5-2B-Q4_K_M.gguf"

TEXT2SQL_URL = "https://huggingface.co/frizynn/qwen2.5-coder-1.5b-instruct-q4_k_m-gguf/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
NLG_DOWNLOAD_URL = "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf"

# Ports for native servers
TEXT2SQL_PORT = 8081
NLG_PORT = 8082
API_PORT = 8000


# ---------------------------------------------------------------------------
# Check dataclass
# ---------------------------------------------------------------------------
class Check:
    __slots__ = ("name", "ok", "detail", "fixable")

    def __init__(self, name: str, ok: bool, detail: str = "", fixable: bool = False):
        self.name = name
        self.ok = ok
        self.detail = detail
        self.fixable = fixable


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_python() -> Check:
    v = sys.version_info
    ok = v >= (3, 9)
    return Check("Python ≥ 3.9", ok, f"{v.major}.{v.minor}.{v.micro}")


def check_llama_server() -> Check:
    path = shutil.which("llama-server")
    if path:
        return Check("llama-server", True, path)
    return Check("llama-server", False, "not found", fixable=True)


def check_model(name: str) -> Check:
    path = _MODELS_DIR / name
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        return Check(name, True, f"{size_mb:.0f} MB")
    return Check(name, False, "not found", fixable=True)


def check_pip_package(pkg_name: str, import_name: str | None = None) -> Check:
    try:
        __import__(import_name or pkg_name)
        return Check(pkg_name, True)
    except ImportError:
        return Check(pkg_name, False, "not installed", fixable=True)


# ---------------------------------------------------------------------------
# Run all checks
# ---------------------------------------------------------------------------
def run_checks() -> tuple[list[Check], bool]:
    checks = [
        check_python(),
        check_llama_server(),
        check_model(TEXT2SQL_MODEL),
        check_model(NLG_MODEL),
        check_pip_package("fastapi"),
        check_pip_package("uvicorn"),
        check_pip_package("httpx"),
        check_pip_package("rich"),
    ]
    all_ok = all(c.ok for c in checks)
    return checks, all_ok


def render_checks(checks: list[Check]) -> None:
    from rich.table import Table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold", width=4)
    table.add_column(width=50)
    table.add_column(style="dim")

    for c in checks:
        icon = "[green]✓[/green]" if c.ok else "[red]✗[/red]"
        table.add_row(icon, c.name, c.detail)

    console.print(table)


# ---------------------------------------------------------------------------
# Fix missing dependencies
# ---------------------------------------------------------------------------
def _install_llama_server() -> bool:
    """Install llama-server. Uses Homebrew on macOS, builds from source otherwise."""
    system = platform.system()
    if system == "Darwin":
        console.print("  Installing llama.cpp via Homebrew...")
        try:
            subprocess.run(
                ["brew", "install", "llama.cpp"],
                check=True,
                capture_output=True,
            )
            if shutil.which("llama-server"):
                console.print("  [green]✓[/green] llama-server installed")
                return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            console.print(f"  [red]✗[/red] brew install failed: {e}")

    # Fallback: build from source
    console.print("  Building llama.cpp from source (b8270)...")
    try:
        tmp = Path("/tmp/llama-cpp-build")
        if tmp.exists():
            shutil.rmtree(tmp)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", "b8270",
             "https://github.com/ggml-org/llama.cpp.git", str(tmp)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["cmake", "-B", str(tmp / "build"),
             "-DGGML_NATIVE=ON", "-DCMAKE_BUILD_TYPE=Release"],
            cwd=str(tmp), check=True, capture_output=True,
        )
        subprocess.run(
            ["cmake", "--build", str(tmp / "build"),
             "--target", "llama-server", f"-j{os.cpu_count() or 4}"],
            cwd=str(tmp), check=True, capture_output=True,
        )
        # Copy binary to a standard location
        built = tmp / "build" / "bin" / "llama-server"
        if built.exists():
            dest = Path.home() / ".local" / "bin" / "llama-server"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(built, dest)
            dest.chmod(0o755)
            console.print(f"  [green]✓[/green] llama-server installed to {dest}")
            console.print(f"  [dim]Add {dest.parent} to PATH if not already.[/dim]")
            return True
    except (subprocess.CalledProcessError, OSError) as e:
        console.print(f"  [red]✗[/red] Build failed: {e}")

    return False


def _download_model(name: str, url: str) -> bool:
    dest = _MODELS_DIR / name
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    console.print(f"  Downloading [bold]{name}[/bold]...")
    try:
        cmd = ["curl", "-L", "-o", str(dest), "--progress-bar", url]
        subprocess.run(cmd, check=True)
        size_mb = dest.stat().st_size / (1024 * 1024)
        console.print(f"  [green]✓[/green] {name} ({size_mb:.0f} MB)")
        return True
    except (subprocess.CalledProcessError, OSError) as e:
        console.print(f"  [red]✗[/red] Download failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def _install_pip_packages(packages: list[str]) -> bool:
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet"] + packages,
            check=True, capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def fix_missing(checks: list[Check]) -> bool:
    """Fix all missing dependencies. Returns True if everything resolved."""
    failed = [c for c in checks if not c.ok]
    if not failed:
        return True

    console.print()

    # llama-server
    llama_missing = any(c.name == "llama-server" for c in failed)
    if llama_missing:
        if not _install_llama_server():
            console.print("\n[red]Could not install llama-server.[/red]")
            console.print("[dim]Install manually: brew install llama.cpp (macOS) or build from source.[/dim]")
            return False

    # pip packages
    pip_missing = [c.name for c in failed
                   if c.name in ("fastapi", "uvicorn", "httpx", "rich")]
    if pip_missing:
        console.print(f"  Installing packages: [bold]{', '.join(pip_missing)}[/bold]")
        if _install_pip_packages(pip_missing):
            console.print("  [green]✓[/green] Packages installed")
        else:
            console.print("  [red]✗[/red] pip install failed")
            return False

    # Models
    model_map = {
        TEXT2SQL_MODEL: TEXT2SQL_URL,
        NLG_MODEL: NLG_DOWNLOAD_URL,
    }
    models_missing = [(c.name, model_map[c.name])
                      for c in failed if c.name in model_map]
    if models_missing:
        total_mb = sum(1100 if "1.5b" in n.lower() else 500 for n, _ in models_missing)
        console.print(f"  Missing models (~{total_mb / 1024:.1f} GB download):")
        for name, _ in models_missing:
            console.print(f"    • {name}")
        answer = console.input("  Download? [Y/n] ").strip().lower()
        if answer in ("", "y", "yes", "si", "sí"):
            for name, url in models_missing:
                if not _download_model(name, url):
                    return False
        else:
            return False

    return True


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------
_procs: dict[str, subprocess.Popen] = {}


def _port_listening(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def start_server(name: str, port: int, model: str,
                 extra_flags: list[str] | None = None) -> bool:
    """Start a llama-server instance and wait for health check."""
    if _port_listening(port):
        console.print(f"  [green]✓[/green] {name} already running on :{port}")
        return True

    model_path = str(_MODELS_DIR / model)
    if not os.path.isfile(model_path):
        console.print(f"  [red]✗[/red] Model not found: {model_path}")
        return False

    cmd = [
        "llama-server",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--model", model_path,
        "--ctx-size", "2048",
        "-ngl", "99",
    ]
    if extra_flags:
        cmd.extend(extra_flags)

    _procs[name] = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    health_url = f"http://localhost:{port}/health"
    for _ in range(90):
        try:
            urllib.request.urlopen(health_url, timeout=1)
            return True
        except Exception:
            # Check if process died
            if _procs[name].poll() is not None:
                console.print(f"  [red]✗[/red] {name} process exited unexpectedly")
                return False
            time.sleep(1)

    console.print(f"  [red]✗[/red] {name} health check timed out (90s)")
    return False


def start_api() -> bool:
    """Start the FastAPI server."""
    if _port_listening(API_PORT):
        console.print(f"  [green]✓[/green] api already running on :{API_PORT}")
        return True

    env = os.environ.copy()
    env["TEXT2SQL_URL"] = f"http://localhost:{TEXT2SQL_PORT}/v1/chat/completions"
    env["NLG_URL"] = f"http://localhost:{NLG_PORT}/v1/chat/completions"
    env["PYTHONPATH"] = str(_API_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    _procs["api"] = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app",
         "--host", "0.0.0.0", "--port", str(API_PORT)],
        cwd=str(_PROJECT_ROOT),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=env,
    )

    health_url = f"http://localhost:{API_PORT}/health"
    for _ in range(30):
        try:
            urllib.request.urlopen(health_url, timeout=1)
            return True
        except Exception:
            if _procs["api"].poll() is not None:
                console.print(f"  [red]✗[/red] api process exited unexpectedly")
                return False
            time.sleep(1)

    console.print(f"  [red]✗[/red] API health check timed out (30s)")
    return False


def start_stack() -> bool:
    """Start text2sql + nlg + api servers."""
    with console.status("[bold]Starting servers...[/bold]", spinner="dots"):
        ok = start_server("text2sql", TEXT2SQL_PORT, TEXT2SQL_MODEL)
        if not ok:
            console.print("  [red]✗[/red] text2sql server failed to start")
            return False
        console.print(f"  [green]✓[/green] text2sql on :{TEXT2SQL_PORT}")

        ok = start_server("nlg", NLG_PORT, NLG_MODEL,
                          extra_flags=["--reasoning-budget", "0"])
        if not ok:
            console.print("  [red]✗[/red] NLG server failed to start")
            return False
        console.print(f"  [green]✓[/green] nlg on :{NLG_PORT}")

        ok = start_api()
        if not ok:
            console.print("  [red]✗[/red] API server failed to start")
            return False
        console.print(f"  [green]✓[/green] api on :{API_PORT}")

    return True


def stop_stack() -> None:
    """Terminate all managed server processes."""
    for name, proc in _procs.items():
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    _procs.clear()


def dry_run() -> bool:
    """Send a test query to verify the pipeline works end-to-end."""
    console.print()
    with console.status("[bold]Verifying pipeline...[/bold]", spinner="dots"):
        try:
            import json
            payload = json.dumps({"question": "¿Cuántos productos distintos hay?"}).encode()
            req = urllib.request.Request(
                f"http://localhost:{API_PORT}/ask",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())

            sql = data.get("sql", "")
            rows = data.get("results", {}).get("rows", [])

            if rows:
                console.print(f"  [green]✓[/green] Pipeline OK — {rows[0]}")
                return True
            else:
                console.print(f"  [yellow]![/yellow] Query returned no rows: {sql[:60]}")
                return True

        except Exception as e:
            console.print(f"  [red]✗[/red] Dry run failed: {e}")
            return False
