"""Keep a Windows deployment on pushed covid-files code, then serve it.

Run this under the deployment PC's scheduled-task account. It updates tracked code
only; the ignored dataset and llm.env remain on that PC. The task must use a
dedicated virtual environment containing the project's requirements.
"""
from __future__ import annotations

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
BRANCH = "covid-files"
LOGGER = logging.getLogger("local_sync")


def git(*args: str, timeout: int = 90, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update({"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"})
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                            text=True, env=env, timeout=timeout, check=False)
    if check and result.returncode:
        raise RuntimeError(f"git {args[0]} failed: {(result.stderr or result.stdout).strip()[:300]}")
    return result


def local_environment() -> dict[str, str]:
    """Read only literal PUBMED_* assignments, as start_all.ps1 does."""
    env = os.environ.copy()
    source = ROOT / "llm.env"
    if not source.is_file():
        return env
    for number, line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"\s*(PUBMED_[A-Z0-9_]+)\s*=(.*)", line)
        if not match:
            LOGGER.warning("Ignored invalid llm.env entry on line %s", number)
            continue
        name, value = match.groups()
        if name in env:
            continue
        value = value.strip()
        if value.startswith(("'", '"')):
            if len(value) < 2 or value[-1] != value[0]:
                LOGGER.warning("Ignored malformed llm.env entry on line %s", number)
                continue
            value = value[1:-1]
        env[name] = value
    return env


def dataset_ready(manifest: Path) -> bool:
    if not manifest.is_file():
        return False
    try:
        cfg = json.loads(manifest.read_text(encoding="utf-8"))
        if cfg.get("status") != "ready" or cfg.get("pilot"):
            return False
        for key in ("store", "index"):
            target = Path(cfg[key])
            if not target.is_absolute():
                target = manifest.parent / target
            if not target.is_dir():
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def update_available(fetch: bool = True) -> tuple[str, str] | None:
    branch = git("branch", "--show-current").stdout.strip()
    if branch != BRANCH:
        raise RuntimeError(f"Expected branch {BRANCH}; found {branch or 'detached HEAD'}")
    if git("status", "--porcelain", "--untracked-files=no").stdout.strip():
        raise RuntimeError("Tracked files have local changes; commit or review them before syncing")
    if fetch:
        git("fetch", "--no-tags", "origin",
            f"refs/heads/{BRANCH}:refs/remotes/origin/{BRANCH}", timeout=120)
    current = git("rev-parse", "HEAD").stdout.strip()
    target = git("rev-parse", f"refs/remotes/origin/{BRANCH}").stdout.strip()
    if current == target:
        return None
    if git("merge-base", "--is-ancestor", current, target, check=False).returncode:
        raise RuntimeError("Local and remote branches diverged; refusing to overwrite either one")
    changed = git("diff", "--name-only", current, target).stdout.splitlines()
    if "requirements.txt" in changed:
        raise RuntimeError("New Python requirements need a supervised install; code was not changed")
    return current, target


def port_busy(port: int) -> bool:
    with socket.socket() as connection:
        connection.settimeout(1)
        return connection.connect_ex(("127.0.0.1", port)) == 0


def prepare_performance(manifest: Path, env: dict[str, str], output) -> None:
    if env.get("PUBMED_PREPARE_PERFORMANCE", "1") == "0":
        return
    LOGGER.info("Checking optional performance indexes; first preparation may take several minutes")
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "pipeline" / "manage.py"), "performance",
             "--dataset", str(manifest)], cwd=ROOT, env=env, stdout=output,
            stderr=subprocess.STDOUT, timeout=1800,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False)
        if result.returncode:
            LOGGER.warning("Performance preparation incomplete; serving with original-data fallback")
    except (OSError, subprocess.TimeoutExpired):
        LOGGER.warning("Performance preparation unavailable; serving with original-data fallback")


def start_server(manifest: Path, port: int) -> subprocess.Popen[bytes]:
    if port_busy(port):
        raise RuntimeError(f"Port {port} is already in use; stop the other server first")
    logs = ROOT / "ops" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    output = (logs / "server.log").open("ab")
    try:
        env = local_environment()
        prepare_performance(manifest, env, output)
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "pipeline" / "manage.py"), "serve",
             "--dataset", str(manifest), "--port", str(port)],
            cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    finally:
        output.close()
    LOGGER.info("Started application process %s on http://127.0.0.1:%s", process.pid, port)
    return process


def stop_server(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    LOGGER.info("Stopping owned application process %s", process.pid)
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def configure_logging() -> None:
    logs = ROOT / "ops" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(),
                    RotatingFileHandler(logs / "sync.log", maxBytes=5_000_000,
                                        backupCount=3, encoding="utf-8")):
        handler.setFormatter(fmt)
        LOGGER.addHandler(handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "covid-files" / "dataset.json")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--interval", type=int, default=300,
                        help="Seconds between GitHub checks (minimum 60)")
    parser.add_argument("--check", action="store_true",
                        help="Report a pending update without changing code or starting the server")
    parser.add_argument("--offline", action="store_true",
                        help="With --check, inspect already-fetched Git refs without network access")
    args = parser.parse_args()
    configure_logging()
    if args.offline and not args.check:
        parser.error("--offline requires --check")
    if args.interval < 60 or not 1 <= args.port <= 65535:
        parser.error("interval must be at least 60 seconds and port must be 1-65535")
    manifest = args.dataset.resolve()
    if args.check:
        try:
            pending = update_available(fetch=not args.offline)
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
            LOGGER.error("Update check failed: %s", error)
            return 1
        if pending:
            LOGGER.info("Ready to update %s -> %s", pending[0][:12], pending[1][:12])
        else:
            LOGGER.info("Up to date at %s", git("rev-parse", "--short", "HEAD").stdout.strip())
        ready = dataset_ready(manifest)
        LOGGER.info("Local dataset ready: %s", ready)
        return 0 if ready else 2
    if not dataset_ready(manifest):
        LOGGER.error("Verified local dataset is missing or incomplete: %s", manifest)
        return 1
    if port_busy(args.port):
        LOGGER.error("Port %s is occupied; stop the other server before starting this supervisor", args.port)
        return 1
    process = None
    last_check = 0.0
    try:
        process = start_server(manifest, args.port)
        while True:
            if process.poll() is not None:
                LOGGER.error("Application exited with status %s; retrying in 30 seconds", process.returncode)
                time.sleep(30)
                process = start_server(manifest, args.port)
            if time.monotonic() - last_check >= args.interval:
                last_check = time.monotonic()
                try:
                    pending = update_available()
                    if pending:
                        LOGGER.info("Applying pushed code %s -> %s", pending[0][:12], pending[1][:12])
                        stop_server(process)
                        process = None
                        git("merge", "--ff-only", f"refs/remotes/origin/{BRANCH}")
                        process = start_server(manifest, args.port)
                except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
                    LOGGER.error("Update skipped: %s", error)
                    if process is None:
                        process = start_server(manifest, args.port)
            time.sleep(10)
    except KeyboardInterrupt:
        LOGGER.info("Supervisor stopped by operator")
    finally:
        stop_server(process)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
