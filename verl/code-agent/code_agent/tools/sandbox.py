"""Lightweight sandbox execution tool used during agent rollout."""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from pathlib import Path


def _workspace_cwd(tmpdir: str, cwd: str | Path | None = None) -> str:
    if cwd is not None:
        path = Path(cwd).expanduser()
        if path.is_dir():
            return str(path.resolve())

    for key in ["CODE_AGENT_WORKSPACE_PATH", "CODE_AGENT_REPO_ROOT", "CODE_AGENT_REPO_DIR"]:
        value = os.environ.get(key)
        if value and Path(value).expanduser().is_dir():
            return str(Path(value).expanduser().resolve())
    return tmpdir


def run_code(command: str, timeout_sec: int | None = None, cwd: str | Path | None = None) -> str:
    """Execute a lightweight sandbox command and return a compact observation."""
    timeout = timeout_sec or int(os.environ.get("CODE_AGENT_TOOL_TIMEOUT", "10"))
    command = "" if command is None else str(command).strip()
    if not command:
        return "ERROR: empty command"

    try:
        argv = shlex.split(command)
    except ValueError as exc:
        return f"ERROR: failed to parse command: {exc}"
    if not argv:
        return "ERROR: empty command"

    with tempfile.TemporaryDirectory(prefix="code_agent_") as tmpdir:
        cwd = _workspace_cwd(tmpdir, cwd=cwd)

        try:
            proc = subprocess.run(
                argv,
                cwd=cwd,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"TIMEOUT: command execution exceeded {timeout}s"
        except Exception as exc:
            return f"ERROR: failed to execute command: {exc}"

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    parts = [f"exit_code={proc.returncode}"]
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    return "\n\n".join(parts)
