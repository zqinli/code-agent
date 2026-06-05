"""Lightweight sandbox execution tool used during agent rollout."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


def run_code(code: str, timeout_sec: int | None = None) -> str:
    """Execute Python code in a temporary directory and return a compact observation."""
    timeout = timeout_sec or int(os.environ.get("CODE_AGENT_TOOL_TIMEOUT", "10"))
    code = "" if code is None else str(code)
    if not code.strip():
        return "ERROR: empty code"

    with tempfile.TemporaryDirectory(prefix="code_agent_") as tmpdir:
        script_path = Path(tmpdir) / "main.py"
        script_path.write_text(code, encoding="utf-8")

        try:
            proc = subprocess.run(
                ["python", str(script_path)],
                cwd=tmpdir,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"TIMEOUT: code execution exceeded {timeout}s"
        except Exception as exc:
            return f"ERROR: failed to execute code: {exc}"

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    parts = [f"exit_code={proc.returncode}"]
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    return "\n\n".join(parts)

