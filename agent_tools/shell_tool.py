"""
Cross-platform shell command tool with a function-calling interface.

This module provides `ShellCommandTool`, a minimal dependency-free utility
that agents can call to execute arbitrary shell commands on macOS, Linux,
and Windows. It exposes a function-calling style schema and a simple `call`
method that accepts arguments and returns a structured result.

Safety note: This executes arbitrary commands. Only use in trusted contexts.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, Optional, TypedDict


class ShellExecutionResult(TypedDict):
    """Structured result of a shell command execution."""

    command: str
    shell: str
    return_code: int
    stdout: str
    stderr: str
    cwd: str
    duration_seconds: float
    timed_out: bool
    truncated: bool


def _is_executable_available(executable_name: str) -> bool:
    return shutil.which(executable_name) is not None


def _detect_default_shell(prefer_bash: bool = True) -> str:
    """Detect a reasonable default shell for the current platform.

    Priority:
    - If prefer_bash and `bash` exists, use `bash` with `-lc` (macOS/Linux, or Windows with Git Bash/WSL)
    - On Windows: prefer PowerShell (pwsh or powershell) if available, otherwise fallback to `cmd`
    - On POSIX without bash: use `sh`
    """

    system = platform.system().lower()
    if prefer_bash and _is_executable_available("bash"):
        return "bash"

    if system == "windows":
        if _is_executable_available("pwsh"):
            return "pwsh"
        if _is_executable_available("powershell"):
            return "powershell"
        return "cmd"

    # POSIX fallback
    return "sh"


def _build_shell_invocation(shell_name: str, command: str) -> Dict[str, object]:
    """Build subprocess invocation (args and shell flag) for a given shell.

    We avoid `shell=True` and prefer explicit shell executable invocation
    for predictable quoting and cross-platform behavior.
    """

    shell_name_lower = shell_name.lower()
    if shell_name_lower in {"bash", "sh"}:
        # Run via: <shell> -lc "command"
        return {"args": [shell_name_lower, "-lc", command], "shell": False}

    if shell_name_lower == "pwsh":
        # PowerShell 7+
        return {
            "args": [
                "pwsh",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
            ],
            "shell": False,
        }

    if shell_name_lower == "powershell":
        # Windows PowerShell
        return {
            "args": [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                command,
            ],
            "shell": False,
        }

    if shell_name_lower in {"cmd", "cmd.exe"}:
        # Use cmd.exe with recommended flags. `/d` disables AutoRun, `/s` preserves quoting
        return {"args": ["cmd", "/d", "/s", "/c", command], "shell": False}

    # Unknown shell, fallback to POSIX-like `sh`
    return {"args": ["sh", "-lc", command], "shell": False}


@dataclass
class ShellCommandTool:
    """A simple, cross-platform shell execution tool for agents.

    Example usage:
        tool = ShellCommandTool()
        result = tool.call({"command": "echo hello"})
        print(result["stdout"])  # => "hello\n"

    The `tool_schema` attribute can be advertised to LLMs for function calling.
    """

    prefer_bash: bool = True
    max_output_bytes: int = 2_000_000  # Prevent unbounded memory usage

    def __post_init__(self) -> None:
        self.default_shell = _detect_default_shell(prefer_bash=self.prefer_bash)

        self.tool_schema = {
            "name": "run_shell_command",
            "description": (
                "Execute an arbitrary shell command. Supports macOS/Linux (bash/sh) and "
                "Windows (PowerShell/cmd). Prefer non-interactive commands."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The exact shell command to execute.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Optional working directory to execute within.",
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "description": "Optional timeout in seconds; process is terminated on timeout.",
                    },
                    "env": {
                        "type": "object",
                        "additionalProperties": {"type": ["string", "number", "boolean"]},
                        "description": "Optional environment variable overrides.",
                    },
                    "shell": {
                        "type": "string",
                        "enum": ["bash", "sh", "pwsh", "powershell", "cmd"],
                        "description": "Optional shell override; defaults to an auto-detected shell.",
                    },
                },
                "required": ["command"],
            },
        }

    def call(self, arguments: Dict[str, object]) -> ShellExecutionResult:
        """Function-calling style entry point.

        Arguments schema:
            - command (str): required shell command to run
            - cwd (str): optional working directory
            - timeout_seconds (float|int): optional timeout seconds
            - env (dict[str, str|int|float|bool]): optional env overrides
            - shell (str): optional shell override ("bash"|"sh"|"pwsh"|"powershell"|"cmd")
        """

        command_value = str(arguments.get("command", "")).strip()
        if not command_value:
            raise ValueError("'command' must be a non-empty string")

        cwd_value_raw = arguments.get("cwd")
        cwd_value = str(cwd_value_raw) if cwd_value_raw is not None else None
        if cwd_value and not os.path.isdir(cwd_value):
            raise FileNotFoundError(f"cwd does not exist or is not a directory: {cwd_value}")

        timeout_value_raw = arguments.get("timeout_seconds")
        timeout_seconds: Optional[float]
        if timeout_value_raw is None:
            timeout_seconds = None
        else:
            try:
                timeout_seconds = float(timeout_value_raw)  # type: ignore[assignment]
            except Exception as exc:  # noqa: BLE001 - keep broad for schema inputs
                raise ValueError("'timeout_seconds' must be a number") from exc

        env_value_raw = arguments.get("env")
        env_overrides: Optional[Dict[str, str]] = None
        if env_value_raw is not None:
            if not isinstance(env_value_raw, dict):
                raise ValueError("'env' must be an object/dict of key-value pairs")
            env_overrides = {str(k): str(v) for k, v in env_value_raw.items()}

        shell_override_raw = arguments.get("shell")
        allowed_shells = {"bash", "sh", "pwsh", "powershell", "cmd"}
        shell_name = (
            str(shell_override_raw).lower()
            if shell_override_raw in allowed_shells
            else self.default_shell
        )

        return self._execute(command_value, shell_name, cwd_value, env_overrides, timeout_seconds)

    # Public helper for non function-calling usage
    def execute(
        self,
        command: str,
        *,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_seconds: Optional[float] = None,
        shell: Optional[str] = None,
    ) -> ShellExecutionResult:
        shell_name = (shell or self.default_shell).lower()
        return self._execute(command, shell_name, cwd, env, timeout_seconds)

    def _execute(
        self,
        command: str,
        shell_name: str,
        cwd: Optional[str],
        env: Optional[Dict[str, str]],
        timeout_seconds: Optional[float],
    ) -> ShellExecutionResult:
        invocation = _build_shell_invocation(shell_name, command)

        # Prepare environment
        final_env = os.environ.copy()
        if env:
            final_env.update(env)

        start_time = time.perf_counter()
        process = subprocess.Popen(  # noqa: S603 - intentional external command
            invocation["args"],
            cwd=cwd or None,
            env=final_env,
            shell=bool(invocation["shell"]),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        timed_out = False
        try:
            stdout_bytes, stderr_bytes = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            stdout_bytes, stderr_bytes = process.communicate()
        end_time = time.perf_counter()

        # Decode with replacement to be robust against mixed encodings
        stdout_decoded = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        stderr_decoded = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

        # Truncate overly large output to keep memory bounded
        truncated = False
        if len(stdout_decoded.encode("utf-8")) > self.max_output_bytes:
            stdout_decoded = stdout_decoded.encode("utf-8")[: self.max_output_bytes].decode(
                "utf-8", errors="replace"
            )
            truncated = True
        if len(stderr_decoded.encode("utf-8")) > self.max_output_bytes:
            stderr_decoded = stderr_decoded.encode("utf-8")[: self.max_output_bytes].decode(
                "utf-8", errors="replace"
            )
            truncated = True

        result: ShellExecutionResult = {
            "command": command,
            "shell": shell_name,
            "return_code": process.returncode,
            "stdout": stdout_decoded,
            "stderr": stderr_decoded,
            "cwd": os.path.abspath(cwd) if cwd else os.getcwd(),
            "duration_seconds": end_time - start_time,
            "timed_out": timed_out,
            "truncated": truncated,
        }

        return result


