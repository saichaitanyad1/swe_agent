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
import re
import select
import shutil
import subprocess
import time
from dataclasses import dataclass
import threading
import uuid
from typing import Callable, Dict, Optional, Tuple, TypedDict

# Optional Windows ConPTY support via pywinpty
_HAS_PYWINPTY = False
try:  # pragma: no cover - only relevant on Windows
    if platform.system().lower() == "windows":
        from pywinpty import PtyProcessUnicode as _WinPtyProcess  # type: ignore
        _HAS_PYWINPTY = True
except Exception:
    _HAS_PYWINPTY = False


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


class _PersistentShellSession:
    """Maintain a persistent shell process to preserve state per session.

    Not thread-safe across sessions; a per-session lock serializes commands.
    """

    def __init__(self, shell_name: str, cwd: Optional[str], env: Optional[Dict[str, str]]):
        self.shell_name = shell_name
        self.cwd = cwd
        self.env = env
        self._lock = threading.Lock()
        self._process = None
        system = platform.system().lower()
        self._use_pty = system != "windows"
        self._use_conpty = system == "windows" and _HAS_PYWINPTY
        self._conpty = None
        self._pty_master_fd: Optional[int] = None
        self._pty_slave_fd: Optional[int] = None
        self._prompt_marker = f"__PS1__{uuid.uuid4().hex}__"
        self._rc_marker = f"__RC__{uuid.uuid4().hex}__"
        self._done_marker = f"__DONE__{uuid.uuid4().hex}__"
        self._process = self._start_process()
        self._closed = False
        self.last_used_ts: float = time.time()

    def _start_process(self) -> subprocess.Popen:
        shell = self.shell_name.lower()
        if shell in {"bash", "sh"}:
            # Prefer interactive bash for prompt support
            args = ["bash" if _is_executable_available("bash") else shell, "-i"]
        elif shell == "pwsh":
            args = ["pwsh", "-NoProfile", "-NonInteractive"]
        elif shell == "powershell":
            args = ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass"]
        elif shell in {"cmd", "cmd.exe"}:
            args = ["cmd", "/Q", "/D"]
        else:
            args = ["sh", "-i"]

        final_env = os.environ.copy()
        if self.env:
            final_env.update(self.env)
        # Set a recognizable prompt for POSIX shells so we can detect command completion
        if self._use_pty:
            final_env["PS1"] = self._prompt_marker + " "

        if self._use_conpty:
            # Windows ConPTY path
            try:
                # Spawn PowerShell/cmd attached to a ConPTY
                self._conpty = _WinPtyProcess.spawn(args, cwd=self.cwd or None, env=final_env)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("Failed to start Windows ConPTY shell") from exc
            # Return a dummy subprocess-like object placeholder; not used in ConPTY branch
            return subprocess.Popen(["cmd"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if self._use_pty:
            # Create PTY pair
            import pty
            master_fd, slave_fd = pty.openpty()
            self._pty_master_fd = master_fd
            self._pty_slave_fd = slave_fd
            # Spawn the shell attached to the slave end
            proc = subprocess.Popen(  # noqa: S603
                args,
                cwd=self.cwd or None,
                env=final_env,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                preexec_fn=os.setsid,  # make it a session leader
                bufsize=0,
                text=False,
                close_fds=True,
            )
            # Configure shell prompt and completion hooks without echoing our setup commands
            try:
                # temporarily disable echo
                os.write(master_fd, b"stty -echo\n")
                # set PS1 and PROMPT_COMMAND to emit RC and DONE markers before prompt
                setup_lines = (
                    f"export PS1='{self._prompt_marker} ';"
                    f"export PROMPT_COMMAND=\"printf '{self._rc_marker}%d\\n{self._done_marker}\\n' $?\"\n"
                )
                os.write(master_fd, setup_lines.encode("utf-8", errors="replace"))
                os.write(master_fd, b"stty echo\n")
                # drain initial output quickly
                start = time.perf_counter()
                buf = b""
                while time.perf_counter() - start < 0.2:
                    r, _, _ = select.select([master_fd], [], [], 0.05)
                    if not r:
                        continue
                    try:
                        chunk = os.read(master_fd, 4096)
                        if not chunk:
                            break
                        buf += chunk
                    except Exception:
                        break
            except Exception:
                pass
            return proc
        else:
            # Fallback to pipes on non-POSIX
            return subprocess.Popen(  # noqa: S603
                args,
                cwd=self.cwd or None,
                env=final_env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                if self._use_conpty and self._conpty is not None:
                    try:
                        self._conpty.close(True)
                    except Exception:
                        pass
                if self._process and self._process.poll() is not None:
                    # already exited
                    pass
                elif self._process:
                    self._process.kill()
            finally:
                self._closed = True
                self.last_used_ts = time.time()
                if self._use_pty:
                    try:
                        if self._pty_master_fd is not None:
                            os.close(self._pty_master_fd)
                    except Exception:
                        pass
                    try:
                        if self._pty_slave_fd is not None:
                            os.close(self._pty_slave_fd)
                    except Exception:
                        pass

    def is_idle(self, now_ts: float, idle_timeout_seconds: float) -> bool:
        if idle_timeout_seconds is None:
            return False
        return (now_ts - self.last_used_ts) >= idle_timeout_seconds

    def run(
        self,
        command: str,
        timeout_seconds: Optional[float],
        max_output_bytes: int,
        on_stdout: Optional[Callable[[str], None]] = None,
        on_stderr: Optional[Callable[[str], None]] = None,
    ) -> Tuple[int, str, str, bool]:
        """Run a command in the persistent shell.

        Returns: (return_code, stdout, stderr, timed_out)
        """
        with self._lock:
            if self._closed or self._process.poll() is not None:
                # Restart if needed
                self._process = self._start_process()
                self._closed = False

            stdout_accum: list[str] = []
            stderr_accum: list[str] = []
            timed_out = False
            exit_code: Optional[int] = None
            if self._use_conpty and self._conpty is not None:
                # Windows ConPTY streaming with RC/DONE markers
                stdout_accum = []
                buffer = ""
                saw_rc = False
                saw_done = False
                rc_value: Optional[int] = None
                start_time = time.perf_counter()
                total_bytes = 0

                shell = self.shell_name.lower()
                if shell in {"pwsh", "powershell"}:
                    cmd_text = (
                        f"{command}\r\n"
                        f"$code = if ($LASTEXITCODE -ne $null) {{ $LASTEXITCODE }} elseif ($?) {{ 0 }} else {{ 1 }};"
                        f"Write-Output '{self._rc_marker}' + $code;"
                        f"Write-Output '{self._done_marker}'\r\n"
                    )
                else:
                    cmd_text = f"{command} & echo {self._rc_marker}%ERRORLEVEL% & echo {self._done_marker}\r\n"
                try:
                    self._conpty.write(cmd_text)
                except Exception:
                    pass

                def reader() -> None:
                    nonlocal buffer, saw_rc, saw_done, rc_value, total_bytes
                    while True:
                        try:
                            chunk = self._conpty.read(4096)
                        except Exception:
                            break
                        if not chunk:
                            break
                        buffer += chunk
                        # Stream sanitized output
                        sanitized = re.sub(rf"{re.escape(self._rc_marker)}\d+\s*", "", chunk)
                        sanitized = sanitized.replace(self._done_marker, "")
                        if on_stdout is not None and sanitized:
                            try:
                                on_stdout(sanitized)
                            except Exception:
                                pass
                        enc = sanitized.encode("utf-8", errors="replace")
                        if total_bytes + len(enc) <= max_output_bytes:
                            stdout_accum.append(sanitized)
                            total_bytes += len(enc)
                        if not saw_rc:
                            m = re.search(re.escape(self._rc_marker) + r"(\d+)", buffer)
                            if m:
                                try:
                                    rc_value = int(m.group(1))
                                except Exception:
                                    rc_value = rc_value if rc_value is not None else 1
                                saw_rc = True
                        if not saw_done and self._done_marker in buffer:
                            saw_done = True

                t = threading.Thread(target=reader, daemon=True)
                t.start()
                while True:
                    if saw_rc and saw_done:
                        break
                    if timeout_seconds is not None and (time.perf_counter() - start_time) > timeout_seconds:
                        timed_out = True
                        try:
                            self._conpty.close(True)
                        except Exception:
                            pass
                        self._closed = True
                        break
                    time.sleep(0.05)
                t.join(timeout=0.2)

                self.last_used_ts = time.time()
                return (
                    rc_value if rc_value is not None else (124 if timed_out else 0),
                    "".join(stdout_accum),
                    "",
                    timed_out,
                )

            if self._use_pty and self._pty_master_fd is not None:
                # Write command with RC marker and a completion marker
                # Write the command and let PROMPT_COMMAND print RC and DONE once command finishes
                # Use a trailing newline only; avoid in-band markers to reduce interleaving
                cmd_text = f"{command}\n"
                os.write(self._pty_master_fd, cmd_text.encode("utf-8", errors="replace"))

                buffer = ""
                saw_rc = False
                saw_done = False
                saw_prompt = False
                rc_value: Optional[int] = None
                start_time = time.perf_counter()
                total_bytes = 0

                while True:
                    if timeout_seconds is not None and (time.perf_counter() - start_time) > timeout_seconds:
                        timed_out = True
                        try:
                            self._process.kill()
                        except Exception:
                            pass
                        self._closed = True
                        break
                    rlist, _, _ = select.select([self._pty_master_fd], [], [], 0.05)
                    if not rlist:
                        if saw_rc and saw_done:
                            break
                        continue
                    try:
                        chunk = os.read(self._pty_master_fd, 4096)
                    except Exception:
                        break
                    if not chunk:
                        break
                    text = chunk.decode("utf-8", errors="replace")
                    buffer += text
                    # Build a sanitized view for user-facing streaming (hide internal markers)
                    sanitized = text.replace(self._prompt_marker + " ", "")
                    sanitized = re.sub(rf"^{re.escape(self._rc_marker)}\d+\s*$", "", sanitized, flags=re.MULTILINE)
                    sanitized = re.sub(rf"^{re.escape(self._done_marker)}\s*$", "", sanitized, flags=re.MULTILINE)
                    # Stream to stdout callback using sanitized text
                    if on_stdout is not None and sanitized:
                        try:
                            on_stdout(sanitized)
                        except Exception:
                            pass
                    # Accumulate sanitized (treat as stdout); no separate stderr under PTY
                    encoded_sanitized = sanitized.encode("utf-8", errors="replace")
                    if total_bytes + len(encoded_sanitized) <= max_output_bytes:
                        stdout_accum.append(sanitized)
                        total_bytes += len(encoded_sanitized)

                    # Quick checks without waiting for full newlines
                    if not saw_rc:
                        m = re.search(re.escape(self._rc_marker) + r"(\d+)", buffer)
                        if m:
                            try:
                                rc_value = int(m.group(1))
                            except Exception:
                                rc_value = rc_value if rc_value is not None else 1
                            saw_rc = True
                    if not saw_done and self._done_marker in buffer:
                        saw_done = True
                    if not saw_prompt and self._prompt_marker in buffer:
                        saw_prompt = True

                    # Additionally scan per-line for robustness
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if not saw_rc and line.startswith(self._rc_marker):
                            try:
                                rc_str = line.split(self._rc_marker, 1)[1].strip()
                                rc_value = int(rc_str)
                            except Exception:
                                rc_value = rc_value if rc_value is not None else 1
                            saw_rc = True
                        if not saw_done and self._done_marker in line:
                            saw_done = True
                        if not saw_prompt and self._prompt_marker in line:
                            saw_prompt = True

                    if saw_done or (saw_rc and saw_prompt):
                        break

                self.last_used_ts = time.time()
                return (
                    rc_value if rc_value is not None else (124 if timed_out else 0),
                    "".join(stdout_accum),
                    "",
                    timed_out,
                )
            else:
                # Non-PTY path (Windows). Use previous pipe-based readers.
                end_token = f"__END__{uuid.uuid4().hex}__"
                exit_token = f"__EXIT__{uuid.uuid4().hex}__:"

                shell = self.shell_name.lower()
                if shell in {"bash", "sh"}:
                    wrapped = (
                        f"{command}\n"
                        f"__ret=$?; printf '{exit_token}%d\\n' \"$__ret\"; printf '{end_token}\\n'; printf '{end_token}\\n' 1>&2\n"
                    )
                elif shell == "pwsh":
                    wrapped = (
                        f"& {{ {command} }}; $code = if ($LASTEXITCODE -ne $null) {{ $LASTEXITCODE }} elseif ($?) {{ 0 }} else {{ 1 }}; "
                        f"Write-Output '{exit_token}' + $code; Write-Output '{end_token}'; [Console]::Error.WriteLine('{end_token}')\n"
                    )
                elif shell == "powershell":
                    wrapped = (
                        f"& {{ {command} }}; $code = if ($LASTEXITCODE -ne $null) {{ $LASTEXITCODE }} elseif ($?) {{ 0 }} else {{ 1 }}; "
                        f"Write-Output '{exit_token}' + $code; Write-Output '{end_token}'; [Console]::Error.WriteLine('{end_token}')\n"
                    )
                elif shell in {"cmd", "cmd.exe"}:
                    wrapped = (
                        f"{command} & echo {exit_token}%ERRORLEVEL% & echo {end_token} & echo {end_token} 1>&2\r\n"
                    )
                else:
                    wrapped = (
                        f"{command}\n"
                        f"__ret=$?; printf '{exit_token}%d\\n' \"$__ret\"; printf '{end_token}\\n'; printf '{end_token}\\n' 1>&2\n"
                    )

                assert self._process.stdin is not None
                assert self._process.stdout is not None
                assert self._process.stderr is not None

                self._process.stdin.write(wrapped)
                self._process.stdin.flush()

                # Reader threads
                exit_seen = threading.Event()
                end_stdout_seen = threading.Event()
                end_stderr_seen = threading.Event()

                def read_stream(
                    stream,
                    accum: list[str],
                    end_event: threading.Event,
                    *,
                    is_stdout: bool,
                ) -> None:
                    nonlocal exit_code
                    total_bytes2 = 0
                    while True:
                        line2 = stream.readline()
                        if line2 == "":
                            break
                        if line2.startswith(exit_token) or exit_token in line2:
                            try:
                                exit_code = int(line2.split(":", 1)[1].strip())
                            except Exception:
                                exit_code = exit_code if exit_code is not None else 1
                            exit_seen.set()
                            continue
                        if line2.strip() == end_token or (end_token in line2):
                            end_event.set()
                            continue
                        encoded2 = line2.encode("utf-8", errors="replace")
                        if total_bytes2 + len(encoded2) <= max_output_bytes:
                            accum.append(line2)
                            total_bytes2 += len(encoded2)
                            try:
                                if is_stdout and on_stdout is not None:
                                    on_stdout(line2)
                                if not is_stdout and on_stderr is not None:
                                    on_stderr(line2)
                            except Exception:
                                pass

                t_out = threading.Thread(
                    target=read_stream,
                    args=(self._process.stdout, stdout_accum, end_stdout_seen),
                    kwargs={"is_stdout": True},
                    daemon=True,
                )
                t_err = threading.Thread(
                    target=read_stream,
                    args=(self._process.stderr, stderr_accum, end_stderr_seen),
                    kwargs={"is_stdout": False},
                    daemon=True,
                )
                t_out.start()
                t_err.start()

                start_time2 = time.perf_counter()
                while True:
                    if end_stdout_seen.is_set() and end_stderr_seen.is_set() and exit_seen.is_set():
                        break
                    if timeout_seconds is not None and (time.perf_counter() - start_time2) > timeout_seconds:
                        timed_out = True
                        try:
                            self._process.kill()
                        except Exception:
                            pass
                        self._closed = True
                        break
                    time.sleep(0.01)

                t_out.join(timeout=0.1)
                t_err.join(timeout=0.1)

                self.last_used_ts = time.time()
                stdout_text = "".join(stdout_accum)
                stderr_text = "".join(stderr_accum)
                return (
                    exit_code if exit_code is not None else (124 if timed_out else 0),
                    stdout_text,
                    stderr_text,
                    timed_out,
                )

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
    enable_persistent_sessions: bool = True
    session_idle_timeout_seconds: Optional[float] = 1800.0  # 30 minutes

    # session_id -> _PersistentShellSession
    _sessions: Dict[str, _PersistentShellSession] = None  # type: ignore[assignment]
    _sessions_lock: threading.Lock = threading.Lock()

    def __post_init__(self) -> None:
        self.default_shell = _detect_default_shell(prefer_bash=self.prefer_bash)
        if self._sessions is None:
            self._sessions = {}
        # Start janitor thread once
        self._ensure_janitor_thread()

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
            - session_id (str): optional session identifier for persistent shell state
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

        session_id_raw = arguments.get("session_id")
        session_id = str(session_id_raw).strip() if session_id_raw else None

        if self.enable_persistent_sessions and session_id:
            result = self._execute_persistent(
                session_id=session_id,
                command=command_value,
                shell_name=shell_name,
                cwd=cwd_value,
                env=env_overrides,
                timeout_seconds=timeout_seconds,
            )
        else:
            result = self._execute(command_value, shell_name, cwd_value, env_overrides, timeout_seconds)
        return result

    # --- Persistent session management API ---
    def close_session(self, session_id: str) -> bool:
        """Close and remove a session. Returns True if a session was closed."""
        assert self._sessions is not None
        with self._sessions_lock:
            sess = self._sessions.pop(session_id, None)
        if sess is not None:
            try:
                sess.close()
            except Exception:
                pass
            return True
        return False

    def list_sessions(self) -> Dict[str, Dict[str, object]]:
        """Return lightweight metadata about active sessions."""
        assert self._sessions is not None
        with self._sessions_lock:
            info = {}
            now_ts = time.time()
            for sid, sess in self._sessions.items():
                info[sid] = {
                    "shell": sess.shell_name,
                    "cwd": sess.cwd,
                    "last_used_seconds_ago": now_ts - sess.last_used_ts,
                }
            return info

    _janitor_started: bool = False

    def _ensure_janitor_thread(self) -> None:
        if self._janitor_started:
            return
        self._janitor_started = True

        def _janitor():
            while True:
                time.sleep(30.0)
                timeout = self.session_idle_timeout_seconds
                if timeout is None:
                    continue
                now_ts = time.time()
                to_close = []
                with self._sessions_lock:
                    for sid, sess in list(self._sessions.items()):
                        if sess.is_idle(now_ts, timeout):
                            to_close.append(sid)
                for sid in to_close:
                    try:
                        self.close_session(sid)
                    except Exception:
                        pass

        t = threading.Thread(target=_janitor, name="ShellCommandTool-Janitor", daemon=True)
        t.start()

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

    def _execute_stream(
        self,
        command: str,
        shell_name: str,
        cwd: Optional[str],
        env: Optional[Dict[str, str]],
        timeout_seconds: Optional[float],
        on_stdout: Optional[Callable[[str], None]] = None,
        on_stderr: Optional[Callable[[str], None]] = None,
    ) -> ShellExecutionResult:
        invocation = _build_shell_invocation(shell_name, command)

        final_env = os.environ.copy()
        if env:
            final_env.update(env)

        process = subprocess.Popen(  # noqa: S603
            invocation["args"],
            cwd=cwd or None,
            env=final_env,
            shell=bool(invocation["shell"]),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        stdout_accum: list[str] = []
        stderr_accum: list[str] = []
        start_time = time.perf_counter()
        timed_out = False

        def read_stream(stream, accum: list[str], is_stdout: bool) -> None:
            total_bytes = 0
            line_buffer: list[str] = []
            while True:
                ch = stream.read(1)
                if ch == "":
                    if line_buffer:
                        line = "".join(line_buffer)
                        encoded = line.encode("utf-8", errors="replace")
                        if total_bytes + len(encoded) <= self.max_output_bytes:
                            accum.append(line)
                            total_bytes += len(encoded)
                    break
                try:
                    if is_stdout and on_stdout is not None:
                        on_stdout(ch)
                    if not is_stdout and on_stderr is not None:
                        on_stderr(ch)
                except Exception:
                    pass
                line_buffer.append(ch)
                if ch == "\n" or ch == "\r":
                    line = "".join(line_buffer)
                    line_buffer.clear()
                    encoded = line.encode("utf-8", errors="replace")
                    if total_bytes + len(encoded) <= self.max_output_bytes:
                        accum.append(line)
                        total_bytes += len(encoded)

        assert process.stdout is not None
        assert process.stderr is not None
        t_out = threading.Thread(target=read_stream, args=(process.stdout, stdout_accum, True), daemon=True)
        t_err = threading.Thread(target=read_stream, args=(process.stderr, stderr_accum, False), daemon=True)
        t_out.start()
        t_err.start()

        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()

        t_out.join()
        t_err.join()

        end_time = time.perf_counter()
        stdout_text = "".join(stdout_accum)
        stderr_text = "".join(stderr_accum)

        truncated = (
            len(stdout_text.encode("utf-8")) > self.max_output_bytes
            or len(stderr_text.encode("utf-8")) > self.max_output_bytes
        )

        return {
            "command": command,
            "shell": shell_name,
            "return_code": process.returncode or (124 if timed_out else 0),
            "stdout": stdout_text,
            "stderr": stderr_text,
            "cwd": os.path.abspath(cwd) if cwd else os.getcwd(),
            "duration_seconds": end_time - start_time,
            "timed_out": timed_out,
            "truncated": truncated,
        }

    def _get_or_create_session(
        self, session_id: str, shell_name: str, cwd: Optional[str], env: Optional[Dict[str, str]]
    ) -> _PersistentShellSession:
        assert self._sessions is not None
        with self._sessions_lock:
            sess = self._sessions.get(session_id)
            if sess is None or sess.shell_name.lower() != shell_name.lower() or sess.cwd != cwd:
                # Replace session if shell/cwd changed to avoid undefined states
                if sess is not None:
                    try:
                        sess.close()
                    except Exception:
                        pass
                sess = _PersistentShellSession(shell_name=shell_name, cwd=cwd, env=env)
                self._sessions[session_id] = sess
        return sess

    def _execute_persistent(
        self,
        *,
        session_id: str,
        command: str,
        shell_name: str,
        cwd: Optional[str],
        env: Optional[Dict[str, str]],
        timeout_seconds: Optional[float],
    ) -> ShellExecutionResult:
        session = self._get_or_create_session(session_id=session_id, shell_name=shell_name, cwd=cwd, env=env)

        start_time = time.perf_counter()
        return_code, stdout_text, stderr_text, timed_out = session.run(
            command=command,
            timeout_seconds=timeout_seconds,
            max_output_bytes=self.max_output_bytes,
        )
        end_time = time.perf_counter()

        truncated = (
            len(stdout_text.encode("utf-8")) > self.max_output_bytes
            or len(stderr_text.encode("utf-8")) > self.max_output_bytes
        )

        return {
            "command": command,
            "shell": shell_name,
            "return_code": return_code,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "cwd": os.path.abspath(cwd) if cwd else os.getcwd(),
            "duration_seconds": end_time - start_time,
            "timed_out": timed_out,
            "truncated": truncated,
        }

    def _execute_persistent_stream(
        self,
        *,
        session_id: str,
        command: str,
        shell_name: str,
        cwd: Optional[str],
        env: Optional[Dict[str, str]],
        timeout_seconds: Optional[float],
        on_stdout: Optional[Callable[[str], None]] = None,
        on_stderr: Optional[Callable[[str], None]] = None,
    ) -> ShellExecutionResult:
        session = self._get_or_create_session(session_id=session_id, shell_name=shell_name, cwd=cwd, env=env)

        start_time = time.perf_counter()
        return_code, stdout_text, stderr_text, timed_out = session.run(
            command=command,
            timeout_seconds=timeout_seconds,
            max_output_bytes=self.max_output_bytes,
            on_stdout=on_stdout,
            on_stderr=on_stderr,
        )
        end_time = time.perf_counter()

        truncated = (
            len(stdout_text.encode("utf-8")) > self.max_output_bytes
            or len(stderr_text.encode("utf-8")) > self.max_output_bytes
        )

        return {
            "command": command,
            "shell": shell_name,
            "return_code": return_code,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "cwd": os.path.abspath(cwd) if cwd else os.getcwd(),
            "duration_seconds": end_time - start_time,
            "timed_out": timed_out,
            "truncated": truncated,
        }

    # Public streaming wrapper
    def call_stream(
        self,
        arguments: Dict[str, object],
        *,
        on_stdout: Optional[Callable[[str], None]] = None,
        on_stderr: Optional[Callable[[str], None]] = None,
    ) -> ShellExecutionResult:
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
            timeout_seconds = float(timeout_value_raw)  # may raise

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

        session_id_raw = arguments.get("session_id")
        session_id = str(session_id_raw).strip() if session_id_raw else None

        if self.enable_persistent_sessions and session_id:
            return self._execute_persistent_stream(
                session_id=session_id,
                command=command_value,
                shell_name=shell_name,
                cwd=cwd_value,
                env=env_overrides,
                timeout_seconds=timeout_seconds,
                on_stdout=on_stdout,
                on_stderr=on_stderr,
            )
        return self._execute_stream(
            command=command_value,
            shell_name=shell_name,
            cwd=cwd_value,
            env=env_overrides,
            timeout_seconds=timeout_seconds,
            on_stdout=on_stdout,
            on_stderr=on_stderr,
        )


