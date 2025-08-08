## Agent Tools

Minimal Python tools for agent orchestration.

### ShellCommandTool

Cross-platform tool that executes arbitrary shell commands with a function-calling interface suitable for LLM agents.

Usage:

```python
from agent_tools import ShellCommandTool

tool = ShellCommandTool()
result = tool.call({"command": "echo hello"})
print(result["stdout"])  # "hello\n"
```

Arguments supported via `call`:
- `command` (str, required): command to execute
- `cwd` (str, optional): working directory
- `timeout_seconds` (float, optional): timeout
- `env` (dict, optional): environment overrides
- `shell` ("bash"|"sh"|"pwsh"|"powershell"|"cmd", optional): shell override

Default shell detection:
- Prefers `bash` if available
- On Windows, prefers `pwsh`, then `powershell`, then `cmd`
- Otherwise uses `sh`

The result contains `stdout`, `stderr`, `return_code`, `duration_seconds`, and metadata flags.


### Persistent sessions and streaming

- Use `session_id` to persist shell state (cwd, env) across calls.
- Use `call_stream` for terminal-like rolling logs.

Non-streaming example:

```python
from agent_tools import ShellCommandTool

tool = ShellCommandTool()
res = tool.call({"command": "echo hi", "session_id": "s1"})
print(res["stdout"])  # prints after completion
```

Streaming example (recommended for long commands):

```python
from agent_tools import ShellCommandTool

tool = ShellCommandTool()

def out(s: str) -> None:
    print(s, end="")  # forward to your UI/log

tool.call_stream(
    {"command": "mvn -q -DskipTests clean install", "session_id": "s1"},
    on_stdout=out,
    on_stderr=out,
)
```

Notes:
- POSIX uses a PTY for true terminal behavior; Windows uses ConPTY when `pywinpty` is installed (falls back otherwise).
- Use `close_session(session_id)` to end a session; sessions auto-expire after `session_idle_timeout_seconds`.


