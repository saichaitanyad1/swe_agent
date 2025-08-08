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


