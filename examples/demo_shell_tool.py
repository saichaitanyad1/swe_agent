import os
from agent_tools import ShellCommandTool


def main() -> None:
    tool = ShellCommandTool()
    result = tool.call({"command": "echo hello from shell tool"})
    print("Shell:", result["shell"])  # noqa: T201
    print("Return code:", result["return_code"])  # noqa: T201
    print("STDOUT:\n", result["stdout"])  # noqa: T201
    print("STDERR:\n", result["stderr"])  # noqa: T201

    # Windows PowerShell example
    if os.name == "nt":
        ps = tool.call({"command": "Get-Location", "shell": "powershell"})
        print("PowerShell Get-Location:\n", ps["stdout"])  # noqa: T201


if __name__ == "__main__":
    main()


