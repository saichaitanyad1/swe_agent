import os
from agent_tools import ShellCommandTool


def main() -> None:
    tool = ShellCommandTool()
    session_id = "demo-session-1"
    result = tool.call({"command": "echo hello from shell tool", "session_id": session_id})
    print("Shell:", result["shell"])  # noqa: T201
    print("Return code:", result["return_code"])  # noqa: T201
    print("STDOUT:\n", result["stdout"])  # noqa: T201
    print("STDERR:\n", result["stderr"])  # noqa: T201
    result = tool.call({
        "command": "git clone --depth 1 --single-branch --quiet https://github.com/javabycode/spring-boot-maven-example-helloworld.git",
        "session_id": session_id,
        "timeout_seconds": 180,
    })
    print("Shell:", result["shell"])  # noqa: T201
    print("Return code:", result["return_code"])  # noqa: T201
    print("STDOUT:\n", result["stdout"])  # noqa: T201
    print("STDERR:\n", result["stderr"])

    result = tool.call({"command": "cd spring-boot-maven-example-helloworld", "session_id": session_id})
    print("Shell:", result["shell"])  # noqa: T201
    print("Return code:", result["return_code"])  # noqa: T201
    print("STDOUT:\n", result["stdout"])  # noqa: T201
    print("STDERR:\n", result["stderr"])

    result = tool.call({
        "command": "mvn -q -DskipTests clean install",
        "session_id": session_id,
        "timeout_seconds": 300,
    })
    print("Shell:", result["shell"])  # noqa: T201
    print("Return code:", result["return_code"])  # noqa: T201
    print("STDOUT:\n", result["stdout"])  # noqa: T201
    print("STDERR:\n", result["stderr"])

    # Windows PowerShell example
    if os.name == "nt":
        ps = tool.call({"command": "Get-Location", "shell": "powershell", "session_id": session_id})
        print("PowerShell Get-Location:\n", ps["stdout"])  # noqa: T201


if __name__ == "__main__":
    main()


