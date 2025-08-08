from agent_tools import ShellCommandTool
import os


def main() -> None:
    tool = ShellCommandTool()
    session_id = "demo-session-1"

    def out_stdout(ch: str) -> None:
        print(ch, end="", flush=True)

    def out_stderr(ch: str) -> None:
        print(ch, end="", flush=True)

    print("== Echo test ==")
    tool.call_stream({"command": "echo hello from shell tool", "session_id": session_id},
                     on_stdout=out_stdout, on_stderr=out_stderr)
    print()

    print("== Git clone (shallow) ==")
    tool.call_stream({
        "command": "git clone --depth 1 --single-branch --quiet https://github.com/javabycode/spring-boot-maven-example-helloworld.git",
        "session_id": session_id,
        "timeout_seconds": 180,
    }, on_stdout=out_stdout, on_stderr=out_stderr)
    print()

    print("== cd project ==")
    tool.call_stream({"command": "cd spring-boot-maven-example-helloworld", "session_id": session_id},
                     on_stdout=out_stdout, on_stderr=out_stderr)
    print()

    print("== Maven build (quiet, skip tests) ==")
    tool.call_stream({
        "command": "mvn -q -DskipTests clean install",
        "session_id": session_id,
        "timeout_seconds": 300,
    }, on_stdout=out_stdout, on_stderr=out_stderr)
    print()

    if os.name == "nt":
        print("== PowerShell Get-Location ==")
        tool.call_stream({"command": "Get-Location", "shell": "powershell", "session_id": session_id},
                         on_stdout=out_stdout, on_stderr=out_stderr)


if __name__ == "__main__":
    main()


