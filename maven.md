# Google ADK Multi-Agent Orchestrator Example (Maven Build)

This example shows how to build a **multi-agent orchestrator** using Google ADK for Maven builds.  

We define:
1. **Builder Agent** → runs Maven builds and summarizes results.  
2. **Log Agent** → fetches build logs on demand.  
3. **Orchestrator** → coordinates agents.  

---

## 1. Define Tools

```python
from adk.agent import tool, expose
import subprocess, uuid, os

LOG_DIR = "/tmp/maven_build_logs"
os.makedirs(LOG_DIR, exist_ok=True)

@tool
def maven_build(project_path: str) -> dict:
    """
    Runs Maven build, exposes only summary + exit code to LLM.
    Full logs stored externally.
    """
    build_id = str(uuid.uuid4())
    log_file = os.path.join(LOG_DIR, f"{build_id}.log")

    with open(log_file, "w") as f:
        process = subprocess.Popen(
            ["mvn", "clean", "install"],
            cwd=project_path,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        exit_code = process.wait()

    with open(log_file, "r") as f:
        log_content = f.read()

    errors = [line for line in log_content.splitlines() if "[ERROR]" in line]

    summary = (
        f"Maven build {'succeeded' if exit_code == 0 else 'failed'}."
        f" Exit code {exit_code}. Errors found: {len(errors)}."
    )

    return {
        "summary": expose(summary),      # visible to LLM
        "exit_code": expose(exit_code),  # visible to LLM
        "build_id": build_id,            # kept in orchestrator state
        "log_file": log_file             # kept in state, not LLM
    }


@tool
def fetch_build_log(build_id: str, max_lines: int = 50) -> dict:
    """
    Fetches last N lines from Maven build log. 
    Exposes only snippet to LLM.
    """
    log_file = os.path.join(LOG_DIR, f"{build_id}.log")
    if not os.path.exists(log_file):
        return {"error": expose("Log not found.")}

    with open(log_file, "r") as f:
        lines = f.readlines()

    snippet = "".join(lines[-max_lines:])

    return {
        "log_snippet": expose(snippet)  # 👈 only snippet goes to LLM
    }
```

---

## 2. Define Agents

```python
from adk.agent import Agent

builder_agent = Agent(
    name="BuilderAgent",
    description="Runs Maven builds and summarizes results.",
    tools=[maven_build]
)

log_agent = Agent(
    name="LogAgent",
    description="Fetches Maven logs on demand.",
    tools=[fetch_build_log]
)
```

---

## 3. Orchestrator Setup

```python
from adk.orchestrator import Orchestrator

orchestrator = Orchestrator(
    agents=[builder_agent, log_agent],
    system_prompt="""
    You are an orchestrator for Maven builds.
    - Use BuilderAgent to run builds.
    - If user requests logs or details, call LogAgent with the build_id.
    - Do not include entire logs in responses unless explicitly asked.
    """
)
```

---

## 4. Example Run

```python
# User asks to build a project
response = orchestrator.run("Build the project at /my/project")
print(response)
# Output: "Maven build failed. Exit code 1. Errors found: 2."

# User asks for more details
response = orchestrator.run("Show me the last 20 lines of the build log")
print(response)
# Output: "Here are the last 20 lines of the log:\n[ERROR] ... \n[INFO] BUILD FAILURE"
```

---

✅ **Key Points**:
- Only `summary` and `exit_code` go into LLM context.  
- Full logs are stored externally.  
- `fetch_build_log` provides on-demand access to snippets.  
