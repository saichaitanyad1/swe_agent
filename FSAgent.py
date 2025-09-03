# file: fs_agent_ctx_workspace.py
# pip install google-adk

from __future__ import annotations
import os, glob, hashlib, time
from typing import Optional, List
from dataclasses import dataclass

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool, ToolContext
from google.adk.tools.base_toolset import BaseToolset
from google.adk.context import ReadonlyContext
from google.genai import types

# ---------- Safety helpers ----------
@dataclass
class SafeWorkspace:
    root: str
    allow_exts: List[str]
    max_bytes: int

    def real_root(self) -> str:
        return os.path.realpath(self.root)

    def resolve(self, rel_path: str) -> str:
        if rel_path.startswith(("~", "/", "\\")):
            raise ValueError("Use a relative path under the workspace root.")
        abs_target = os.path.realpath(os.path.join(self.real_root(), rel_path))
        if not (abs_target == self.real_root() or abs_target.startswith(self.real_root() + os.sep)):
            raise ValueError("Path escapes the workspace root.")
        return abs_target

    def check_ext(self, path: str) -> None:
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        if self.allow_exts and ext not in self.allow_exts:
            raise ValueError(f"Extension '.{ext}' not allowed in this workspace.")

    @staticmethod
    def sha256_bytes(data: bytes) -> str:
        h = hashlib.sha256(); h.update(data); return h.hexdigest()

# ---------- Toolset using ToolContext.state ----------
class FileSystemToolset(BaseToolset):
    """
    Reads workspace_root from ToolContext.state['workspace_root'].
    Allows changing the workspace per-session without exposing it as a tool arg.
    """

    def __init__(self, allow_exts: List[str], max_bytes: int):
        self.allow_exts = allow_exts
        self.max_bytes = max_bytes
        self._tools = [
            FunctionTool(func=self.fs_list_files),
            FunctionTool(func=self.fs_read_file),
            FunctionTool(func=self.fs_upsert_file),
            FunctionTool(func=self.fs_replace_in_file),
            FunctionTool(func=self.fs_append_file),
            FunctionTool(func=self.fs_move_file),
            FunctionTool(func=self.fs_delete_file),
        ]

    async def get_tools(self, readonly_context: Optional[ReadonlyContext] = None):
        return self._tools

    # --- internal helper ---
    def _get_ws(self, tool_context: ToolContext) -> SafeWorkspace:
        root = tool_context.state.get("workspace_root")
        if not root:
            raise ValueError("workspace_root not set in ToolContext.state")
        return SafeWorkspace(root=root, allow_exts=self.allow_exts, max_bytes=self.max_bytes)

    # --- tools ---
    def fs_list_files(self, pattern: str, max_results: int, tool_context: ToolContext) -> dict:
        ws = self._get_ws(tool_context)
        hits = glob.glob(os.path.join(ws.real_root(), pattern), recursive=True)
        files = [os.path.relpath(p, ws.real_root()) for p in hits[:max(0, max_results)] if os.path.isfile(p)]
        return {"status": "success", "files": files}

    def fs_read_file(self, path: str, tool_context: ToolContext) -> dict:
        try:
            ws = self._get_ws(tool_context)
            abs_p = ws.resolve(path)
            if not os.path.isfile(abs_p):
                return {"status": "error", "error_message": f"File not found: {path}"}
            b = open(abs_p, "rb").read()
            if len(b) > ws.max_bytes:
                return {"status": "error", "error_message": f"File too large (> {ws.max_bytes} bytes)."}
            text = b.decode("utf-8", errors="replace")
            sha = SafeWorkspace.sha256_bytes(b)
            tool_context.state["temp:last_read_path"] = path
            tool_context.state["temp:last_read_sha256"] = sha
            return {"status": "success", "path": path, "content": text, "sha256": sha, "size": len(b)}
        except Exception as e:
            return {"status": "error", "error_message": str(e)}

    def fs_upsert_file(self, path: str, content: str, expected_sha256: str, overwrite: bool, create_parents: bool, tool_context: ToolContext) -> dict:
        try:
            ws = self._get_ws(tool_context)
            abs_p = ws.resolve(path)
            ws.check_ext(abs_p)
            if create_parents:
                os.makedirs(os.path.dirname(abs_p), exist_ok=True)

            exists = os.path.exists(abs_p)
            new_bytes = content.encode("utf-8")
            if len(new_bytes) > ws.max_bytes:
                return {"status": "error", "error_message": f"Write exceeds {ws.max_bytes} bytes limit."}

            if exists:
                cur = open(abs_p, "rb").read()
                cur_sha = SafeWorkspace.sha256_bytes(cur)
                if expected_sha256 and expected_sha256 != cur_sha:
                    return {"status": "conflict", "current_sha256": cur_sha}
                if not expected_sha256 and not overwrite:
                    return {"status": "error", "error_message": "File exists; set overwrite=True or supply expected_sha256."}
                open(abs_p + f".bak.{time.strftime('%Y%m%d-%H%M%S')}", "wb").write(cur)

            tmp = abs_p + ".tmp"
            with open(tmp, "wb") as f:
                f.write(new_bytes)
            os.replace(tmp, abs_p)
            return {
                "status": "success",
                "path": path,
                "sha256": SafeWorkspace.sha256_bytes(new_bytes),
                "created": (not exists),
            }
        except Exception as e:
            return {"status": "error", "error_message": str(e)}

    def fs_replace_in_file(self, path: str, find: str, replace: str, count: int, create_backup: bool, tool_context: ToolContext) -> dict:
        try:
            ws = self._get_ws(tool_context)
            abs_p = ws.resolve(path)
            ws.check_ext(abs_p)
            if not os.path.isfile(abs_p):
                return {"status": "error", "error_message": f"File not found: {path}"}
            b = open(abs_p, "rb").read()
            if len(b) > ws.max_bytes:
                return {"status": "error", "error_message": f"File too large (> {ws.max_bytes} bytes)."}
            text = b.decode("utf-8", errors="strict")
            # do controlled replacement count
            total = text.count(find)
            n = total if count < 0 else min(count, total)
            new_text = text.replace(find, replace, n)
            if create_backup:
                open(abs_p + f".bak.{time.strftime('%Y%m%d-%H%M%S')}", "wb").write(b)
            open(abs_p, "wb").write(new_text.encode("utf-8"))
            return {"status": "success", "path": path, "replacements": n, "sha256": SafeWorkspace.sha256_bytes(new_text.encode("utf-8"))}
        except UnicodeDecodeError:
            return {"status": "error", "error_message": "File is not valid UTF-8 text."}
        except Exception as e:
            return {"status": "error", "error_message": str(e)}

    def fs_append_file(self, path: str, content: str, create_parents: bool, tool_context: ToolContext) -> dict:
        try:
            ws = self._get_ws(tool_context)
            abs_p = ws.resolve(path)
            ws.check_ext(abs_p)
            if create_parents:
                os.makedirs(os.path.dirname(abs_p), exist_ok=True)
            existed = os.path.exists(abs_p)
            with open(abs_p, "ab") as f:
                f.write(content.encode("utf-8"))
            new_b = open(abs_p, "rb").read()
            return {"status": "success", "path": path, "sha256": SafeWorkspace.sha256_bytes(new_b), "created": (not existed)}
        except Exception as e:
            return {"status": "error", "error_message": str(e)}

    def fs_move_file(self, src: str, dst: str, allow_overwrite: bool, create_parents: bool, tool_context: ToolContext) -> dict:
        try:
            ws = self._get_ws(tool_context)
            abs_src = ws.resolve(src)
            abs_dst = ws.resolve(dst)
            ws.check_ext(abs_src); ws.check_ext(abs_dst)
            if not os.path.isfile(abs_src):
                return {"status": "error", "error_message": f"Source not found: {src}"}
            if os.path.exists(abs_dst) and not allow_overwrite:
                return {"status": "error", "error_message": "Destination exists; set allow_overwrite=True."}
            if create_parents:
                os.makedirs(os.path.dirname(abs_dst), exist_ok=True)
            os.replace(abs_src, abs_dst)
            return {"status": "success", "path": dst}
        except Exception as e:
            return {"status": "error", "error_message": str(e)}

    def fs_delete_file(self, path: str, require_exists: bool, tool_context: ToolContext) -> dict:
        try:
            ws = self._get_ws(tool_context)
            abs_p = ws.resolve(path)
            if os.path.isdir(abs_p):
                return {"status": "error", "error_message": "Refusing to delete a directory."}
            if not os.path.exists(abs_p) and require_exists:
                return {"status": "error", "error_message": f"File not found: {path}"}
            if os.path.exists(abs_p):
                os.remove(abs_p)
            return {"status": "success", "path": path}
        except Exception as e:
            return {"status": "error", "error_message": str(e)}

# ---------- Agent wiring ----------
ALLOWED_CODE_EXTS = [
    "py","java","kt","kts","js","jsx","ts","tsx","c","h","cpp","hpp","cs","go",
    "rb","rs","sql","yaml","yml","toml","ini","xml","json","gradle","md","sh","ps1"
]

def build_agent() -> Agent:
    tools = FileSystemToolset(allow_exts=ALLOWED_CODE_EXTS, max_bytes=2_000_000)
    instruction = """
You are a coding assistant with file tools.
- The workspace root is provided via ToolContext.state['workspace_root'].
- Use only relative paths under that root.
- Read a file first and reuse its sha256 when doing full-file writes, unless overwrite=True.
- Use fs_list_files for discovery; avoid binary edits.
"""
    return Agent(
        name="FileOpsAgent",
        model="gemini-2.0-flash",
        description="File ops with workspace supplied via ToolContext.state.",
        instruction=instruction,
        tools=[tools],
    )

# ---------- Minimal runner: sets workspace_root in state ----------
APP_NAME = "file_ops_ctx_app"
USER_ID = "local_user"
SESSION_ID = "dev_session"

async def main():
    session = InMemorySessionService()
    await session.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

    agent = build_agent()
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session)

    # Set the workspace root for this session (can be changed later)
    workspace_root = os.getenv("ADK_WORKSPACE_ROOT", os.getcwd())

    # IMPORTANT: seed the tool context state for this session
    # Runner exposes a per-session tool state you can mutate like this:
    runner.tool_context.state["workspace_root"] = workspace_root

    # Demo task
    prompt = "Create src/hello.py that prints 'hello, world', then read it back."
    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
    for e in events:
        if e.is_final_response():
            print(e.content.parts[0].text)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
